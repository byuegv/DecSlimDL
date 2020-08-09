package AccurateML.nonLinearRegression

import java.io.File

import AccurateML.blas.{ZFBLAS, ZFUtils}
import AccurateML.lsh.{ZFHash, IncreSVD}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.{CollectionAccumulator, LongAccumulator}
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random



class ZFNNLSHPart3(
                    fitmodel: NeuralNetworkModel,
                    data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])],
                    r: Double,
                    redisHost: String
                  ) extends Serializable {

  var nnModel: NeuralNetworkModel = fitmodel
  var dim = fitmodel.getDim()
  var train: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = data
  //  var trainN: Int = train.count().toInt
  var numFeature: Int = train.first()._1(0)(0).features.size
  val nnRatio: Double = r
  var bcWeights: Broadcast[BDV[Double]] = null
  var nnItN = -1
  val mapT = data.sparkContext.longAccumulator
  val zipN = data.sparkContext.longAccumulator
  val selectZipN = data.sparkContext.longAccumulator


  /**
    * Return the objective function dimensionality which is essentially the model's dimensionality
    */
  def getDim(): Int = {
    return this.dim
  }


  /**
    * This method is inherited by Breeze DiffFunction. Given an input vector of weights it returns the
    * objective function and the first order derivative.
    * It operates using treeAggregate action on the training pair data.
    * It is essentially the same implementation as the one used for the Stochastic Gradient Descent
    * Partial subderivative vectors are calculated in the map step
    * val per = fitModel.eval(w, feat)
    * val gper = fitModel.grad(w, feat)
    * and are aggregated by summation in the reduce part.
    */
  def zfNNMap(pit: Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])], tempRecord: CollectionAccumulator[Double]): Iterator[(Double, BDV[Double], Double)] = {
    var nnMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
    val objectData = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
    while (pit.hasNext) {
      objectData += pit.next()
    }
    val partZipN = objectData.size
    val partSelectZipN: Int = math.max((partZipN * nnRatio).toInt, 1)
    val chooseLshRound = 1 //set chooseRound
    val weights = bcWeights.value.toArray
    val weightsBDV = new BDV[Double](weights)
    //      val ans = new ArrayBuffer[(BDV[Double], Double, Int)]()
    val ansG1 = BDV.zeros[Double](dim) //g1
    var ansF1 = 0.0
    var ansN = 0.0


    val zipIndex = if (nnRatio == 2) Array.range(0, partZipN) // nnRatio == 1,if(nnRatio==2)相当于if(false)
    else {
      var diffIndexBuffer = new ArrayBuffer[(Int, Double, Double)]()
      for (i <- objectData.indices) {
        val zip = objectData(i)._1(0).last // lshRound = 1 第一层只有一个zip
        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        val gper = nnModel.grad(weightsBDV, feat)
        val g1 = 2.0 * (per - zip.label) * gper
        val norm: Double = math.sqrt(g1 dot g1)
        tempRecord.add(norm)
        //          diffIndexBuffer += Tuple3(i, gper.toArray.map(math.abs).sum, math.abs(per - zip.label))
        diffIndexBuffer += Tuple3(i, norm, 0.0)
        //          diffIndexBuffer += Tuple2(i, math.abs(per - zip.label))
      }
      diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1)
    }

    for (i <- 0 until partSelectZipN) {
      val zipi = zipIndex(i)

      //      {
      //        val zip = objectData(zipi)._1(0).last // lshRound = 1 第一层只有一个zip
      //      val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
      //        val per = nnModel.eval(weightsBDV, feat)
      //        val gper = nnModel.grad(weightsBDV, feat)
      //        val g1 = 2.0 * (per - zip.label) * gper
      //        val norm: Double = math.sqrt(g1 dot g1)
      //      }

      val chooseRound = {
        if (i < partSelectZipN) chooseLshRound else 0
      }
      val iter = objectData(zipi)._1(chooseRound).iterator
      while (iter.hasNext) {
        val point = iter.next()
        val feat: BDV[Double] = new BDV[Double](point.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        val gper = nnModel.grad(weightsBDV, feat)
        val f1 = 0.5 * Math.pow(point.label - per, 2)
        val g1 = 2.0 * (per - point.label) * gper


        //          ans += Tuple3(g1, f1, 1)
        ansF1 += f1
        ansG1 += g1
        ansN += 1
      }
    }


    mapT.add(System.currentTimeMillis() - nnMapT)
    jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
    zipN.add(partZipN)
    selectZipN.add(partSelectZipN)
    //    jedis.append("zipN", "," + partZipN)
    //    jedis.append("setN", "," + partSelectZipN)
    jedis.close()
    Array(Tuple3(ansF1, ansG1, ansN)).iterator
  }

  def zfNNMapBigNorm(pit: Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])],
                     tempRecord: CollectionAccumulator[Double],
                     usedNormDataN: LongAccumulator): Iterator[(Double, BDV[Double], Double)] = {

    var nnMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    val (pit1, pit2) = pit.duplicate
    val partZipN = pit1.size
    //    val partSelectZipN: Int = math.max((partZipN * nnRatio).toInt, 1)
    val objectData = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
    val rand = new Random()
    while (pit2.hasNext) {
      val temp = pit2.next()
      if (rand.nextDouble() < nnRatio)
        objectData += temp
    }


    val weights = bcWeights.value.toArray
    val weightsBDV = new BDV[Double](weights)
    val ansG1 = BDV.zeros[Double](dim) //g1
    var ansF1 = 0.0
    var ansN = 0.0


    val zipIndex = if (nnRatio == 2) Array.range(0, partZipN) // nnRatio == 1,if(nnRatio==2)相当于if(false)
    else {
      var diffIndexBuffer = new ArrayBuffer[(Int, Double, Double)]()
      for (i <- objectData.indices) {
        val zip = objectData(i)._1(0).last // lshRound = 1 第一层只有一个zip
        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        //        val gper = nnModel.grad(weightsBDV, feat)
        val C = 2.0 * (per - zip.label)
        val tempBound = 1E-5 * 1E-5 / C / C
        val tempGrad2 = nnModel.zfGrad2(weightsBDV, feat)
        val bigThanNorm: Boolean = if (C == 0 || tempGrad2 <= tempBound) false else true

        if (bigThanNorm)
          diffIndexBuffer += Tuple3(i, tempGrad2, 0.0)

      }
      //      diffIndexBuffer.toArray.map(_._1)
      val tempN = (diffIndexBuffer.size * 0.2).toInt
      diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1).slice(0, tempN)
    }

    for (i <- 0 until zipIndex.size) {
      //    for (i <- 0 until partSelectZipN) {
      val zipi = zipIndex(i)
      val iter = objectData(zipi)._1(1).iterator
      while (iter.hasNext) {
        val point = iter.next()
        val feat: BDV[Double] = new BDV[Double](point.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        val gper = nnModel.grad(weightsBDV, feat)
        val f1 = 0.5 * Math.pow(point.label - per, 2)
        val g1 = 2.0 * (per - point.label) * gper

        ansF1 += f1
        ansG1 += g1
        usedNormDataN.add(1)
        //        ansN += 1
      }
    }
    ansN = objectData.map(_._1(1).size).sum


    mapT.add(System.currentTimeMillis() - nnMapT)
    jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
    zipN.add(partZipN)
    selectZipN.add(objectData.size)
    jedis.close()
    Array(Tuple3(ansF1, ansG1, ansN)).iterator
  }

  def calculate(weights: BDV[Double], iN: Int): (BDV[Double], Double, Int) = {
    assert(dim == weights.length)
    nnItN = iN
    bcWeights = train.context.broadcast(weights)

    val fitModel: NonlinearModel = nnModel
    val n: Int = dim
    val bcDim = train.context.broadcast(dim)
    val tempRecord = this.data.sparkContext.collectionAccumulator[Double]
    val usedNormDataN = this.data.sparkContext.longAccumulator
    val mapData = train.mapPartitions(this.zfNNMapBigNorm(_, tempRecord, usedNormDataN))
    val (lossSum, gradientSum, miniBatchSize) = mapData.treeAggregate(0.0, BDV.zeros[Double](n), 0.0)(
      seqOp = (c, v) => (c, v) match {
        case ((f, grad, n), (af, ag, an)) =>
          grad += ag
          (f + af, grad, n + an)
      },
      combOp = (u1, u2) => (u1, u2) match {
        case ((f1, grad1, n1), (f2, grad2, n2)) =>
          grad1 += grad2
          (f1 + f2, grad1, n1 + n2)
      }
    )
    //        val tempP = tempRecord.value.toList.sortWith(_ > _)
    //    val tempN = 50
    //    val lowBound = 1E-2
    //    val lowN = tempRecord.value.filter(_ < lowBound).size
    //    println("ggRecord < " + lowBound + ",\t" + lowN + " / " + tempRecord.value.size() + "\t," + lowN.toDouble / tempRecord.value.size())
    //    println("it " + iN + " ggRecord:\t" + tempP.slice(0, tempN).mkString(",") + "\t...\t" + tempP.slice(tempP.size - tempN, tempP.size).mkString(","))
    println("usedNormDataN: " + usedNormDataN.value / 10000.0 + "\t ggRecord AVG: " + tempRecord.value.toList.sum / tempRecord.value.size())
    return (gradientSum, lossSum, miniBatchSize.toInt)
  }


}

object ZFNNLSHPart3 {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val stepSize: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val hiddenNodesN: Int = args(3).toInt //5
    val nnItN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Array[Double] = args(7).split(",").map(_.toDouble)

    val weightsPath = args(8)
    val isSparse = args(9).toBoolean

    val itqbitN = args(10).toInt
    val itqitN = args(11).toInt
    val itqratioN = args(12).toInt //from 1 not 0
    val minPartN = args(13).toInt
    val upBound = args(14).toInt
    val redisHost = args(15) //"172.18.11.97"
    val splitN = 2 //args(15).toDouble


    val splitChar = ",|\\s+"

    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    val w0 = if (initW == -1) {
      val iter = Source.fromFile(new File(weightsPath)).getLines()
      val weights = iter.next().split(",").map(_.toDouble)
      new BDV(weights)
    } else BDV(Array.fill(dim)(initW))


    val data = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features))
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }

    val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
    val train = if (testPath.size > 3) data.cache() else splits(0).cache()
    val test = if (testPath.size > 3) {
      println("testPath,\t" + testPath)
      if (!isSparse) {
        sc.textFile(testPath).map(line => {
          val vs = line.split(splitChar).map(_.toDouble)
          val features = vs.slice(0, vs.size - 1)
          LabeledPoint(vs.last, Vectors.dense(features))
        })
      } else {
        MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
      }
    } else splits(1)

    val jedis = new Jedis(redisHost)
    jedis.flushAll()
    val oHash = new ZFHash(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse, redisHost, sc)
    val objectData = train.mapPartitions(oHash.zfHashMap).persist(StorageLevel.MEMORY_AND_DISK)


    val on = objectData.count()
    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",testPart," + test.getNumPartitions)

    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val HashMapT = jedis.get("HashMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

    println("objectDataN," + on + ",itqbitN," + itqbitN + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN + ",splitN," + splitN)
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + oHash.mapT.value + "," + HashMapT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


    val ratioL = test100
    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Long]()
    for (r <- ratioL) {
      val jedis = new Jedis(redisHost)
      jedis.flushAll()
      val nnRatio = r / 100.0
      val train = objectData
      var trainN = 0.0
      val model: NeuralNetworkModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNLSHPart3 = new ZFNNLSHPart3(model, train, nnRatio, redisHost)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to nnItN) {
        val (g1, f1, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val itStepSize = stepSize / itTrainN / math.sqrt(i) //this is stepSize for each iteration
        w -= itStepSize * g1
        trainN += itTrainN
      }
      trainN /= nnItN
      vecs += Vectors.dense(w.toArray)
      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point.features.toArray))
        (point.label, prediction)
      }.map { case (v, p) => math.pow((v - p), 2) }.mean()

      println()
      val zipN = modelTrain.zipN.value //jedis.get("zipN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      val setN = modelTrain.selectZipN.value //jedis.get("setN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

      jedis.close()

      println(",nnRatio," + nnRatio + ",nnItN," + nnItN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeature," + numFeature)
      println("zipN," + zipN + ",setN," + setN + ",allUsedPointN," + trainN + ",nnMapT," + modelTrain.mapT.value + "," + nnMapT.sum)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      mesb += MSE
      nntimesb += modelTrain.mapT.value // nnMapT.sum

    }

    val n = vecs.length

    println()
    println(this.getClass.getName + ",step," + stepSize + ",data," + dataPath)
    println("ratio,MSE,nnMapT")
    for (i <- vecs.toArray.indices) {
      println(ratioL(i) / 100.0 + "," + mesb(i) + "," + nntimesb(i))
    }

  }


}
