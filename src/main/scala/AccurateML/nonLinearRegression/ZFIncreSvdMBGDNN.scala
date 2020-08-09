package AccurateML.nonLinearRegression

import java.io.File

import AccurateML.blas.ZFUtils
import AccurateML.lsh.ZFHash
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

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

/**
  * Created by zhangfan on 16/11/17.
  */

class ZFIncreSvdMBGDNN(
                        fitmodel: NeuralNetworkModel,
                        data: RDD[Tuple2[LabeledPoint, Array[LabeledPoint]]],
                        r: Double,
                        redisHost: String,
                        minpartN: Int
                      ) extends Serializable {

  val minPartN = minpartN
  var nnModel: NeuralNetworkModel = fitmodel
  var dim = fitmodel.getDim()
  var train: RDD[Tuple2[LabeledPoint, Array[LabeledPoint]]] = data
  //  var trainN: Int = train.count().toInt
  var numFeature: Int = train.first()._1.features.size
  val nnRatio: Double = r
  var bcWeights: Broadcast[BDV[Double]] = null
  var nnItN = -1
  val mapT = data.sparkContext.longAccumulator
  val zipMapT = data.sparkContext.longAccumulator
  val zipN = data.sparkContext.longAccumulator
  val selectZipN = data.sparkContext.longAccumulator


  val cZipN = data.sparkContext.longAccumulator
  val cPointsN = data.sparkContext.longAccumulator
  val cPointsAfterFilterN = data.sparkContext.longAccumulator


  def resetData(xydata: RDD[Tuple2[LabeledPoint, Array[LabeledPoint]]]): Unit = {
    train = xydata
  }

  /**
    * Return the objective function dimensionality which is essentially the model's dimensionality
    */
  def getDim(): Int = {
    return this.dim
  }


  /**
    * 对全部zip排序,选取ratio部分的数据展开用其原始点更新gradient
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

      //      val chooseRound = {
      //        if (i < partSelectZipN) chooseLshRound else 0
      //      }
      val iter = objectData(zipi)._1(chooseLshRound).iterator
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

  //  /**
  //    * zip与其对应的原始点存储在一起,输入为pit,pit._1中0层存储压缩点,1层存储对应原始点
  //    * 从zip中随机选取ratio比例的zip;满足Gradient大于阈值的zip展开,用其对应原始点更新gradient
  //    */
  //  def zfNNMapBigNorm(pit: Iterator[Tuple2[LabeledPoint, Array[LabeledPoint]]],
  //                     usedNormDataN: LongAccumulator,
  //                     usedLittleNormDataN: LongAccumulator): Iterator[(Double, BDV[Double], Double)] = {
  //
  //    var nnMapT = System.currentTimeMillis()
  //
  //    val (pit1, pit2) = pit.duplicate
  //    val partZipN = pit1.size
  //    //    val partSelectZipN: Int = math.max((partZipN * nnRatio).toInt, 1)
  //    val objectData = new ArrayBuffer[Tuple2[LabeledPoint, Array[LabeledPoint]]]()
  //    val rand = new Random()
  //    while (pit2.hasNext) {
  //      val temp = pit2.next()
  //      if (rand.nextDouble() < nnRatio)
  //        objectData += temp
  //    }
  //
  //
  //    val weights = bcWeights.value.toArray
  //    val weightsBDV = new BDV[Double](weights)
  //    val ansG1 = BDV.zeros[Double](dim) //g1
  //    var ansF1 = 0.0
  //    var ansN = 0.0
  //
  //
  //    val zipIndex = if (nnRatio == 2) Array.range(0, partZipN) // nnRatio == 1,if(nnRatio==2)相当于if(false)
  //    else {
  //      var diffIndexBuffer = new ArrayBuffer[Int]()
  //      for (i <- objectData.indices) {
  //        val zip = objectData(i)._1(0).last // lshRound = 1 第一层只有一个zip
  //        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
  //        val per = nnModel.eval(weightsBDV, feat)
  //        //        val gper = nnModel.grad(weightsBDV, feat)
  //        val C = 2.0 * (per - zip.label)
  //        val tempBound = 1E-5 * 1E-5 / C / C
  //        val bigThanNorm: Boolean = if (C == 0 || nnModel.zfGrad2(weightsBDV, feat, tempBound) == false) false else true
  //        //        val g1 = 2.0 * (per - zip.label) * gper
  //        //        val norm: Double = math.sqrt(g1 dot g1)
  //        //        diffIndexBuffer += Tuple3(i, norm, 0.0)
  //        if (bigThanNorm)
  //          diffIndexBuffer += i
  //
  //      }
  //      diffIndexBuffer.toArray
  //      //      diffIndexBuffer.toArray.filter(_._2 > 1E-5).map(_._1)
  //      //      diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1)
  //    }
  //
  //    for (i <- 0 until zipIndex.size) {
  //      val zipi = zipIndex(i)
  //      val iter = objectData(zipi)._1(1).iterator
  //      while (iter.hasNext) {
  //        val point = iter.next()
  //        val feat: BDV[Double] = new BDV[Double](point.features.toArray)
  //        val per = nnModel.eval(weightsBDV, feat)
  //        val gper = nnModel.grad(weightsBDV, feat)
  //        val f1 = 0.5 * Math.pow(point.label - per, 2)
  //        val g1 = 2.0 * (per - point.label) * gper
  //
  //        ansF1 += f1
  //        ansG1 += g1
  //        usedNormDataN.add(1)
  //        if (g1.dot(g1) < 1E-5) {
  //          usedLittleNormDataN.add(1)
  //        }
  //      }
  //    }
  //    ansN = objectData.map(_._1(1).size).sum
  //
  //
  //    mapT.add(System.currentTimeMillis() - nnMapT)
  //    zipN.add(partZipN)
  //    selectZipN.add(objectData.size)
  //    Array(Tuple3(ansF1, ansG1, ansN)).iterator
  //  }

  def calculate(weights: BDV[Double], iN: Int, gradientN: LongAccumulator, bigGradientN: LongAccumulator): (BDV[Double], Double, Int) = {
    assert(dim == weights.length)
    nnItN = iN
    bcWeights = train.context.broadcast(weights)

    val fitModel: NonlinearModel = nnModel
    val n: Int = dim
    val bcDim = train.context.broadcast(dim)
    val randomAnsN = train.context.longAccumulator
    val mapData: RDD[(Double, BDV[Double], Double)] = train.mapPartitions(pit => {
      val nnMapT = System.currentTimeMillis()
      val rand = new Random()
      val objectData: Array[(LabeledPoint, Array[LabeledPoint])] = if (nnRatio == 1) pit.toArray else pit.toArray.filter(o => rand.nextDouble() < nnRatio)
      randomAnsN.add(objectData.map(_._2.size).sum)
      val ans = new ArrayBuffer[LabeledPoint]()
      val weights = bcWeights.value.toArray
      val weightsBDV = new BDV[Double](weights)

      val sortIndex = new ArrayBuffer[(Int, Double)]()
      for (i <- objectData.indices) {
        val zip = objectData(i)._1 // lshRound = 1 第一层只有一个zip
        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        //        val gper = nnModel.grad(weightsBDV, feat)
        val C = 2.0 * (per - zip.label)
        val tempBound = 1E-5 * 1E-5 / C / C
        val tempGrad2 = nnModel.zfGrad2(weightsBDV, feat)
        val bigThanNorm: Boolean = if (C == 0 || tempGrad2 <= tempBound) false else true
        if (bigThanNorm)
          sortIndex += Tuple2(i, tempGrad2)
      }

      val zipIndex = sortIndex.sortWith(_._2 > _._2).map(_._1).slice(0, (sortIndex.size * 1.0).toInt)
      zipIndex.foreach(i => {
        ans ++= objectData(i)._2
      })
      // cancel- statistic
      val czipN = objectData.size
      val cAllPointsN = objectData.map(_._2.size).sum
      val cAfterFilterPointsN = ans.size
      cZipN.add(czipN)
      cPointsN.add(cAllPointsN)
      cPointsAfterFilterN.add(cAfterFilterPointsN)

      // end-cancel
      zipMapT.add((System.currentTimeMillis() - nnMapT))
      ans.toIterator
    }).mapPartitions(pit => {
      val nnMapT = System.currentTimeMillis()
      val ansG1 = BDV.zeros[Double](bcDim.value) //g1
      var ansF1 = 0.0
      var ansN = 0.0
      pit.foreach { inc =>
        val label = inc.label
        val features = inc.features
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcWeights.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper
        if (g1.dot(g1) > 1E-5)
          bigGradientN.add(1)
        ansG1 += g1
        ansF1 += f1
        ansN += 1
      }
      mapT.add((System.currentTimeMillis() - nnMapT))
      Array(Tuple3(ansF1, ansG1, ansN)).toIterator
    }).repartition(minpartN).persist(StorageLevel.MEMORY_AND_DISK)

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
    gradientN.add(miniBatchSize.toLong)
    mapData.unpersist()

    return (gradientSum, lossSum, randomAnsN.value.toInt)
  }


}

object ZFIncreSvdMBGDNN {
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
    val foldN = args(16).toInt


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
    val train = if (testPath.size > 3) data.persist(StorageLevel.MEMORY_AND_DISK) else splits(0).persist(StorageLevel.MEMORY_AND_DISK)
    val test = if (testPath.size > 3) {
      println("testPath,\t" + testPath)
      if (!isSparse) {
        sc.textFile(testPath, minPartN).map(line => {
          val vs = line.split(splitChar).map(_.toDouble)
          val features = vs.slice(0, vs.size - 1)
          LabeledPoint(vs.last, Vectors.dense(features))
        })
      } else {
        MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
      }
    } else splits(1)

    println("trainN:" + train.count())
    var hashTime = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
    jedis.flushAll()
    val oHash = new ZFHash(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse, redisHost, sc)
    val objectData: RDD[(LabeledPoint, Array[LabeledPoint])] = train
      .mapPartitions(oHash.zfHashMap) //incrementalSVD
      .map(t3 => Tuple2(t3._1(0).last, t3._1(1).toArray)).persist(StorageLevel.MEMORY_AND_DISK)

    val on = objectData.count()
    hashTime = System.currentTimeMillis() - hashTime
    train.unpersist()
    println()
    println("hashTime: " + hashTime)
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",testPart," + test.getNumPartitions)

    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

    println("objectDataN," + on + ",itqbitN," + itqbitN + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN + ",splitN," + splitN)
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + oHash.mapT.value + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
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
      val modelTrain: ZFIncreSvdMBGDNN = new ZFIncreSvdMBGDNN(model, train, nnRatio, redisHost, minPartN)
      val hissb = new StringBuilder()
      val w = w0.copy
      var aTime = 0L
      var itTrainN = 0.0
      for (it <- 1 to nnItN) {
        itTrainN = 0
        val itGradientN = sc.longAccumulator
        val itBigGradientN = sc.longAccumulator
        val itT = System.currentTimeMillis()
        val splits = train.randomSplit(Array.fill(foldN)(1.0 / foldN))

        for (f <- 0 until foldN) {
          val foldTrain = splits(f)
          foldTrain.repartition(minPartN)
          //          foldTrain.persist(StorageLevel.MEMORY_AND_DISK)
          modelTrain.resetData(foldTrain)

          val (g1, f1, foldTrainN) = modelTrain.calculate(w, it, itGradientN, itBigGradientN)
          //          hissb.append("," + f1 / itTrainN)
          val itStepSize = stepSize / foldTrainN / math.sqrt(it) //this is stepSize for each iteration
          w -= itStepSize * g1

          itTrainN += foldTrainN
          //          foldTrain.unpersist()
        }
        aTime += (System.currentTimeMillis() - itT)

        if (it % 10 == 0) {
          val itMse = test.map { point =>
            val prediction = model.eval(w, new BDV[Double](point.features.toArray))
            (point.label, prediction)
          }.map { case (v, p) => math.pow((v - p), 2) }.mean()
          println(it + " : \tMSE: " + itMse + ",\tTime: " + aTime +
            ",\tzipMapT: " + modelTrain.zipMapT.value + ",\tMapT: " + modelTrain.mapT.value + "\t," + (modelTrain.zipMapT.value.toDouble / (modelTrain.zipMapT.value + modelTrain.mapT.value)) +
            ",\titBigGradientN: " + itBigGradientN.value + ", " + itGradientN.value + ", " + itBigGradientN.value.toDouble / itGradientN.value.toDouble)
          println("zipN," + modelTrain.cZipN.value + " ,pointsN," + modelTrain.cPointsN.value + ", pointsAfterFilterN," + modelTrain.cPointsAfterFilterN.value)
        }
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

      jedis.close()

      println(",nnRatio," + nnRatio + ",nnItN," + nnItN + ",foldN," + foldN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeature," + numFeature)
      println("zipN," + zipN + ",setN," + setN + ",allUsedPointN," + trainN + ",nnMapT," + modelTrain.mapT.value)
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
