package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.{CollectionAccumulator, LongAccumulator}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

//import redis.clients.jedis.Jedis
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFNNPartMBGD(fitmodel: NonlinearModel, xydata: RDD[LabeledPoint], redisHost: String) extends Serializable {
  var model: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[LabeledPoint] = xydata
  var m: Int = data.count().toInt
  var n: Int = data.first().features.size
  val mapT = data.sparkContext.longAccumulator


  def resetData(xydata: RDD[LabeledPoint]): Unit = {
    data = xydata
    m = data.count().toInt
    n = data.first().features.size
  }

  /**
    * Return the objective function dimensionality which is essentially the model's dimensionality
    */
  def getDim(): Int = {
    return this.dim
  }


  /**
    * caculate one hidden nodes w-grad, for (i<-0 to 0)
    *
    * @param w
    * @param x
    * @param normBound
    * @return
    */
  def zfGrad2(w: BDV[Double], x: BDV[Double], normBound: Double): Boolean = {
    assert(x.size == n)
    assert(w.size == dim)

    var sum = 0.0

    //    val gper: BDV[Double] = BDV.zeros(dim) // (n+2)*nodes
    val nodes = dim / (n + 2)

    for (i <- 0 to 1) {
      // nodes - 1
      var arg: Double = 0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      val sig: Double = 1.0 / (1.0 + Math.exp(-arg))
      //      gper(i * (n + 2) + n + 1) = sig
      //      gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
      sum += math.pow(sig, 2)
      sum += math.pow(w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)

      for (j <- 0 to n - 1) {
        //        gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
        sum += math.pow(x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)
      }
    }
    //    return gper;
    return sum > normBound
  }

  /**
    * calculate fratio part of w-grad
    *
    * @param w
    * @param x
    * @param normBound
    * @return
    */
  def zfGrad3(w: BDV[Double], x: BDV[Double], normBound: Double): Boolean = {
    assert(x.size == n)
    assert(w.size == dim)

    var sum = 0.0

    //    val gper: BDV[Double] = BDV.zeros(dim) // (n+2)*nodes
    val nodes = dim / (n + 2)
    val fratio = 0.5
    val rand = new Random()

    for (i <- 0 to nodes - 1) {
      // nodes - 1
      var arg: Double = 0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      val sig: Double = 1.0 / (1.0 + Math.exp(-arg))
      //      gper(i * (n + 2) + n + 1) = sig
      //      gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
      if (rand.nextDouble() < fratio)
        sum += math.pow(sig, 2)
      if (rand.nextDouble() < fratio)
        sum += math.pow(w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)

      for (j <- 0 to n - 1) {
        //        gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
        if (rand.nextDouble() < fratio)
          sum += math.pow(x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)
      }
    }
    //    return gper;
    return sum > normBound
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


  def calculate(weights: BDV[Double], ratio: Double, bigGradientN: LongAccumulator, itBigGradientValue: CollectionAccumulator[Double]): (Double, BDV[Double], Double) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)


    val mapData: RDD[(Double, BDV[Double], Double)] = data.mapPartitions(pit => {
      val nnMapT = System.currentTimeMillis()
      val rand = new Random()
      val randPit = pit.filter(vec => rand.nextDouble() <= ratio)
      val ansG1 = BDV.zeros[Double](bcDim.value) //g1
      var ansF1 = 0.0
      var ansN = 0.0
      randPit.foreach { inc =>
        val label = inc.label
        val features = inc.features
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper

        val cancel = g1.dot(g1)
        if (cancel > 1E-5) {
          bigGradientN.add(1)
          itBigGradientValue.add(cancel)
        }
        ansG1 += g1
        ansF1 += f1
        ansN += 1
      }

      mapT.add((System.currentTimeMillis() - nnMapT))

      //      Sorting.quickSort(ggf)(Ordering.by[(Double, BDV[Double], Double), Double](-_._1))
      //      ggf.slice(0, (partN * ratio).toInt).toIterator
      Array(Tuple3(ansF1, ansG1, ansN)).toIterator
    })
    val (allF, allGrad, allN) = mapData.treeAggregate((0.0, BDV.zeros[Double](n), 0.0))(
      seqOp = (c, v) => (c, v) match {
        case ((f, g, n), (af, ag, an)) =>
          g += ag
          (f + af, g, n + an)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((f1, g1, n1), (f2, g2, n2)) =>
          g1 += g2
          (f1 + f2, g1, n1 + n2)
      })

    return (allF, allGrad, allN)
  }

  def calculateBigNorm(weights: BDV[Double], lowBound: Double, ratio: Double, usedNormDataN: LongAccumulator, norms: CollectionAccumulator[Double]): (Double, BDV[Double], Double) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)

    val mapData: RDD[(Double, BDV[Double], Double)] = data.mapPartitions(pit => {
      val rand = new Random()
      val randPit = pit.filter(vec => rand.nextDouble() <= ratio)
      val nnMapT = System.currentTimeMillis()
      val ansG1 = BDV.zeros[Double](bcDim.value) //g1
      var ansF1 = 0.0
      var ansN = 0.0
      randPit.foreach { inc =>
        val label = inc.label
        val features = inc.features
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)

        val C = 2.0 * (per - label)
        //        val tempBound = 0
        val tempBound = 1E-5 * 1E-5 / C / C
        val bigThanNorm: Boolean = if (C == 0 || zfGrad2(w, feat, tempBound) == false) false else true
        //        val bigThanNorm = true

        if (bigThanNorm) {
          val gper = fitModel.grad(w, feat)
          val f1 = 0.5 * Math.pow(label - per, 2)
          val g1 = 2.0 * (per - label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          norms.add(norm)

          ansG1 += g1
          ansF1 += f1
          usedNormDataN.add(1)
        } else {
          //statistic norm
          val gper = fitModel.grad(w, feat)
          val g1 = 2.0 * (per - label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          norms.add(norm)

        }
        ansN += 1

      }
      mapT.add((System.currentTimeMillis() - nnMapT))
      Array(Tuple3(ansF1, ansG1, ansN)).toIterator
    })
    val (allF, allGrad, allN) = mapData.treeAggregate((0.0, BDV.zeros[Double](n), 0.0))(
      seqOp = (c, v) => (c, v) match {
        case ((f, g, n), (af, ag, an)) =>
          g += ag
          (f + af, g, n + an)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((f1, g1, n1), (f2, g2, n2)) =>
          g1 += g2
          (f1 + f2, g1, n1 + n2)
      })


    return (allF, allGrad, allN)
  }


}

object ZFNNPartMBGD {
  /**
    * this algorithm ( one_layer neural_network ) is sensitive to w initial. the number of w (args) is big (approximate featurnN * hiddenN)
    *
    * @param args
    */
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val stepSize: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val hiddenNodesN: Int = args(3).toInt //5
    val itN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Array[Double] = args(7).split(",").map(_.toDouble)

    val weightsPath = args(8)
    val isSparse = args(9).toBoolean
    val minPartN = args(10).toInt
    val redisHost = args(11) //"172.18.11.97"
    val foldN = args(12).toInt

    val splitChar = ",|\\s+"
    val ratioL = test100


    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Long]()

    for (r <- ratioL) {
      val dataTxt = sc.textFile(dataPath, minPartN) // "/Users/zhangfan/Documents/nonlinear.f10.n100.h5.txt"
      val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
      val w0 = if (initW == -1) {
        val iter = Source.fromFile(new File(weightsPath)).getLines()
        val weights = iter.next().split(",").map(_.toDouble)
        new BDV(weights)
      } else BDV(Array.fill(dim)(initW))
      val ratio = r / 100.0
      val data = if (!isSparse) {
        dataTxt.map(line => {
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

      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNPartMBGD = new ZFNNPartMBGD(model, train, redisHost)
      val w = w0.copy
      val hissb = new StringBuilder()
      var aTime = 0L
      var itTrainN = 0.0

      for (it <- 1 to itN) {
        itTrainN = 0
        val itBigGradientN = sc.longAccumulator
        val itBigGradientValue = sc.collectionAccumulator[Double]
        val itT = System.currentTimeMillis()
        val splits = train.randomSplit(Array.fill(foldN)(1.0 / foldN.toDouble))
        for (f <- 0 until foldN) {
          val foldTrain = splits(f)
          foldTrain.repartition(minPartN)
          //          foldTrain.persist(StorageLevel.MEMORY_AND_DISK)
          modelTrain.resetData(foldTrain)
          //            val usedNormDataN = sc.longAccumulator
          //            val normS = sc.collectionAccumulator[Double]
          //            val (f1, g1, itTrainN) = modelTrain.calculateBigNorm(w, -1, ratio, usedNormDataN, normS)
          val (f1, g1, foldTrainN) = modelTrain.calculate(w, ratio, itBigGradientN, itBigGradientValue)
          //          hissb.append("," + f1 / foldTrainN)
          val itStepSize = stepSize / foldTrainN / math.sqrt(it) //this is stepSize for each iteration
          w -= itStepSize * g1
          itTrainN += foldTrainN
          //          foldTrain.unpersist()

        }
        aTime += (System.currentTimeMillis() - itT)

        val gvalue = itBigGradientValue.value.toList.sortWith(_ < _)
        val gmin = gvalue(0)
        val gmax = gvalue.last
        val sliceN = 20
        val tempe = Array(-5, -3, -1) ++ Array.range(0, 17, 1)
        val iMax = tempe.map(i => {
          math.pow(10, i)
        })
        val iCount = new Array[Int](sliceN)
        var tempi = 0
        for (v <- gvalue) {
          while (v > iMax(tempi) && (tempi < (sliceN - 1))) tempi += 1
          iCount(tempi) += 1
        }
        println(it + "Gradient Value Count:  " + iCount.sum + "\t gmin: " + gmin)
        println(iMax.mkString(","))
        println()
        println(iCount.mkString(","))
        println()

        /** 10次迭代计算一次mse */
        if (it % 1 == 0) {
          val itMse = test.map { point =>
            val prediction = model.eval(w, new BDV[Double](point.features.toArray))
            (point.label, prediction)
          }.map { case (v, p) => math.pow((v - p), 2) }.mean()
          println(it + " : \tMSE: " + itMse + ",\tTime: " + aTime + ",\tMapT: " + modelTrain.mapT.value + ",\titBigGradientN: " + itBigGradientN.value + ", " + itTrainN + ", " + itBigGradientN.value.toDouble / itTrainN)


        }
      }


//      val cancel: Array[Double] = test.map(point => model.eval(w, new BDV[Double](point.features.toArray))).collect().sortWith(_<_)
//      println("yp: \t " + cancel.slice(0,10).mkString(",")+",,,"+cancel.slice(cancel.size-10,cancel.size).mkString(","))

      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point.features.toArray))
        (point.label, prediction)
      }.map { case (v, p) => math.pow((v - p), 2) }.mean()

      val nnMapT = modelTrain.mapT.value //jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

      println()

      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",testPart," + test.getNumPartitions)
      println(",ratio," + ratio + ",itN," + itN + ",foldN," + foldN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + itTrainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeatures," + numFeature)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      println("nnMapT," + nnMapT)

      vecs += Vectors.dense(w.toArray)
      mesb += MSE
      nntimesb += nnMapT
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
