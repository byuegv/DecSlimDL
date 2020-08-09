package AccurateML.nonLinearRegression

import java.io.File

import AccurateML.blas.ZFBLAS
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.lendap.spark.lsh.ZFLshRoundRDD
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 17/11/1.
  */
class ZFNNLshKmeansMBGD {

}

object ZFNNLshKmeansMBGD {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")



    ///Users/zhangfan/Documents/data/a9a - 0 0 10,50,100 true 123 1


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
    val minPartN = args(10).toInt
    val foldN = args(11).toInt

    val lshPerBucketN = args(12).toInt //should be cancel
    val lshRoundN = args(13).toInt
    val kmeansItN = args(14).toInt
    val redisHost = args(15)

    println("lshPerBucketN: " + lshPerBucketN + "\t lshRoundN: " + lshRoundN + "\t kmeansItN:" + kmeansItN)
    val splitChar = "\\s+|,"
    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    val w0 = if (initW == -1) {
      val iter = Source.fromFile(new File(weightsPath)).getLines()
      val weights = iter.next().split(",").map(_.toDouble)
      new BDV(weights)
    } else BDV(Array.fill(dim)(initW))

    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features))
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }
    val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
    val train = if (testPath.size > 3) data else splits(0)
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


    //zip-----lsh+kmeans
    val zipTime = System.currentTimeMillis()
    val zflshRoundRdd = new ZFLshRoundRDD(train, numFeature, lshRoundN, lshPerBucketN)
    val zipArray: RDD[Array[LabeledPoint]] = zflshRoundRdd.zfRoundLsh()
    println("zipArray: " + zipArray.count())
    println("lshTime: " + (System.currentTimeMillis() - zipTime))
    val bigZip: RDD[Array[LabeledPoint]] = zipArray.filter(_.size > lshPerBucketN / 2)
    val litZip0: Array[LabeledPoint] = zipArray.filter(_.size <= lshPerBucketN / 2).flatMap(arr => arr).collect()
    val litZip: RDD[Array[LabeledPoint]] = if (litZip0.size <= lshPerBucketN) {
      sc.parallelize(Array(litZip0))
    } else {
      val K = litZip0.size / lshPerBucketN
      val kmeansData: RDD[Vector] = sc.parallelize(litZip0.map(_.features)).repartition(minPartN)
      val kmodel = KMeans.train(kmeansData, K, kmeansItN - 1)
      val ans = new Array[ArrayBuffer[LabeledPoint]](K)
      for (k <- 0 until K) {
        ans(k) = new ArrayBuffer[LabeledPoint]()
      }
      litZip0.foreach(lp => {
        val key = kmodel.predict(lp.features)
        ans(key) += lp
      })

      sc.parallelize(ans.map(_.toArray))
    }
    println("filter + kmeans: " + (System.currentTimeMillis() - zipTime))

    val zipData: RDD[(LabeledPoint, Array[LabeledPoint])] = {
      if (litZip0.size > 0)
        bigZip.union(litZip)
      else
        bigZip
    }.map(lps => {
      val zipfea = Vectors.zeros(numFeature)
      var ziplabel = 0.0
      lps.foreach(lp => {
        ZFBLAS.axpy(1.0, lp.features, zipfea)
        ziplabel += lp.label
      })
      val divn = 1.0 / lps.size
      ZFBLAS.scal(divn, zipfea)
      ziplabel *= divn
      val ans: (LabeledPoint, Array[LabeledPoint]) = Tuple2(new LabeledPoint(ziplabel, zipfea), lps)
      ans
    }).persist(StorageLevel.MEMORY_AND_DISK)

    val cancelzN = zipData.map(_._2.size).collect().sortWith(_ < _)
    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))
    val realPerBucketN = (cancelzN.sum / cancelzN.size.toDouble).toInt
    println("zipN:\t" + cancelzN.size + "\t" + realPerBucketN + "\t" + cancelzN.slice(0, 10).mkString(",") + "\t" + cancelzN.slice(cancelzN.size - 10, cancelzN.size))

    val ratioL = test100
    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Long]()
    for (r <- ratioL) {
      val jedis = new Jedis(redisHost)
      jedis.flushAll()
      val nnRatio = r / 100.0
      val train = zipData
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
          modelTrain.resetData(foldTrain)

          val (g1, f1, foldTrainN) = modelTrain.calculate(w, it, itGradientN, itBigGradientN)
          //          hissb.append("," + f1 / itTrainN)
          val itStepSize = stepSize / foldTrainN / math.sqrt(it) //this is stepSize for each iteration
          w -= itStepSize * g1

          itTrainN += foldTrainN
        }
        aTime += (System.currentTimeMillis() - itT)

        if (it % 1 == 0) {
          val itMse = test.map { point =>
            val prediction = model.eval(w, new BDV[Double](point.features.toArray))
            (point.label, prediction)
          }.map { case (v, p) => math.pow((v - p), 2) }.mean()
          println(it + " : \tMSE: " + itMse + ",\tTime: " + aTime +
            ",\tzipMapT: " + modelTrain.zipMapT.value + ",\tMapT: " + modelTrain.mapT.value + "\t," + (modelTrain.zipMapT.value.toDouble / (modelTrain.zipMapT.value + modelTrain.mapT.value)) +
            ",\titBigGradientN: " + itBigGradientN.value + ", " + itGradientN.value + ", " + itBigGradientN.value.toDouble / itGradientN.value.toDouble)
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
