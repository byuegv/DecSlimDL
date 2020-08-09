package AccurateML.svm.github

import AccurateML.lsh.ZFHash
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 17/8/24.
  */


object ZFIncreSvdSVM {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val logFile = "README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)


    ///Users/zhangfan/Documents/data/a9a - 0 0 10,50,100 true 123 1
    val dataPath = args(0)
    val testPath = args(1)
    val packN = args(2).toInt
    val itN = args(3).toInt
    val ratioL = args(4).split(",").map(_.toDouble)
    val isSparse = args(5).toBoolean
    val numFeature = args(6).toInt
    val kernel = args(7)
    val step = args(8).toDouble
    val gamma = args(9).toDouble
    val minPartN = args(10).toInt
    val checkPointDir = args(11) //hdfs://172.18.11.90:9000/checkpoint
    val perCheckN = args(12).toInt

    val itqbitN = args(13).toInt
    val itqitN = args(14).toInt
    val itqratioN = args(15).toInt //from 1 not 0
    val upBound = args(16).toInt
    val redisHost = args(17) //"172.18.11.97"


    val splitChar = ","

    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features).toSparse.asInstanceOf[SparseVector]) // must be sparse
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }
    val cancel1 = data.first()

    val zipTime = System.currentTimeMillis()
    val indexedData: IndexedRDD[Long, (LabeledPoint, Double)] = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })
    val oHash = new ZFHash(itqbitN, itqitN, itqratioN, upBound, 2, isSparse, redisHost, sc)
    val zipIndex: RDD[(String, Array[Long])] = indexedData.mapPartitions(oHash.zfHashMapIndexedRDD)
    val cancelzN: Array[Int] = zipIndex.collect().map(_._2.toArray.size).sortWith(_ < _)
    println("zipN:\t " + cancelzN.size + "\t" + cancelzN.sum / cancelzN.size + ", " + cancelzN.slice(1, 10).mkString(",") + ",,," + cancelzN.slice(cancelzN.size - 10, cancelzN.size).mkString(","))

    /**
      * zipData: Array[("0101".toInt, (lp_0, 0, ids_Array[Long]))]
      */
    val zipData: Array[(Long, (LabeledPoint, Double, Array[Long]))] =
      zipIndex.map { case (binary, ids) =>
        val centerId: Long = java.lang.Long.parseLong(binary, 2)
        val lp: LabeledPoint = new LabeledPoint(0.0, Vectors.zeros(numFeature))
        (centerId, Tuple3(lp, 0.0D, ids))
      }.collect()
    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))


    val tempP = new ArrayBuffer[String]()
    for (r <- ratioL) {
      val ratio = r / 100.0
      val splits = indexedData.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) IndexedRDD(indexedData.sample(false, ratio)) else IndexedRDD(splits(0).sample(false, ratio))
      val test = if (testPath.size > 3) {
        if (!isSparse) {
          sc.textFile(testPath).map(line => {
            val vs = line.split(splitChar).map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1).map(_._2._1)
      println("partN:\t" + train.getNumPartitions)
      val t1 = System.currentTimeMillis
      val svm = new ZFHashSVMTrain(train, zipData, test, step, kernel, gamma, checkPointDir) //gamma = 0.1
      svm.train(itN * packN, packN, perCheckN)
      val t2 = System.currentTimeMillis
      val runtime = (t2 - t1)

      println("model count:\t" + svm.model.count())
      println("Ratio\titN\tpackN\tACC\tT")

      val acc = svm.getAccuracy(test.collect())
      var ss = ratio + "\t" + itN + "*" + packN + "\t" + acc + "\t" + runtime + "\n"

      System.out.println(ss)
      tempP += ss
      //              test.collect().foreach { x => System.out.println(svm.predict(x) + " "  + x.label) }
      data.unpersist()
      train.unpersist()

    }
    println("Ratio\titN\tpackN\tACC\tT")
    println(tempP.mkString("\n"))

  }
}
