//package AccurateML.svm.github
//
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.rdd.RDD
//import org.apache.spark.{SparkConf, SparkContext}
//
//import scala.collection.mutable.ArrayBuffer
//
///**
//  * Created by zhangfan on 17/8/23.
//  */
//class ZFKmeansLshSVM(kmeansK: Int, kmeansItN: Int, numFeature: Int) {
//  val k = kmeansK
//  val kItN = kmeansItN
//
//  def kmeansLSH(indexedData: RDD[(Long, LabeledPoint)]): Unit = {
//    indexedData.mapPartitions(pit => {
//      val partData = new ArrayBuffer[(Long, LabeledPoint)]()
//      while (pit.hasNext) {
//        partData += pit.next()
//      }
//      val centers = partData.slice(0,k).map(vec=>(vec,0))
//
//
//
//
//
//      pit
//
//    })
//  }
//
//
//}
//
//object ZFKmeansLshSVM {
//  def main(args: Array[String]) {
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//    val logFile = "README.md" // Should be some file on your system
//    val conf = new SparkConf().setAppName("github.KernelSVM Test")
//    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
//    val sc = new SparkContext(conf)
//
//    val splitChar = ","
//
//    val dataPath = args(0)
//    val testPath = args(1)
//    val packN = args(2).toInt
//    val itN = args(3).toInt
//    val ratioL = args(4).split(",").map(_.toDouble)
//    val isSparse = args(5).toBoolean
//    val numFeature = args(6).toInt
//    val minPartN = args(7).toInt
//    val checkPointDir = args(8)
//
//    val kmeansK = args(9).toInt
//    val kmeansItN = args(10).toInt
//
//
//    val data: RDD[LabeledPoint] = if (!isSparse) {
//      sc.textFile(dataPath, minPartN).map(line => {
//        val vs = line.split(splitChar).map(_.toDouble)
//        val features = vs.slice(0, vs.size - 1)
//        LabeledPoint(vs.last, Vectors.dense(features))
//      })
//    } else {
//      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
//    }
//
//    val indexedData = data.zipWithUniqueId().map(_.swap)
//    val kmeansLsh = new ZFKmeansLshSVM()
//    kmeansLsh.kmeansLSH(indexedData)
//
//
//  }
//}
