package AccurateML.gm

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangfan on 17/3/22.
  */
class TestGMM {

}

object TestGMM {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("gmm")
    val sc = new SparkContext(conf)

    val dataPath = args(0)
    val k = args(1).toInt
    val convergenceTol = args(2).toDouble
    val maxIterations = args(3).toInt
    val ratios: Array[Int] = (args(4).split(",")(0).toInt to args(4).split(",")(1).toInt).toArray
    val minPartition = args(5).toInt

    for (r <- ratios) {
      var rTime = System.currentTimeMillis()
      val ratio = r / 100.0
      val data = sc.textFile(dataPath, minPartition).map(s => Vectors.dense(s.trim.split(',').map(_.toDouble)))
      val splits = data.randomSplit(Array(ratio, 1 - ratio)) //"/Users/zhangfan/Documents/data/mllib/gmm_data.txt"
      val train = splits(0).cache()
      val gmm = new GaussianMixture().setK(k).setConvergenceTol(convergenceTol).setMaxIterations(maxIterations).run(train)
      //      for (i <- 0 until gmm.k) {
      //        println("weight=%f\nmu=%s\nsigma=\n%s\n" format
      //          (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
      //      }
      rTime = System.currentTimeMillis() - rTime
      val probs = gmm.predictSoft(data).map(vs => vs.max).sum()
      println(ratio + "\t" + train.count() + "\t" + probs + "\t" + rTime)

    }
  }
}
