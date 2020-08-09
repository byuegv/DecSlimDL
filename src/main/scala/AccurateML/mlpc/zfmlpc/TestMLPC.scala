package AccurateML.mlpc.zfmlpc

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.ZFMLPC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangfan on 18/4/19.
  */
class TestMLPC {

}

object TestMLPC {
  def zfGetWeightN(layers: Array[Int]): Int = {
    var n = 0
    for (i <- 0 until layers.length - 1) {
      n += (layers(i) + 1) * layers(i + 1)
    }
    n
  }

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)
    // Load the data stored in LIBSVM format as a DataFrame.

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    // Split the data into train and test

    // /Users/zhangfan/Documents/zf.txt /Users/zhangfan/Documents/zf.txt 1 6,10,3 100 -1 1 1.0
    val dataPath = args(0) //"/Users/zhangfan/Downloads/mnist/train.txt"
    val testPath = args(1) //"/Users/zhangfan/Downloads/mnist/test.txt"
    val minPart = args(2).toInt
    val layers = args(3).split(",").map(_.toInt) //784..10
    val numIt = args(4).toInt
    val initW = args(5).toDouble
    val blockSize = args(6).toInt
    val ratio = args(7).toDouble



    println("minPart, " + minPart + " ,layers, " + layers.mkString(",") + " ,numIt, " + numIt + " ,blockSize, " + blockSize + " ,ratio, " + ratio)

    val t0 = System.currentTimeMillis()


    val train = spark.read.format("libsvm").load(dataPath).sample(false, ratio).repartition(minPart)
    val test = spark.read.format("libsvm").load(testPath)

    // create the trainer and set its parameters
    val weightsN = zfGetWeightN(layers)
    val trainer = new ZFMLPC(test, layers)
      .setSolver("gd") //'l-bfgs' default
      .setSeed(1234L)
      .setBlockSize(blockSize)
      .setMaxIter(numIt)
      .setStepSize(1.0)


    if (initW != -1) {
      trainer.setInitialWeights(Vectors.dense(Array.fill(weightsN)(initW)))
    }

    // train the model
    val model = trainer.train(train)

    val trainT = System.currentTimeMillis() - t0 - model.zfOtherTime
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels) + " ,Time, " + trainT)
  }
}
