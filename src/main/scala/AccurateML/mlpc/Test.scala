package AccurateML.mlpc

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangfan on 18/4/13.
  */
class Test {

}

object Test {

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

    val dataPath = args(0) //"/Users/zhangfan/Downloads/mnist/train.txt"
    val testPath = args(1) //"/Users/zhangfan/Downloads/mnist/test.txt"
    val layers = args(2).split(",").map(_.toInt) //784..10
    val numIt = args(3).toInt
    val initW = args(4).toDouble


    println("layers, " + layers.mkString(",") + " ,numIt, " + numIt)

    val t0 = System.currentTimeMillis()

    val train = spark.read.format("libsvm").load(dataPath)
    val test = spark.read.format("libsvm").load(testPath)
    // create the trainer and set its parameters
    val weightsN = zfGetWeightN(layers)
    val trainer = new MultilayerPerceptronClassifier()
      .setSolver("gd") //'l-bfgs' default
      .setSeed(1234L)
      .setBlockSize(128)
      .setMaxIter(numIt)
      .setLayers(layers)
    if (initW != -1) {
      trainer.setInitialWeights(Vectors.dense(Array.fill(weightsN)(initW)))
    }

    // train the model
    val model = trainer.fit(train)

    val trainT = System.currentTimeMillis() - t0
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels) + " ,Time, " + trainT)
  }
}
