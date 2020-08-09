package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.DenseVector
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 17/6/21.
  */
class TestLinerRegression {

}
object TestLinerRegression{
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val stepSize: Double = args(0).toDouble
    val numFeature: Int = args(1).toInt //10
    val itN: Int = args(2).toInt
    val testPath: String = args(3)
    val dataPath: String = args(4)
    val test100: Array[Double] = args(5).split(",").map(_.toDouble)
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt

    val splitChar = ",|\\s+"
    val ratioL = test100


    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Long]()

    for (r <- ratioL) {
      val dataTxt = sc.textFile(dataPath, minPartN) // "/Users/zhangfan/Documents/nonlinear.f10.n100.h5.txt"
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
      val train = if (testPath.size > 3) data.sample(false, ratio).cache() else splits(0).sample(false, ratio).cache()
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


      val model = LinearRegressionWithSGD.train(train, itN, stepSize)
      // Evaluate model on training examples and compute training error
      val valuesAndPreds = test.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
      println("training Mean Squared Error = " + MSE)
      mesb += MSE
      nntimesb += 0
    }

    println()
    println(this.getClass.getName+" ,itN"+itN+ " ,step," + stepSize + " ,data," + dataPath)
    println("ratio,MSE,nnMapT")
    for (i <- ratioL.toArray.indices) {
      println(ratioL(i) / 100.0 + "," + mesb(i) + "," + nntimesb(i))
    }
  }
}
