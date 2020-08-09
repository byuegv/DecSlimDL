

package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, norm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.ml.ann.{ZFANNGradient, ZFDataStacker}
import org.apache.spark.ml.classification.ZFMultilayerPerceptronClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector => mlVector, Vectors => mlVectors}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer


class ZFGradientDescent(var gradient: ZFANNGradient, var updater: ZFUpdater, layers: Array[Int], testset: Dataset[_])
  extends Logging with Serializable {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001


  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }


  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }


  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }


  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }


  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }


  def setGradient(gradient: ZFANNGradient): this.type = {
    this.gradient = gradient
    this
  }


  def setUpdater(updater: ZFUpdater): this.type = {
    this.updater = updater
    this
  }


  def zfGetAcc(weights: Vector): Double = {
    val model = new ZFMultilayerPerceptronClassificationModel("", layers, weights.asML)
    // compute accuracy on the test set
    val result = model.transform(testset)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val acc = evaluator.evaluate(predictionAndLabels)
    acc
  }

  //  @DeveloperApi
  //  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
  //    val (weights, _) = runMiniBatchSGD(
  //      data,
  //      gradient,
  //      updater,
  //      stepSize,
  //      numIterations,
  //      regParam,
  //      miniBatchFraction,
  //      initialWeights,
  //      convergenceTol)
  //    weights
  //  }

  def runMiniBatchSGD(
                       data: RDD[(mlVector, mlVector)],
                       dataStacker: ZFDataStacker,
                       initialWeights: Vector
                       //                       convergenceTol: Double = 0.001
                     ): (Vector, Array[Double], Long) = {

    val stackData = dataStacker.stack(data).map { v =>
      (v._1, Vectors.fromML(v._2))
    }

    println(" stackData.partN, " + stackData.partitions.size)
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None
    var accTime = 0L

    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size


    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false
    var i = 1

    var itTime = 0L
    //    val t0 = data.sparkContext.longAccumulator
    //    val t1 = data.sparkContext.longAccumulator
    //    val t2 = data.sparkContext.longAccumulator

    //    val grads = data.sparkContext.collectionAccumulator[Double]
    while (i <= numIterations) {
      //!converged &&
      val itStartT = System.currentTimeMillis()
      //      val zfAccumulator = stackData.sparkContext.collectionAccumulator[Double]
      val bcWeights = stackData.context.broadcast(weights)


      val (gradientSum, lossSum, miniBatchSize) = stackData.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {

            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {

            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })



      if (miniBatchSize > 0) {

        stochasticLossHistory.append(lossSum / miniBatchSize + regVal)
        val update = updater.compute(
          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
          stepSize, i, regParam)
        weights = update._1
        regVal = update._2

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = false //ZFGradientDescent.isConverged(previousWeights.get, currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      //      val c0 = zfAccumulator.value.sortWith(_ < _)
      //      val c1 = c0.grouped(600).map(_.sum / 600)
      //      println(i + "," + c0.filter(_ != 0).size + " ,[, " + c1.mkString(",")) //.map(dd=>(f"$dd%1.2f"))

      itTime += System.currentTimeMillis() - itStartT
      val accT = System.currentTimeMillis()
      if (i % 10 == 0) {
        val acc = zfGetAcc(weights)
        println(i + " , " + acc + " ,itTime, " + itTime)
        //        val c = grads.value.sortWith(_ < _)
        //        println("Grads, " + c.sum / c.size + ",[," + c.slice(0, 10) + ",,," + c.slice(c.size - 10, c.size))
      }
      accTime += System.currentTimeMillis() - accT
      i += 1
    }

    logInfo("ZFGradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray, accTime)

  }


}


@DeveloperApi
object ZFGradientDescent extends Logging {


  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {

    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector


    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

}
