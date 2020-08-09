

package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV, norm}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.ml.ann.{ZFANNGradient, ZFDataStacker}
import org.apache.spark.ml.classification.{ZFLabelConverter, ZFMultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector => mlVector, Vectors => mlVectors}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer


class ZFZipGradientDescent(var gradient: ZFANNGradient, var updater: ZFUpdater, layers: Array[Int], testset: Dataset[_])
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


  //  def zfGetGradient(weights: Vector, gradient: ZFANNGradient): Array[Double] = {
  //    val sn = zfcc.size
  //    val cc = zfcc.map { v =>
  //      (0.0,
  //        org.apache.spark.ml.linalg.Vectors.fromBreeze(BDV.vertcat(
  //          v._1.asBreeze.toDenseVector,
  //          v._2.asBreeze.toDenseVector))
  //        )
  //    }.map { v =>
  //      (v._1, Vectors.fromML(v._2))
  //    }
  //
  //
  //
  //    val ans = cc.map(lp => {
  //      val g = Vectors.fromBreeze(BDV.zeros[Double](sn))
  //      gradient.compute(lp._2, lp._1, weights, g)
  //      g dot g
  //    })
  //    ans
  //  }

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
                       zipObjectData: RDD[(Vector, Array[LabeledPoint])],
                       dataStacker: ZFDataStacker,
                       initialWeights: Vector,
                       ratio: Double
                       //                       convergenceTol: Double = 0.001
                     ): (Vector, Array[Double], Long) = {

    println(" zipObjectData.partN, " + zipObjectData.partitions.size)
    val labels = layers.last
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size


    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    var converged = false
    var i = 1

    var itTime = 0L
    var accTime = 0L
    val partZipTime = zipObjectData.sparkContext.longAccumulator
    val partPointsTime = zipObjectData.sparkContext.longAccumulator

    //    val zt0 = zipObjectData.sparkContext.longAccumulator
    //    val zt1 = zipObjectData.sparkContext.longAccumulator
    //    val zt2 = zipObjectData.sparkContext.longAccumulator
    //
    //    val t0 = zipObjectData.sparkContext.longAccumulator
    //    val t1 = zipObjectData.sparkContext.longAccumulator
    //    val t2 = zipObjectData.sparkContext.longAccumulator
    //    var itZipTime = 0L
    //    var itPointsTime = 0L


    while (i <= numIterations) {
      //!converged &&
      val itStartT = System.currentTimeMillis()
      val usedPointsN = zipObjectData.sparkContext.longAccumulator
      val usedZipsN = zipObjectData.sparkContext.longAccumulator
      val zipGrads = zipObjectData.sparkContext.collectionAccumulator[Double]


      val bcWeights = zipObjectData.context.broadcast(weights)
      val selectlpData: RDD[LabeledPoint] = zipObjectData.mapPartitions(pit => {
        val zpstart = System.currentTimeMillis()
        val (p1, p2) = pit.duplicate
        val zipGrads: Array[Double] = p1.map(t => gradient.computeZipGrad(t._1, bcWeights.value)).toArray
        val zipGradMean = zipGrads.sum / zipGrads.size / 50
        usedZipsN.add(zipGrads.size)
        partZipTime.add(System.currentTimeMillis() - zpstart)

        p2.zip(zipGrads.iterator).filter(t => t._2 > 1E-4 && t._2 > zipGradMean).flatMap(t => {
          usedPointsN.add(t._1._2.size)
          t._1._2
        })
      }) //.persist(StorageLevel.MEMORY_AND_DISK)
      //      val selectlpData: RDD[LabeledPoint] = zipObjectData.filter(t2 => {
      //        val zipg = gradient.computeZipGrad(t2._1, bcWeights.value, Vectors.zeros(n))
      //        zipg > 10
      //      }).flatMap(_._2).persist(StorageLevel.MEMORY_AND_DISK)
      //      val selectN = selectlpData.count()
      //      usedPointsN.add(selectN)
      //      itZipTime += System.currentTimeMillis() - itStartT

      val vecData = selectlpData.repartition(zipObjectData.partitions.size).map(lp => ZFLabelConverter.encodeLabeledPoint(lp, labels))
      val selectStackData = dataStacker.stack(vecData).map { v =>
        (v._1, Vectors.fromML(v._2))
      }
      val (gradientSum, lossSum, miniBatchSize) = selectStackData
        .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
          seqOp = (c, v) => {
            val pstart = System.currentTimeMillis()
            val l = gradient.compute(v._2, v._1, bcWeights.value, Vectors.fromBreeze(c._1))
            partPointsTime.add(System.currentTimeMillis() - pstart)
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
          converged = false //ZFGradientDescent.isConverged(previousWeights.get,
          //            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      //      val c0 = zfAccumulator.value.sortWith(_ < _)
      //      val c1 = c0.grouped(600).map(_.sum / 600)
      //      println(i + "," + c0.filter(_ != 0).size + " ,[, " + c1.mkString(",")) //.map(dd=>(f"$dd%1.2f"))


      itTime += System.currentTimeMillis() - itStartT
      val accT = System.currentTimeMillis()
      if (i % 100 == 0) {
        val acc = zfGetAcc(weights)
        println(i + " ,ACC, " + acc + " ,itTime , " + itTime + " ,partZipTime, " + partZipTime.value  + " ,partPointsTime, " + partPointsTime.value  + " ,usedZipsN, " + usedZipsN.value + " ,usedPointsN, " + usedPointsN.value)
//        val c = zipGrads.value.sortWith(_ < _)
//        println("zipGrads, " + c.sum / c.size + ",[," + c.slice(0, 10) + ",,," + c.slice(c.size - 10, c.size))
      }

      accTime += System.currentTimeMillis() - accT
      i += 1
    }

    logInfo("ZFZipGradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    Tuple3(weights, stochasticLossHistory.toArray, accTime)
  }


}


@DeveloperApi
object ZFZipGradientDescent extends Logging {


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
