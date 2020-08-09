///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package org.apache.spark.mllib.clustering.zf
//
//import AccurateML.blas.ZFUtils
//import AccurateML.nonLinearRegression.ZFHash3
//import breeze.linalg.{DenseMatrix => BreezeMatrix, DenseVector => BDV, Vector => BV, diag}
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.annotation.Since
//import org.apache.spark.mllib.clustering.GaussianMixtureModel
//import org.apache.spark.mllib.linalg.{BLAS, DenseMatrix, Matrices, Vector, Vectors}
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
//import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.rdd.RDD
//import org.apache.spark.storage.StorageLevel
//import org.apache.spark.util.Utils
//import org.apache.spark.{SparkConf, SparkContext}
//import redis.clients.jedis.Jedis
//
//import scala.collection.mutable.{ArrayBuffer, IndexedSeq}
//
///**
//  * This class performs expectation maximization for multivariate Gaussian
//  * Mixture Models (GMMs).  A GMM represents a composite distribution of
//  * independent Gaussian distributions with associated "mixing" weights
//  * specifying each's contribution to the composite.
//  *
//  * Given a set of sample points, this class will maximize the log-likelihood
//  * for a mixture of k Gaussians, iterating until the log-likelihood changes by
//  * less than convergenceTol, or until it has reached the max number of iterations.
//  * While this process is generally guaranteed to converge, it is not guaranteed
//  * to find a global optimum.
//  *
//  * Note: For high-dimensional data (with many features), this algorithm may perform poorly.
//  * This is due to high-dimensional data (a) making it difficult to cluster at all (based
//  * on statistical/theoretical arguments) and (b) numerical issues with Gaussian distributions.
//  *
//  * @param k              Number of independent Gaussians in the mixture model.
//  * @param convergenceTol Maximum change in log-likelihood at which convergence
//  *                       is considered to have occurred.
//  * @param maxIterations  Maximum number of iterations allowed.
//  */
//@Since("1.3.0")
//class ZFHashGaussianMixture private(
//                                     private var k: Int,
//                                     private var convergenceTol: Double,
//                                     private var maxIterations: Int,
//                                     private var seed: Long) extends Serializable {
//
//  /**
//    * Constructs a default instance. The default parameters are {k: 2, convergenceTol: 0.01,
//    * maxIterations: 100, seed: random}.
//    */
//  @Since("1.3.0")
//  def this() = this(2, 0.01, 100, Utils.random.nextLong())
//
//  val zfllHistory = new ArrayBuffer[Double]()
//
//  var zfRatio =0.0
//
//  // number of samples per cluster to use when initializing Gaussians
//  private val nSamples = 5
//
//  // an initializing GMM can be provided rather than using the
//  // default random starting point
//  private var initialModel: Option[GaussianMixtureModel] = None
//
//  /**
//    * Set the initial GMM starting point, bypassing the random initialization.
//    * You must call setK() prior to calling this method, and the condition
//    * (model.k == this.k) must be met; failure will result in an IllegalArgumentException
//    */
//  @Since("1.3.0")
//  def setInitialModel(model: GaussianMixtureModel): this.type = {
//    require(model.k == k,
//      s"Mismatched cluster count (model.k ${model.k} != k ${k})")
//    initialModel = Some(model)
//    this
//  }
//
//  /**
//    * Return the user supplied initial GMM, if supplied
//    */
//  @Since("1.3.0")
//  def getInitialModel: Option[GaussianMixtureModel] = initialModel
//
//  /**
//    * Set the number of Gaussians in the mixture model.  Default: 2
//    */
//  @Since("1.3.0")
//  def setK(k: Int): this.type = {
//    require(k > 0,
//      s"Number of Gaussians must be positive but got ${k}")
//    this.k = k
//    this
//  }
//
//  /**
//    * Return the number of Gaussians in the mixture model
//    */
//  @Since("1.3.0")
//  def getK: Int = k
//
//  /**
//    * Set the maximum number of iterations allowed. Default: 100
//    */
//  @Since("1.3.0")
//  def setMaxIterations(maxIterations: Int): this.type = {
//    require(maxIterations >= 0,
//      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
//    this.maxIterations = maxIterations
//    this
//  }
//
//  /**
//    * Return the maximum number of iterations allowed
//    */
//  @Since("1.3.0")
//  def getMaxIterations: Int = maxIterations
//
//  /**
//    * Set the largest change in log-likelihood at which convergence is
//    * considered to have occurred.
//    */
//  @Since("1.3.0")
//  def setConvergenceTol(convergenceTol: Double): this.type = {
//    require(convergenceTol >= 0.0,
//      s"Convergence tolerance must be nonnegative but got ${convergenceTol}")
//    this.convergenceTol = convergenceTol
//    this
//  }
//
//  /**
//    * Return the largest change in log-likelihood at which convergence is
//    * considered to have occurred.
//    */
//  @Since("1.3.0")
//  def getConvergenceTol: Double = convergenceTol
//
//  /**
//    * Set the random seed
//    */
//  @Since("1.3.0")
//  def setSeed(seed: Long): this.type = {
//    this.seed = seed
//    this
//  }
//
//  /**
//    * Return the random seed
//    */
//  @Since("1.3.0")
//  def getSeed: Long = seed
//
//  /**
//    * Perform expectation maximization
//    */
//  @Since("1.3.0")
//  def run(data: RDD[(Vector, ArrayBuffer[Vector])]): GaussianMixtureModel = {
//    val sc = data.sparkContext
//
//    // we will operate on the data as breeze data
//    val breezeData = data.map(t2=>(t2._1.asBreeze,t2._2.map(vec=>vec.asBreeze))).cache()
//    val d = breezeData.first()._1.length
//    val shouldDistributeGaussians = ZFHashGaussianMixture.shouldDistributeGaussians(k, d)
//
//    val (weights, gaussians) = initialModel match {
//      case Some(gmm) => (gmm.weights, gmm.gaussians)
//      case None =>
//        val samples = breezeData.flatMap(t2=>t2._2).takeSample(withReplacement = true, k * nSamples, seed)
//        (Array.fill(k)(1.0 / k), Array.tabulate(k) { i =>
//          val slice = samples.view(i * nSamples, (i + 1) * nSamples)
//          new MultivariateGaussian(vectorMean(slice), initCovariance(slice))
//        })
//    }
//    var llh = Double.MinValue // current log-likelihood
//    var llhp = 0.0 // previous log-likelihood
//    var iter = 0
//    while (iter < maxIterations && math.abs(llh - llhp) > convergenceTol) {
//      val compute = sc.broadcast(ExpectationSum.add(weights, gaussians) _)
//      val selectTrain:RDD[BV[Double]] = breezeData.mapPartitions(pit=>{
//        //from zip get zip-llh
//        val (pit1,pit2)=pit.duplicate
//        val llhArr = new ArrayBuffer[Double]()
//        val ans  = new ArrayBuffer[BV[Double]]()
//        while(pit1.hasNext){
//          val zip = pit1.next()._1
//          val psums = compute.value(ExpectationSum.zero(k,d),zip)
//          val pSumWeights = psums.weights.sum
//          var i = 0
//          while (i < k) {
//            val (weight, gaussian) =
//              updateWeightsAndGaussians(psums.means(i), psums.sigmas(i), psums.weights(i), pSumWeights)
//            weights(i) = weight
//            gaussians(i) = gaussian
//            i = i + 1
//          }
//          llh = psums.logLikelihood
//          llhArr += llh
//        }
//
//        val selectIndex=llhArr.zipWithIndex.sortWith(_._1>_._1).slice(0,(zfRatio*llhArr.size).toInt).map(t2=>t2._2)
//
//        for(i<- 0 to selectIndex.size){// to vs until //看似又问题selectIndex.size应该改成llhArr.size
//          val temp = pit2.next()
//          if (selectIndex.contains(i)){
//            ans ++= temp._2
//          }
//        }
//        ans.toIterator
//      }).cache()
//
//      val sums = selectTrain.aggregate(ExpectationSum.zero(k, d))(compute.value, _ += _)
//      val sumWeights = sums.weights.sum
//      if (shouldDistributeGaussians) {
//        val numPartitions = math.min(k, 1024)
//        val tuples =
//          Seq.tabulate(k)(i => (sums.means(i), sums.sigmas(i), sums.weights(i)))
//        val (ws, gs) = sc.parallelize(tuples, numPartitions).map { case (mean, sigma, weight) =>
//          updateWeightsAndGaussians(mean, sigma, weight, sumWeights)
//        }.collect().unzip
//        Array.copy(ws.toArray, 0, weights, 0, ws.length)
//        Array.copy(gs.toArray, 0, gaussians, 0, gs.length)
//      } else {
//        var i = 0
//        while (i < k) {
//          val (weight, gaussian) =
//            updateWeightsAndGaussians(sums.means(i), sums.sigmas(i), sums.weights(i), sumWeights)
//          weights(i) = weight
//          gaussians(i) = gaussian
//          i = i + 1
//        }
//      }
//
//      llhp = llh // current becomes previous
//      llh = sums.logLikelihood // this is the freshly computed log-likelihood
//      iter += 1
//      zfllHistory.append(llh)
//    }
//
//    new GaussianMixtureModel(weights, gaussians)
//  }
//
//  def getTestllh(data: RDD[Vector], gmm: GaussianMixtureModel): Double = {
//
//    val sc = data.sparkContext
//    val breezeData = data.map(_.asBreeze).cache()
//    val d = breezeData.first().length
//    val shouldDistributeGaussians = ZFHashGaussianMixture.shouldDistributeGaussians(k, d)
//    val (weights, gaussians) = (gmm.weights, gmm.gaussians)
//    var llh = Double.MinValue // current log-likelihood
//    var llhp = 0.0 // previous log-likelihood
//
//    var iter = 0
//    var ans = 0.0
//    while (iter < 1) {
//      val compute = sc.broadcast(ExpectationSum.add(weights, gaussians) _)
//      val sums = breezeData.aggregate(ExpectationSum.zero(k, d))(compute.value, _ += _)
//      val sumWeights = sums.weights.sum
//      if (shouldDistributeGaussians) {
//        val numPartitions = math.min(k, 1024)
//        val tuples =
//          Seq.tabulate(k)(i => (sums.means(i), sums.sigmas(i), sums.weights(i)))
//        val (ws, gs) = sc.parallelize(tuples, numPartitions).map { case (mean, sigma, weight) =>
//          updateWeightsAndGaussians(mean, sigma, weight, sumWeights)
//        }.collect().unzip
//        Array.copy(ws.toArray, 0, weights, 0, ws.length)
//        Array.copy(gs.toArray, 0, gaussians, 0, gs.length)
//      } else {
//        var i = 0
//        while (i < k) {
//          val (weight, gaussian) =
//            updateWeightsAndGaussians(sums.means(i), sums.sigmas(i), sums.weights(i), sumWeights)
//          weights(i) = weight
//          gaussians(i) = gaussian
//          i = i + 1
//        }
//      }
//
//      llhp = llh // current becomes previous
//      llh = sums.logLikelihood // this is the freshly computed log-likelihood
//      iter += 1
//      ans = llh
//    }
//
//    return ans
//
//  }
//
//  private def updateWeightsAndGaussians(
//                                         mean: BDV[Double],
//                                         sigma: BreezeMatrix[Double],
//                                         weight: Double,
//                                         sumWeights: Double): (Double, MultivariateGaussian) = {
//    val mu = (mean /= weight)
//    BLAS.syr(-weight, Vectors.fromBreeze(mu),
//      Matrices.fromBreeze(sigma).asInstanceOf[DenseMatrix])
//    val newWeight = weight / sumWeights
//    val newGaussian = new MultivariateGaussian(mu, sigma / weight)
//    (newWeight, newGaussian)
//  }
//
//  /** Average of dense breeze vectors */
//  private def vectorMean(x: IndexedSeq[BV[Double]]): BDV[Double] = {
//    val v = BDV.zeros[Double](x(0).length)
//    x.foreach(xi => v += xi)
//    v / x.length.toDouble
//  }
//
//  /**
//    * Construct matrix where diagonal entries are element-wise
//    * variance of input vectors (computes biased variance)
//    */
//  private def initCovariance(x: IndexedSeq[BV[Double]]): BreezeMatrix[Double] = {
//    val mu = vectorMean(x)
//    val ss = BDV.zeros[Double](x(0).length)
//    x.foreach(xi => ss += (xi - mu) :^ 2.0)
//    diag(ss / x.length.toDouble)
//  }
//}
//
//object ZFHashGaussianMixture {
//  /**
//    * Heuristic to distribute the computation of the [[MultivariateGaussian]]s, approximately when
//    * d > 25 except for when k is very small.
//    *
//    * @param k Number of topics
//    * @param d Number of features
//    */
//  def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25
//
//
//  def main(args: Array[String]) {
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//    val conf = new SparkConf().setAppName("gmm")
//    val sc = new SparkContext(conf)
//
//    val dataPath = args(0)
//    val k = args(1).toInt
//    val convergenceTol = args(2).toDouble
//    val maxIterations = args(3).toInt
//    val ratios: Array[Int] = (args(4).split(",")(0).toInt to args(4).split(",")(1).toInt).toArray
//    val minPartition = args(5).toInt
//
//    val itqbitN = args(8).toInt
//    val itqitN = args(9).toInt
//    val itqratioN = args(10).toInt //from 1 not 0
//    val upBound = args(11).toInt
//    val splitN = 2 //args(12).toDouble
//    val resultPath = args(12)
//
//    val data = sc.textFile(dataPath, minPartition).map(s => new LabeledPoint(0.0, Vectors.dense(s.split(",").map(_.toDouble))))
//    val jedis = new Jedis("localhost")
//    jedis.flushAll()
//    val isSparse = false
//    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
//    val objectData: RDD[(Vector, ArrayBuffer[Vector])] = data.mapPartitions(oHash.zfHashMap)
//      .map(three => ((three._1(0)).last, three._1(1)))
//      .map(t2 => (t2._1.features, t2._2.map(point => point.features)))
//      .persist(StorageLevel.MEMORY_AND_DISK)
//
//    val on = objectData.count()
//    println()
//    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions)
//    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val HashMapT = jedis.get("HashMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//    println("objectDataN," + on + ",itqbitN," + itqbitN + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN + ",splitN," + splitN)
//    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + HashMapT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
//    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
//    jedis.close()
//
//
//    val zfsb = new StringBuilder()
//    zfsb.append("Ratio\tRound\tllh\tTime\n")
//    for (r <- ratios) {
//      var rTime = System.currentTimeMillis()
//      val ratio = r / 100.0
//      val data = sc.textFile(dataPath, minPartition).map(s => Vectors.dense(s.trim.split(",").map(_.toDouble)))
//      val train = objectData
//      val gm = new ZFHashGaussianMixture(k, convergenceTol, maxIterations, Utils.random.nextLong())
//      val gmm = gm.run(train)
//      //      for (i <- 0 until gmm.k) {
//      //        println("weight=%f\nmu=%s\nsigma=\n%s\n" format
//      //          (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
//      //      }
//      println(ratio + "\t" + train.count() + "\t" + train.partitions.size)
//      println(gm.zfllHistory.mkString(","))
//
//      rTime = System.currentTimeMillis() - rTime
//      val testllh = gm.getTestllh(data, gmm)
//      //      val probs = gmm.predictSoft(data).map(vs => vs.max).sum()
//      println(ratio + "\t" + gm.zfllHistory.size + "\t" + gm.zfllHistory.last + "\t" + testllh + "\t" + rTime)
//      if (r != ratios.last)
//        zfsb.append(ratio + "\t" + gm.zfllHistory.size + "\t" + gm.zfllHistory.last + "\t" + testllh + "\t" + rTime + "\n")
//      else
//        zfsb.append(ratio + "\t" + gm.zfllHistory.size + "\t" + gm.zfllHistory.last + "\t" + testllh + "\t" + rTime)
//
//
//    }
//    println(dataPath)
//    println(zfsb)
//
//  }
//}
//
//// companion class to provide zero constructor for ExpectationSum
//private object ExpectationSum {
//  def zero(k: Int, d: Int): ExpectationSum = {
//    new ExpectationSum(0.0, Array.fill(k)(0.0),
//      Array.fill(k)(BDV.zeros(d)), Array.fill(k)(BreezeMatrix.zeros(d, d)))
//  }
//
//  // compute cluster contributions for each input point
//  // (U, T) => U for aggregation
//  def add(
//           weights: Array[Double],
//           dists: Array[MultivariateGaussian])
//         (sums: ExpectationSum, x: BV[Double]): ExpectationSum = {
//    val p = weights.zip(dists).map {
//      case (weight, dist) => MLUtils.EPSILON + weight * dist.pdf(x)
//    }
//    val pSum = p.sum
//    sums.logLikelihood += math.log(pSum)
//    var i = 0
//    while (i < sums.k) {
//      p(i) /= pSum
//      sums.weights(i) += p(i)
//      sums.means(i) += x * p(i)
//      BLAS.syr(p(i), Vectors.fromBreeze(x),
//        Matrices.fromBreeze(sums.sigmas(i)).asInstanceOf[DenseMatrix])
//      i = i + 1
//    }
//    sums
//  }
//}
//
//// Aggregation class for partial expectation results
//private class ExpectationSum(
//                              var logLikelihood: Double,
//                              val weights: Array[Double],
//                              val means: Array[BDV[Double]],
//                              val sigmas: Array[BreezeMatrix[Double]]) extends Serializable {
//
//  val k = weights.length
//
//  def +=(x: ExpectationSum): ExpectationSum = {
//    var i = 0
//    while (i < k) {
//      weights(i) += x.weights(i)
//      means(i) += x.means(i)
//      sigmas(i) += x.sigmas(i)
//      i = i + 1
//    }
//    logLikelihood += x.logLikelihood
//    this
//  }
//}
