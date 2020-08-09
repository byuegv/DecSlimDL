package org.apache.spark.mllib.clustering.zf


import java.util.Random

import AccurateML.blas.ZFUtils
import AccurateML.lsh.ZFHash
import org.apache.log4j.{Level, Logger}
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * A bisecting k-means algorithm based on the paper "A comparison of document clustering techniques"
  * by Steinbach, Karypis, and Kumar, with modification to fit Spark.
  * The algorithm starts from a single cluster that contains all points.
  * Iteratively it finds divisible clusters on the bottom level and bisects each of them using
  * k-means, until there are `k` leaf clusters in total or no leaf clusters are divisible.
  * The bisecting steps of clusters on the same level are grouped together to increase parallelism.
  * If bisecting all divisible clusters on the bottom level would result more than `k` leaf clusters,
  * larger clusters get higher priority.
  *
  * @param k                       the desired number of leaf clusters (default: 4). The actual number could be smaller if
  *                                there are no divisible leaf clusters.
  * @param maxIterations           the max number of k-means iterations to split clusters (default: 20)
  * @param minDivisibleClusterSize the minimum number of points (if >= 1.0) or the minimum proportion
  *                                of points (if < 1.0) of a divisible cluster (default: 1)
  * @param seed                    a random seed (default: hash value of the class name)
  * @see [[http://glaros.dtc.umn.edu/gkhome/fetch/papers/docclusterKDDTMW00.pdf
  *      Steinbach, Karypis, and Kumar, A comparison of document clustering techniques,
  *      KDD Workshop on Text Mining, 2000.]]
  */
@Since("1.6.0")
class ZFHashBisectingKMeans private(
                                     private var k: Int,
                                     private var maxIterations: Int,
                                     private var minDivisibleClusterSize: Double,
                                     private var seed: Long) extends Logging {

  import ZFHashBisectingKMeans._

  /**
    * Constructs with the default configuration
    */
  @Since("1.6.0")
  def this() = this(4, 20, 1.0, classOf[ZFHashBisectingKMeans].getName.##)

  /**
    * Sets the desired number of leaf clusters (default: 4).
    * The actual number could be smaller if there are no divisible leaf clusters.
    */
  @Since("1.6.0")
  def setK(k: Int): this.type = {
    require(k > 0, s"k must be positive but got $k.")
    this.k = k
    this
  }

  /**
    * Gets the desired number of leaf clusters.
    */
  @Since("1.6.0")
  def getK: Int = this.k

  /**
    * Sets the max number of k-means iterations to split clusters (default: 20).
    */
  @Since("1.6.0")
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations > 0, s"maxIterations must be positive but got $maxIterations.")
    this.maxIterations = maxIterations
    this
  }

  /**
    * Gets the max number of k-means iterations to split clusters.
    */
  @Since("1.6.0")
  def getMaxIterations: Int = this.maxIterations

  /**
    * Sets the minimum number of points (if >= `1.0`) or the minimum proportion of points
    * (if < `1.0`) of a divisible cluster (default: 1).
    */
  @Since("1.6.0")
  def setMinDivisibleClusterSize(minDivisibleClusterSize: Double): this.type = {
    require(minDivisibleClusterSize > 0.0,
      s"minDivisibleClusterSize must be positive but got $minDivisibleClusterSize.")
    this.minDivisibleClusterSize = minDivisibleClusterSize
    this
  }

  /**
    * Gets the minimum number of points (if >= `1.0`) or the minimum proportion of points
    * (if < `1.0`) of a divisible cluster.
    */
  @Since("1.6.0")
  def getMinDivisibleClusterSize: Double = minDivisibleClusterSize

  /**
    * Sets the random seed (default: hash value of the class name).
    */
  @Since("1.6.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
    * Gets the random seed.
    */
  @Since("1.6.0")
  def getSeed: Long = this.seed

  /**
    * Runs the bisecting k-means algorithm.
    *
    * @param input RDD of vectors
    * @return model for the bisecting kmeans
    */
  @Since("1.6.0")
  def run(input: RDD[(Vector, Array[Vector])], zfRatio: Double, isSparse: Boolean): BisectingKMeansModel = {

    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning(s"The input RDD ${input.id} is not directly cached, which may hurt performance if"
        + " its parent RDDs are also not cached.")
    }
    val d = input.map(_._1.size).first()
    logInfo(s"Feature dimension: $d.")
    // Compute and cache vector norms for fast distance computation.
    val vectors = input.map(zipPoints => {
      val zip = zipPoints._1
      val points = zipPoints._2

      val zipWithNorm = new VectorWithNorm(zip, Vectors.norm(zip, 2.0))
      val pointsWithNorm = points.map(p => new VectorWithNorm(p, Vectors.norm(p, 2.0)))
      (zipWithNorm, pointsWithNorm)
    })

    var assignments = vectors.map(zipWithPoints => {
      val zipWithIndex = ((ROOT_INDEX, zipWithPoints._1))
      val pointsWithIndex = zipWithPoints._2.map(p => (ROOT_INDEX, p))
      (zipWithIndex, pointsWithIndex)
    })

    var activeClusters = summarize(d, assignments)
    val rootSummary = activeClusters(ROOT_INDEX)
    val n = rootSummary.size
    logInfo(s"Number of points: $n.")
    logInfo(s"Initial cost: ${rootSummary.cost}.")
    val minSize = if (minDivisibleClusterSize >= 1.0) {
      math.ceil(minDivisibleClusterSize).toLong
    } else {
      math.ceil(minDivisibleClusterSize * n).toLong
    }
    logInfo(s"The minimum number of points of a divisible cluster is $minSize.")
    var inactiveClusters = mutable.Seq.empty[(Long, ClusterSummary)]
    val random = new Random(seed)
    var numLeafClustersNeeded = k - 1
    var level = 1
    var zftime = System.currentTimeMillis()
    val zftimesb = new StringBuilder()


    //    println("Memory: before while\t" + Runtime.getRuntime.freeMemory() / 1024 / 1024 + ",\t" + Runtime.getRuntime.totalMemory() / 1024 / 1024 + "\t" + (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / 1024 / 1024)

    while (activeClusters.nonEmpty && numLeafClustersNeeded > 0 && level < LEVEL_LIMIT) {
      // Divisible clusters are sufficiently large and have non-trivial cost.
      var divisibleClusters = activeClusters.filter { case (_, summary) =>
        (summary.size >= minSize) && (summary.cost > MLUtils.EPSILON * summary.size)
      }
      // If we don't need all divisible clusters, take the larger ones.
      if (divisibleClusters.size > numLeafClustersNeeded) {
        divisibleClusters = divisibleClusters.toSeq.sortBy { case (_, summary) =>
          -summary.size
        }.take(numLeafClustersNeeded)
          .toMap
      }
      if (divisibleClusters.nonEmpty) {
        val divisibleIndices = divisibleClusters.keys.toSet
        logInfo(s"Dividing ${divisibleIndices.size} clusters on level $level.")
        var newClusterCenters = divisibleClusters.flatMap { case (index, summary) =>
          val (left, right) = zfsplitCenter(summary.center, random)
          Iterator((leftChildIndex(index), left), (rightChildIndex(index), right))
        }.map(identity)
        var newClusters: Map[Long, ClusterSummary] = null
        var newAssignments: RDD[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])] = null

        zftime = System.currentTimeMillis()
        //        println("first split")
        //        println(activeClusters.map(t => t._1 + " , " + t._2.size + " , " + t._2.center.vector.toArray.slice(0, 5).mkString(",")).toArray.mkString("\t"))
        //        println(newClusterCenters.map(t => t._1 + " , " + t._2.vector.toArray.slice(0, 5).mkString(",")).toArray.mkString("\t"))

        for (iter <- 0 until maxIterations) {
          //          println("Memory: in iter \t" + Runtime.getRuntime.freeMemory() / 1024 / 1024 + ",\t" + Runtime.getRuntime.totalMemory() / 1024 / 1024 + "\t" + (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / 1024 / 1024)

          newAssignments = if (iter == 0) {
            updateAssignmentsHashRatio(isSparse, iter, d, 1.0, assignments.filter(o => divisibleIndices.contains(o._1._1)), newClusterCenters)
          } else {
            updateAssignmentsHashRatio(isSparse, iter, d, zfRatio, newAssignments, newClusterCenters)
          }

          newClusters = summarize(d, newAssignments)
          val zeroNeedToAdd = new ArrayBuffer[(Long, VectorWithNorm)]()
          zeroNeedToAdd ++= newClusterCenters.filter(t => !newClusters.contains(t._1)) //add zero center
          //          if (newClusterCenters.size > newClusters.size) { //add rand point
          //            val zeroKeys = newClusterCenters.keySet -- newClusters.keySet
          //            println("zeroCenter," + zeroKeys.toArray.mkString(","))
          //            for (zerokey <- zeroKeys) {
          //              val randPoint = (zerokey, newAssignments.first()._2(0)._2)
          //              zeroNeedToAdd += randPoint
          //            }
          //          }
          newClusterCenters = newClusters.mapValues(_.center).map(identity)
          newClusterCenters ++= zeroNeedToAdd
        }
        zftimesb.append(",maxIterT, " + (System.currentTimeMillis() - zftime))
        zftime = System.currentTimeMillis()
        // TODO: Unpersist old indices.
        newAssignments.unpersist()
        assignments = updateAssignmentsHashRatioAll(isSparse, d, assignments, divisibleIndices, newClusterCenters).persist(StorageLevel.MEMORY_AND_DISK)
        inactiveClusters ++= activeClusters
        activeClusters = summarize(d, assignments.filter(o => newClusterCenters.contains(o._1._1)))
        //        activeClusters = newClusters
        numLeafClustersNeeded = (k - assignments.map(o => o._1._1).distinct().collect().size)
        //        numLeafClustersNeeded -= (newClusterCenters.size - divisibleClusters.size)
        //        println(level + "\t,leafN," + numLeafClustersNeeded + "\t,divisibleN," + divisibleIndices.size + "\t,newClusterN," + newClusterCenters.size)
        zftimesb.append(",updateIndicesT, " + (System.currentTimeMillis() - zftime))
        zftime = System.currentTimeMillis()
      } else {
        logInfo(s"None active and divisible clusters left on level $level. Stop iterations.")
        inactiveClusters ++= activeClusters
        activeClusters = Map.empty
      }
      println(zftimesb.toString)
      zftimesb.clear()

      level += 1
    }

    val clusters = activeClusters ++ inactiveClusters
    val root = buildTree(clusters)
    new BisectingKMeansModel(root)
  }

  /**
    * Java-friendly version of [[run()]].
    */
  def run(data: JavaRDD[Vector]): BisectingKMeansModel = run(data.rdd)
}

/**
  * updateAssignment compute zip vecter is time&space cost
  */
private object ZFHashBisectingKMeans extends Serializable {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("gmm")
    val sc = new SparkContext(conf)
    def parse(line: String): Vector = Vectors.dense(line.split(",").map(_.toDouble))


    val dataPath = args(0)
    val k = args(1).toInt
    val convergenceTol = args(2).toDouble
    val maxIterations = args(3).toInt
    val ratios: Array[Int] = (args(4).split(",")(0).toInt to args(4).split(",")(1).toInt).toArray
    val minPartN = args(5).toInt
    val isSparse = args(6).toBoolean
    val numFeatures = args(7).toInt

    val itqbitN = args(8).toInt
    val itqitN = args(9).toInt
    val itqratioN = args(10).toInt //from 1 not 0
    val upBound = args(11).toInt
    val resultPath = args(12)
    val splitN = 2 //args(12).toDouble
    val redisHost = "172.18.11.97"

    val data: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
    } else {
      sc.textFile(dataPath, minPartN).map(s => new LabeledPoint(0.0, Vectors.dense(s.split(",").map(_.toDouble))))
    }
    //    println("Memory: after data\t" + Runtime.getRuntime.freeMemory() / 1024 / 1024 + ",\t" + Runtime.getRuntime.totalMemory() / 1024 / 1024 + "\t" + (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / 1024 / 1024)

    val jedis = new Jedis(redisHost)
    jedis.flushAll()
    var time = System.currentTimeMillis()
    val oHash = new ZFHash(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse,redisHost)
    val objectData: RDD[(Vector, Array[Vector])] = data.mapPartitions(oHash.zfHashMap)
      .map(three => ((three._1(0)).last, three._1(1)))
      .map(t2 => (t2._1.features, t2._2.map(point => point.features).toArray))
      .persist(StorageLevel.MEMORY_AND_DISK)

    val on = objectData.count()
    time = System.currentTimeMillis() - time
    println("hastTime,\t" + time)
    //    println("Memory: objectData.count\t" + Runtime.getRuntime.freeMemory() / 1024 / 1024 + ",\t" + Runtime.getRuntime.totalMemory() / 1024 / 1024 + "\t" + (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / 1024 / 1024)

    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions)
    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val HashMapT = jedis.get("HashMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    println("objectDataN," + on + ",itqbitN," + itqbitN + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN + ",splitN," + splitN)
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + HashMapT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


    val sb = new StringBuilder()
    sb.append(time + "\n")
    sb.append("Ratio\tMSE\tTime\n")
    for (r <- ratios) {
      var time = System.currentTimeMillis()
      val ratio = r / 100.0
      val bkm = new ZFHashBisectingKMeans().setK(k).setMaxIterations(maxIterations)
      val model = bkm.run(objectData, ratio, isSparse)
      time = System.currentTimeMillis() - time

      var timecost = System.currentTimeMillis()
      val datav = data.map(lp => lp.features)
      val mse = model.computeCost(datav)
      timecost = System.currentTimeMillis() - timecost
      //      val zfmse = ZFHashKmeans.zfComputeCost(datav, model.clusterCenters)
      sb.append(ratio + "\t" + mse + "\t" + time + "\t" + timecost + "\n")
      println("ratio" + ratio + "\t" + mse + "\t" + time + "\t" + timecost + "\t" + model.root.leafNodes.size + " [ " + model.root.leafNodes.map(o => o.size).mkString(","))
      //      println("ratio" + ratio + s"\tCompute Cost: ${mse}" + "\t" + mse + "\t" + time + "\t" + model.root.leafNodes.size + " [ " + model.root.leafNodes.map(o => o.size).mkString(","))

      model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
        println(s"Cluster Center ${idx}: ${center.toArray.slice(0, 5).mkString(",")}")
      }
    }
    println("k: " + k + "\t itN: " + maxIterations)
    println(sb.toString())
  }

  /** The index of the root node of a tree. */
  private val ROOT_INDEX: Long = 1
  private val MAX_DIVISIBLE_CLUSTER_INDEX: Long = Long.MaxValue / 2

  private val LEVEL_LIMIT = math.log10(Long.MaxValue) / math.log10(2)

  /** Returns the left child index of the given node index. */
  private def leftChildIndex(index: Long): Long = {
    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index.")
    2 * index
  }

  /** Returns the right child index of the given node index. */
  private def rightChildIndex(index: Long): Long = {
    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index + 1.")
    2 * index + 1
  }

  /** Returns the parent index of the given node index, or 0 if the input is 1 (root). */
  private def parentIndex(index: Long): Long = {
    index / 2
  }


  def zfsplitCenter(
                     center: VectorWithNorm,
                     random: Random): (VectorWithNorm, VectorWithNorm) = {
    val d = center.vector.size
    val norm = center.norm
    val level = 1e-1 * math.max(norm, 1e-10) //1e-4 * math.max(norm, 1e-10)
    val noise = Vectors.dense(Array.fill(d)(random.nextDouble()))
    val left = center.vector.copy
    BLAS.axpy(-level, noise, left)
    val right = center.vector.copy
    BLAS.axpy(level, noise, right)
    (new VectorWithNorm(left), new VectorWithNorm(right))
  }

  def summarize(d: Int, assignments: RDD[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])]): Map[Long, ClusterSummary] = {
    assignments.flatMap(t3 => t3._2.toIterator).aggregateByKey(new ClusterSummaryAggregator(d))(
      seqOp = (agg, v) => agg.add(v),
      combOp = (agg1, agg2) => agg1.merge(agg2)
    ).mapValues(_.summary)
      .collect().toMap

  }

  /**
    * Cluster summary aggregator.
    *
    * @param d feature dimension
    */
  private class ClusterSummaryAggregator(val d: Int) extends Serializable {
    private var n: Long = 0L
    private val sum: Vector = Vectors.zeros(d)
    private var sumSq: Double = 0.0

    /** Adds a point. */
    def add(v: VectorWithNorm): this.type = {
      n += 1L
      // TODO: use a numerically stable approach to estimate cost
      sumSq += v.norm * v.norm
      BLAS.axpy(1.0, v.vector, sum)
      this
    }

    /** Merges another aggregator. */
    def merge(other: ClusterSummaryAggregator): this.type = {
      n += other.n
      sumSq += other.sumSq
      BLAS.axpy(1.0, other.sum, sum)
      this
    }

    /** Returns the summary. */
    def summary: ClusterSummary = {
      val mean = sum.copy
      if (n > 0L) {
        BLAS.scal(1.0 / n, mean)
      }
      val center = new VectorWithNorm(mean)
      val cost = math.max(sumSq - n * center.norm * center.norm, 0.0)
      new ClusterSummary(n, center, cost)
    }
  }


  //zf-change
  def updateAssignmentsHashRatioAll(
                                     isSparse: Boolean,
                                     d: Int,
                                     assignments: RDD[((Long, VectorWithNorm),
                                       Array[(Long, VectorWithNorm)])],
                                     divisibleIndices: Set[Long],
                                     newClusterCenters: Map[Long, VectorWithNorm]): RDD[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])] = {
    //    println("ALL\tdiv," + divisibleIndices.toArray.mkString(",") + "\t,newC," + newClusterCenters.map(t => t._1).toArray.mkString(",") + "\t,zip," + assignments.map(o => o._1._1).distinct().collect().mkString(",") + "\t,points," + assignments.flatMap(o => o._2.map(_._1)).distinct().collect().mkString(","))

    assignments.mapPartitions(pit => {
      var ans = new ArrayBuffer[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])]()
      while (pit.hasNext) {
        val zp = pit.next()
        if (divisibleIndices.contains(zp._1._1)) {
          val iz = zp._1
          val ips = zp._2

          val newips = ips.map(ip => {
            val index = ip._1
            val v = ip._2
            val children = Seq(leftChildIndex(index), rightChildIndex(index))
            //            children = children.filter(childi => newClusterCenters.contains(childi))
            val selected = children.minBy { child =>
              KMeans.fastSquaredDistance(newClusterCenters(child), v)
            }
            (selected, v)
          }).groupBy(t2 => t2._1).toArray

          if (newips.size == 1 && newips.last._1 == iz._1) {
            ans += zp
          } else {
            for (i <- 0 until newips.size) {
              val group = newips(i)
              val label = group._1
              val points: Array[(Long, VectorWithNorm)] = group._2
              var zipv: Vector = Vectors.zeros(d)
              points.foreach(p => BLAS.axpy(1, p._2.vector, zipv))
              BLAS.scal(1.0 / points.size, zipv)
              if (isSparse) zipv = zipv.toSparse
              val zipNorm = new VectorWithNorm(zipv, Vectors.norm(zipv, 2.0))
              ans += Tuple2((label, zipNorm), points)
            }
          }
        }
        else {
          ans += zp
        }
      }
      ans.toIterator
    })

  }


  //zf-change
  def updateAssignmentsHashRatio(
                                  isSparse: Boolean,
                                  iter: Int,
                                  d: Int,
                                  zfRatio: Double,
                                  assignments: RDD[((Long, VectorWithNorm),
                                    Array[(Long, VectorWithNorm)])],
                                  newClusterCenters: Map[Long, VectorWithNorm]): RDD[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])] = {
    //print
    //    val pnewcenter = newClusterCenters.map(t => t._1 + " : " + t._2.vector.toArray.slice(0, 5).mkString(",")).toArray.mkString("\t")
    //    val pzipcenter = assignments.map(o => o._1._1).distinct().collect().mkString(",")
    //    val ppoints = assignments.flatMap(o => o._2.map(_._1)).distinct().collect()
    //    val ppmap = assignments.flatMap(o => o._2.map(_._1)).map(index => (index, 1)).reduceByKey((a, b) => a + b).collect()
    //    println("iter," + iter + "\t,newC," + pnewcenter + "\t,zip," + pzipcenter + "\t,points," + ppoints.mkString(",") + "\tpmap" + ppmap.mkString(","))

    assignments.mapPartitions(pit => {
      var selectIndex: ArrayBuffer[Int] = null
      val (pit1, pit2) = pit.duplicate

      if (zfRatio < 1.0 && iter != 0) {
        val zipCost = new ArrayBuffer[Double]()
        while (pit1.hasNext) {
          val temp = pit1.next()
          val one = temp._1._1
          val two = if (one % 2 == 0) one + 1 else one - 1
          val oneCost = KMeans.fastSquaredDistance(newClusterCenters(one), temp._1._2)
          val twoCost = KMeans.fastSquaredDistance(newClusterCenters(two), temp._1._2)
          zipCost.append(oneCost - twoCost) // if oneCost > twoCost then index will be changed
        }
        selectIndex = zipCost.zipWithIndex.sortWith(_._1 > _._1).slice(0, (zfRatio * zipCost.size).toInt).map(t2 => t2._2)
      }

      var i = 0
      lazy val ans = new ArrayBuffer[((Long, VectorWithNorm), Array[(Long, VectorWithNorm)])]()
      while (pit2.hasNext) {
        val zp = pit2.next()
        if (zfRatio == 1.0 || iter == 0 || selectIndex.contains(i)) {
          val iz = zp._1
          val ips = zp._2

          val newips = ips.map(ip => {
            //one point cluster splits to two
            val index = if (iter == 0) ip._1 else ip._1 / 2
            val v = ip._2
            val children = Seq(leftChildIndex(index), rightChildIndex(index))
            //            children = children.filter(childi => newClusterCenters.contains(childi))
            val selected = children.minBy { child =>
              KMeans.fastSquaredDistance(newClusterCenters(child), v)
            }
            (selected, v)
          }).groupBy(t2 => t2._1)

          if (iter != 0 && newips.size == 1 && newips.last._1 == iz._1) {
            ans += zp
          } else {
            newips.foreach(group => {
              val label = group._1
              val points: Array[(Long, VectorWithNorm)] = group._2
              var zipv: Vector = Vectors.zeros(d)
              points.foreach(p => BLAS.axpy(1, p._2.vector, zipv))
              BLAS.scal(1.0 / points.size, zipv)
              if (isSparse) zipv = zipv.toSparse
              val zipNorm = new VectorWithNorm(zipv, Vectors.norm(zipv, 2.0))
              //            var minCost = KMeans.fastSquaredDistance(newClusterCenters(label), zipNorm)
              ans += Tuple2((label, zipNorm), points)
            })
          }


        }
        else {
          ans += zp
        }
        i += 1
      }
      ans.toIterator
    }).persist(StorageLevel.MEMORY_AND_DISK)

  }


  /**
    * Builds a clustering tree by re-indexing internal and leaf clusters.
    *
    * @param clusters a map from cluster indices to corresponding cluster summaries
    * @return the root node of the clustering tree
    */
  private def buildTree(clusters: Map[Long, ClusterSummary]): ClusteringTreeNode = {
    var leafIndex = 0
    var internalIndex = -1

    /**
      * Builds a subtree from this given node index.
      */
    //zf-change
    def buildSubTree(rawIndex: Long): ClusteringTreeNode = {
      val cluster = clusters(rawIndex)
      val size = cluster.size
      val center = cluster.center
      val cost = cluster.cost
      val isInternal = clusters.contains(leftChildIndex(rawIndex))
      if (isInternal) {
        val index = internalIndex
        internalIndex -= 1
        val leftIndex = leftChildIndex(rawIndex)
        val rightIndex = rightChildIndex(rawIndex)
        //zf
        var zfchildreni = Seq(leftIndex, rightIndex)
        zfchildreni = zfchildreni.filter(i => clusters.contains(i))
        val height = math.sqrt(zfchildreni.map { childIndex =>
          KMeans.fastSquaredDistance(center, clusters(childIndex).center)
        }.max)
        val left = if (clusters.contains(leftIndex)) buildSubTree(leftIndex) else null
        val right = if (clusters.contains(rightIndex)) buildSubTree(rightIndex) else null
        val zftemp = new ArrayBuffer[ClusteringTreeNode]()
        if (left != null)
          zftemp += left
        if (right != null)
          zftemp += right
        new ClusteringTreeNode(index, size, center, cost, height, zftemp.toArray)
      } else {
        val index = leafIndex
        leafIndex += 1
        val height = 0.0
        new ClusteringTreeNode(index, size, center, cost, height, Array.empty)
      }
    }

    buildSubTree(ROOT_INDEX)
  }

  /**
    * Summary of a cluster.
    *
    * @param size   the number of points within this cluster
    * @param center the center of the points within this cluster
    * @param cost   the sum of squared distances to the center
    */
  case class ClusterSummary(size: Long, center: VectorWithNorm, cost: Double)

}

