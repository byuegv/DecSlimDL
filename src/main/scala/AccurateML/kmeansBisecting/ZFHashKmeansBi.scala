//package org.apache.spark.mllib.clustering.zf
//
//import java.util.Random
//
//import AccurateML.blas.ZFUtils
//import AccurateML.nonLinearRegression.ZFHash3
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.mllib.clustering._
//import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.rdd.RDD
//import org.apache.spark.storage.StorageLevel
//import org.apache.spark.util.Utils
//import org.apache.spark.{SparkConf, SparkContext}
//import redis.clients.jedis.Jedis
//
//import scala.collection.mutable
//import scala.collection.mutable.ArrayBuffer
//
///**
//  * Created by zhangfan on 17/4/6.
//  */
//class ZFHashKmeansBi(var k: Int,
//                     var maxIterations: Int,
//                     var minDivisibleClusterSize: Double,
//                     var seed: Long) {
//
//  import ZFHashKmeansBi._
//
//  def this() = this(4, 20, 1.0, classOf[BisectingKMeans].getName.##)
//
//  def setK(k: Int): this.type = {
//    require(k > 0, s"k must be positive but got $k.")
//    this.k = k
//    this
//  }
//
//
//  def run(input: RDD[(Vector, ArrayBuffer[Vector])]): BisectingKMeansModel = {
//    if (input.getStorageLevel == StorageLevel.NONE) {
//      println(s"The input RDD ${input.id} is not directly cached, which may hurt performance if"
//        + " its parent RDDs are also not cached.")
//    }
//    val d = input.map(_._1.size).first()
//    val vectors = input.map(zipPoints => {
//      val zip = zipPoints._1
//      val points = zipPoints._2
//
//      val zipWithNorm = new VectorWithNorm(zip, Vectors.norm(zip, 2.0))
//      val pointsWithNorm = points.map(p => new VectorWithNorm(p, Vectors.norm(p, 2.0)))
//      (zipWithNorm, pointsWithNorm)
//    })
//    var assignments = vectors.map(zipWithPoints => {
//      val zipWithIndex = ((ROOT_INDEX, zipWithPoints._1))
//      val pointsWithIndex = zipWithPoints._2.map(p => (ROOT_INDEX, p))
//      (zipWithIndex, pointsWithIndex)
//    })
//
//    var activeClusters = summarize(d, assignments)
//    val rootSummary = activeClusters(ROOT_INDEX)
//    val n = rootSummary.size
//    val minSize = if (minDivisibleClusterSize >= 1.0) {
//      math.ceil(minDivisibleClusterSize).toLong
//    } else {
//      math.ceil(minDivisibleClusterSize * n).toLong
//    }
//    var inactiveClusters = mutable.Seq.empty[(Long, ClusterSummary)]
//    val random = new Random(seed)
//    var numLeafClustersNeeded = k - 1
//    var level = 1
//    while (activeClusters.nonEmpty && numLeafClustersNeeded > 0 && level < LEVEL_LIMIT) {
//      // Divisible clusters are sufficiently large and have non-trivial cost.
//      var divisibleClusters = activeClusters.filter { case (_, summary) =>
//        (summary.size >= minSize) && (summary.cost > MLUtils.EPSILON * summary.size)
//      }
//      // If we don't need all divisible clusters, take the larger ones.
//      if (divisibleClusters.size > numLeafClustersNeeded) {
//        divisibleClusters = divisibleClusters.toSeq.sortBy { case (_, summary) =>
//          -summary.size
//        }.take(numLeafClustersNeeded)
//          .toMap
//      }
//      if (divisibleClusters.nonEmpty) {
//        val divisibleIndices = divisibleClusters.keys.toSet
//        var newClusterCenters = divisibleClusters.flatMap { case (index, summary) =>
//          val (left, right) = splitCenter(summary.center, random)
//          Iterator((leftChildIndex(index), left), (rightChildIndex(index), right))
//        }.map(identity) // workaround for a Scala bug (SI-7005) that produces a not serializable map
//        var newClusters: Map[Long, ClusterSummary] = null
//        var newAssignments: RDD[((Long, VectorWithNorm), ArrayBuffer[(Long, VectorWithNorm)])] = null
//        for (iter <- 0 until maxIterations) {
//          newAssignments = updateAssignments(assignments, divisibleIndices, newClusterCenters)
//            .filter { case (index, _) =>
//              divisibleIndices.contains(parentIndex(index._1))
//            }
//          newClusters = summarize(d, newAssignments)
//          newClusterCenters = newClusters.mapValues(_.center).map(identity)
//        }
//        // TODO: Unpersist old indices.
//        val indices = updateAssignments(assignments, divisibleIndices, newClusterCenters)
//          .persist(StorageLevel.MEMORY_AND_DISK)
//        assignments = indices//indices.zip(vectors)
//        inactiveClusters ++= activeClusters
//        activeClusters = newClusters
//        numLeafClustersNeeded -= divisibleClusters.size
//      } else {
//        inactiveClusters ++= activeClusters
//        activeClusters = Map.empty
//      }
//      level += 1
//    }
//    val clusters = activeClusters ++ inactiveClusters
//    val root = buildTree(clusters)
//    new BisectingKMeansModel(root)
//  }
//
//
//}
//
//object ZFHashKmeansBi extends Serializable {
//  private val ROOT_INDEX: Long = 1
//  private val MAX_DIVISIBLE_CLUSTER_INDEX: Long = Long.MaxValue / 2
//  private val LEVEL_LIMIT = math.log10(Long.MaxValue) / math.log10(2)
//
//
//  def main(args: Array[String]) {
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//    val conf = new SparkConf().setAppName("test nonlinear")
//    val sc = new SparkContext(conf)
//
//    val k = args(0).toInt //4
//    val itN = args(1).toInt //20
//    val numFeatures = args(2).toInt //102660
//    val dataPath = args(3) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
//    val isSparse = args(4).toBoolean
//    val minPartN = args(5).toInt
//
//    val itqbitN = args(6).toInt
//    val itqitN = args(7).toInt
//    val itqratioN = args(8).toInt //from 1 not 0
//    val upBound = args(9).toInt
//    val resultPath = args(10)
//    val splitN = 2 //args(12).toDouble
//
//
//    val data: RDD[LabeledPoint] = if (isSparse) {
//      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
//    } else {
//      sc.textFile(dataPath, minPartN).map(s => new LabeledPoint(0.0, Vectors.dense(s.split(",").map(_.toDouble))))
//    }
//
//    val jedis = new Jedis("localhost")
//    jedis.flushAll()
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
//    val bkm = new ZFHashKmeansBi(k,itN,1.0,Utils.random.nextLong())
//    val model = bkm.run(objectData)
//    val cost = model.computeCost(data.map(lp=>lp.features))
//
//    println("cost is:\t"+cost)
//    // Show the compute cost and the cluster centers
////    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
////      println(s"Cluster Center ${idx}: ${center}")
////    }
//
//
//  }
//
//  /** Returns the parent index of the given node index, or 0 if the input is 1 (root). */
//  private def parentIndex(index: Long): Long = {
//    index / 2
//  }
//
//  private class ClusterSummaryAggregator(val d: Int) extends Serializable {
//    private var n: Long = 0L
//    private val sum: Vector = Vectors.zeros(d)
//    private var sumSq: Double = 0.0
//
//    /** Adds a point. */
//    def add(v: VectorWithNorm): this.type = {
//      n += 1L
//      // TODO: use a numerically stable approach to estimate cost
//      sumSq += v.norm * v.norm
//      BLAS.axpy(1.0, v.vector, sum)
//      this
//    }
//
//    /** Merges another aggregator. */
//    def merge(other: ClusterSummaryAggregator): this.type = {
//      n += other.n
//      sumSq += other.sumSq
//      BLAS.axpy(1.0, other.sum, sum)
//      this
//    }
//
//    /** Returns the summary. */
//    def summary: ClusterSummary = {
//      val mean = sum.copy
//      if (n > 0L) {
//        BLAS.scal(1.0 / n, mean)
//      }
//      val center = new VectorWithNorm(mean)
//      val cost = math.max(sumSq - n * center.norm * center.norm, 0.0)
//      new ClusterSummary(n, center, cost)
//    }
//
//
//  }
//
//  def summarize(d: Int, assignments: RDD[((Long, VectorWithNorm), ArrayBuffer[(Long, VectorWithNorm)])]): Map[Long, ClusterSummary] = {
//    assignments.flatMap(zp => zp._2).aggregateByKey(new ClusterSummaryAggregator(d))(
//      seqOp = (agg, v) => agg.add(v),
//      combOp = (agg1, agg2) => agg1.merge(agg2)
//    ).mapValues(_.summary)
//      .collect().toMap
//
//  }
//
//  private def leftChildIndex(index: Long): Long = {
//    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index.")
//    2 * index
//  }
//
//  /** Returns the right child index of the given node index. */
//  private def rightChildIndex(index: Long): Long = {
//    require(index <= MAX_DIVISIBLE_CLUSTER_INDEX, s"Child index out of bound: 2 * $index + 1.")
//    2 * index + 1
//  }
//
//  private def buildTree(clusters: Map[Long, ClusterSummary]): ClusteringTreeNode = {
//    var leafIndex = 0
//    var internalIndex = -1
//
//    /**
//      * Builds a subtree from this given node index.
//      */
//    def buildSubTree(rawIndex: Long): ClusteringTreeNode = {
//      val cluster = clusters(rawIndex)
//      val size = cluster.size
//      val center = cluster.center
//      val cost = cluster.cost
//      val isInternal = clusters.contains(leftChildIndex(rawIndex))
//      if (isInternal) {
//        val index = internalIndex
//        internalIndex -= 1
//        val leftIndex = leftChildIndex(rawIndex)
//        val rightIndex = rightChildIndex(rawIndex)
//        val height = math.sqrt(Seq(leftIndex, rightIndex).map { childIndex =>
//          KMeans.fastSquaredDistance(center, clusters(childIndex).center)
//        }.max)
//        val left = buildSubTree(leftIndex)
//        val right = buildSubTree(rightIndex)
//        new ClusteringTreeNode(index, size, center, cost, height, Array(left, right))
//      } else {
//        val index = leafIndex
//        leafIndex += 1
//        val height = 0.0
//        new ClusteringTreeNode(index, size, center, cost, height, Array.empty)
//      }
//    }
//
//    buildSubTree(ROOT_INDEX)
//  }
//
//  def updateAssignments(
//                         assignments: RDD[((Long, VectorWithNorm),
//                           ArrayBuffer[(Long, VectorWithNorm)])],
//                         divisibleIndices: Set[Long],
//                         newClusterCenters: Map[Long, VectorWithNorm]): RDD[((Long, VectorWithNorm), ArrayBuffer[(Long, VectorWithNorm)])] = {
//    assignments.map(zp=>{
//      if(divisibleIndices.contains(zp._1._1)){
//        val iz=zp._1
//        val ips = zp._2
//        val newiz={
//          val index = iz._1
//          val v = iz._2
//          val children = Seq(leftChildIndex(index), rightChildIndex(index))
//          val selected = children.minBy { child =>
//            KMeans.fastSquaredDistance(newClusterCenters(child), v)
//          }
//          (selected, v)
//        }
//        val newips = ips.map(ip=>{
//          val index = ip._1
//          val v = ip._2
//          val children = Seq(leftChildIndex(index), rightChildIndex(index))
//          val selected = children.minBy { child =>
//            KMeans.fastSquaredDistance(newClusterCenters(child), v)
//          }
//          (selected, v)
//        })
//        (newiz,newips)
//      }else{
//        zp
//      }
//    })
//
//  }
//
//  private def splitCenter(
//                           center: VectorWithNorm,
//                           random: Random): (VectorWithNorm, VectorWithNorm) = {
//    val d = center.vector.size
//    val norm = center.norm
//    val level = 1e-4 * norm
//    val noise = Vectors.dense(Array.fill(d)(random.nextDouble()))
//    val left = center.vector.copy
//    BLAS.axpy(-level, noise, left)
//    val right = center.vector.copy
//    BLAS.axpy(level, noise, right)
//    (new VectorWithNorm(left), new VectorWithNorm(right))
//  }
//
//  case class ClusterSummary(size: Long, center: VectorWithNorm, cost: Double)
//
//}
