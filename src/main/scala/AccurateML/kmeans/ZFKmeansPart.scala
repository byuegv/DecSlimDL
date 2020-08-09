package AccurateML.kmeans

import AccurateML.blas.ZFBLAS
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 17/1/9.
  */


class ZFKmeansPart(k: Int, itN: Int, sc: SparkContext) extends Serializable {

  val costHis = new ArrayBuffer[Double]()
  var mapT = sc.longAccumulator
  var mapT1 = sc.longAccumulator
  var mapT2 = sc.longAccumulator
  val computePointN = sc.longAccumulator


  //  def runGradientAlgorithm(data: RDD[Vector], origCenters: Array[Vector], ratio: Double): Array[Vector] = {
  //    if (data.getStorageLevel == StorageLevel.NONE) {
  //      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
  //        + " parent RDDs are also uncached.------------------------")
  //    }
  //    val sc = data.sparkContext
  //    val mapdata = data.mapPartitions(points => {
  //      val jedis = new Jedis(redisHost)
  //      val nnMapT = System.currentTimeMillis()
  //
  //      val ans = new ArrayBuffer[Tuple2[Vector, Array[Double]]]()
  //      while (points.hasNext) {
  //        ans += Tuple2(points.next(), Array(0.0, 0.0)) // (lastLabel,lastCost)
  //      }
  //      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
  //      ans.iterator
  //    }).cache()
  //
  //    val centers = origCenters.clone()
  //
  //    for (it <- 0 until itN) {
  //      if (it == 0) {
  //        val bcCenters = sc.broadcast(centers)
  //        val costAccum = sc.doubleAccumulator
  //
  //        val totalContibs = mapdata.mapPartitions { points =>
  //          val jedis = new Jedis(redisHost)
  //          val nnMapT = System.currentTimeMillis()
  //          val thisCenters = bcCenters.value
  //          val dims = thisCenters(0).size
  //          val sums = Array.fill(k)(Vectors.zeros(dims))
  //          val counts = Array.fill(k)(0L)
  //          println("it," + it)
  //          points.foreach { point =>
  //            val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point._1, thisCenters)
  //            counts(centerIndex) += 1
  //            costAccum.add(cost)
  //            ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
  //            point._2(0) = centerIndex
  //            point._2(1) = cost
  //          }
  //          val contribs = for (i <- 0 until k) yield {
  //            (i, (sums(i), counts(i)))
  //          }
  //
  //          val partN = counts.sum
  //          jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
  //          jedis.append("partN", "," + partN)
  //          jedis.close()
  //          contribs.iterator
  //        }.reduceByKey((a, b) => {
  //          ZFBLAS.axpy(1.0, a._1, b._1)
  //          (b._1, a._2 + b._2)
  //        }).collectAsMap()
  //        bcCenters.unpersist(blocking = false)
  //        for (i <- 0 until k) {
  //          val (sum, count) = totalContibs(i)
  //          if (count != 0) {
  //            ZFBLAS.scal(1.0 / count, sum)
  //            centers(i) = sum
  //          }
  //        }
  //        costHis += costAccum.value
  //      } else {
  //        // it>0
  //        val bcCenters = sc.broadcast(centers)
  //        val costAccum = sc.doubleAccumulator
  //
  //        val totalContibs = mapdata.mapPartitions { points =>
  //          val jedis = new Jedis(redisHost)
  //          val nnMapT = System.currentTimeMillis()
  //          val thisCenters = bcCenters.value
  //          val dims = thisCenters(0).size
  //          val sums = Array.fill(k)(Vectors.zeros(dims))
  //          val counts = Array.fill(k)(0L)
  //
  //          var cenCostAB = new ArrayBuffer[Tuple4[Int, Double, Double, Tuple2[Vector, Array[Double]]]]()
  //
  //          points.foreach { point =>
  //            val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point._1, thisCenters)
  //            val gap = math.abs(cost - point._2(1))
  //            cenCostAB += Tuple4(centerIndex, cost, gap, point)
  //          }
  //
  //          val ratioN = (cenCostAB.size * ratio).toInt
  //          cenCostAB = cenCostAB.sortWith(_._3 > _._3) //.slice(0, ratioN)
  //          for (i <- 0 until cenCostAB.size) {
  //            if (i < ratioN) {
  //              cenCostAB(i) match {
  //                case (centerIndex, cost, gap, point) => {
  //                  counts(centerIndex) += 1
  //                  costAccum.add(cost)
  //                  ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
  //                  point._2(0) = centerIndex
  //                  point._2(1) = cost
  //                }
  //              }
  //            } else {
  //              cenCostAB(i) match {
  //                case (_, _, _, point) => {
  //                  val oldCenterIndex = point._2(0).toInt
  //                  val oldCost = point._2(1)
  //                  counts(oldCenterIndex) += 1
  //                  costAccum.add(oldCost)
  //                  ZFBLAS.axpy(1.0, point._1, sums(oldCenterIndex))
  //                }
  //              }
  //            }
  //          }
  //
  //          val contribs = for (i <- 0 until k) yield {
  //            (i, (sums(i), counts(i)))
  //          }
  //
  //          val partN = counts.sum
  //          jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
  //          jedis.append("partN", "," + partN)
  //          jedis.close()
  //          contribs.iterator
  //        }.reduceByKey((a, b) => {
  //          ZFBLAS.axpy(1.0, a._1, b._1)
  //          (b._1, a._2 + b._2)
  //        }).collectAsMap()
  //        bcCenters.unpersist(blocking = false)
  //        for (i <- 0 until k) {
  //          val (sum, count) = totalContibs(i)
  //          if (count != 0) {
  //            ZFBLAS.scal(1.0 / count, sum)
  //            centers(i) = sum
  //          }
  //        }
  //        costHis += costAccum.value
  //      }
  //
  //    } //end-it
  //    centers
  //  }

  def runAlgorithm(data: RDD[Vector], origCenters: Array[Vector], disFunc: String): Array[Vector] = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }

    val sc = data.sparkContext
    val centers = origCenters.clone() //data.takeSample(false, k)
    println("centers -1 :\t" + centers.map(_.toArray.sum).sum)

    var itTime = 0L
    for (it <- 0 until itN) {
      val itStartTime = System.currentTimeMillis()
      val bcCenters = sc.broadcast(centers)
      val costAccum = sc.doubleAccumulator

      val totalContibs = data.mapPartitions { points =>
        val nnMapT = System.currentTimeMillis()
        val thisCenters = bcCenters.value
        val dims = thisCenters(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        points.foreach { point =>
          val temp = System.currentTimeMillis()
          val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point, thisCenters, disFunc)
          mapT1.add(System.currentTimeMillis() - temp)
          counts(centerIndex) += 1
          val temp1 = System.currentTimeMillis()
          ZFBLAS.axpy(1.0, point, sums(centerIndex))
          mapT2.add(System.currentTimeMillis() - temp1)
          costAccum.add(cost)
        }
        val contribs = for (i <- 0 until k) yield {
          (i, (sums(i), counts(i)))
        }

        computePointN.add(counts.sum)
        mapT.add((System.currentTimeMillis() - nnMapT))
        contribs.iterator
      }.reduceByKey((a, b) => {
        ZFBLAS.axpy(1.0, a._1, b._1)
        (b._1, a._2 + b._2)
      }).collectAsMap()
      bcCenters.unpersist(blocking = false)
      for (i <- 0 until k) {
        val (sum, count) = totalContibs(i)
        if (count != 0) {
          ZFBLAS.scal(1.0 / count, sum)
          centers(i) = sum
        }
      }
      itTime += System.currentTimeMillis() - itStartTime
      println(it + "\tMapWSSSE: " + costAccum.value + "\tcenters: " + centers.map(_.toArray.sum).sum + "\titTime:" + itTime + "\t accMapT: " + mapT.value + " , " + mapT1.value + "," + mapT2.value + ", computePointN," + computePointN.value)
      //      println(it + "\tMapWSSSE: " + costAccum.value + "\tcenters: " + centers.map(_.toArray.sum).sum + "\titTime:" + itTime + "\t accMapT: " + mapT.value + ", computePointN," + computePointN.value)
      costHis += costAccum.value
    } //end-it
    centers
  }


}


object ZFKmeansPart {
  def initCenters ( centerPath:String,k:Int,isSparse:Boolean,numFeatures:Int): Array[Vector] ={
    val origCenters = new Array[Vector](k)
    val iter = Source.fromFile(centerPath).getLines()
    var tempk = 0
    while (iter.hasNext && tempk < k) {
      val line = iter.next()
      if (!isSparse) {
        val vs = line.split(",|\\s+").map(_.toDouble)
        origCenters(tempk) = Vectors.dense(vs.slice(0, vs.size - 1))
      } else {
        val vs = line.split("\\s+")
        val features = vs.slice(1, vs.size).map(s => s.split(":"))
        val indexs = features.map(arr => arr(0).toInt)
        val values = features.map(arr => arr(1).toDouble)
        origCenters(tempk) = Vectors.sparse(numFeatures, indexs, values)
      }
      tempk += 1
    }
    origCenters
  }
  def zfComputeCost(data: RDD[Vector], centers: Array[Vector], disFunc: String): Double = {
    val ans = data.map(point => ZFKmeansPart.zfFindClosest(point, centers, disFunc)._2).sum()
    ans
  }


  def zfFindClosest(point: Vector, centers: Array[Vector], disFunc: String): Tuple2[Int, Double] = {
    var minIndex: Int = -1
    var minValue: Double = Double.MaxValue

    for (i <- 0 until centers.size) {
      val cost = zfdistance(centers(i), point, disFunc)
      if (cost < minValue) {
        minValue = cost
        minIndex = i
      }
    }
    Tuple2(minIndex, minValue)
  }

  def zfFindClosest1(point: Vector, centers: Array[Vector], disFunc: String): Tuple3[Int, Double, Double] = {
    //centerIndex,cost,sortValue
    var minIndex: Int = -1
    var minValue: Double = Double.MaxValue

    val costs = Array.fill[Double](centers.size)(0.0)

    for (i <- 0 until centers.size) {
      val cost = zfdistance(centers(i), point, disFunc)
      if (cost < minValue) {
        minValue = cost
        minIndex = i
      }
      costs(i) = cost
    }
    val sortCosts = costs.sortWith(_ < _)
    val sortValue = (sortCosts(1) - sortCosts(0)) / sortCosts(0)

    Tuple3(minIndex, minValue, sortValue)
  }

  def zfFindClosest2(point: Vector, centers: Array[Vector], disFunc: String): Tuple3[Int, Double, Double] = {
    //centerIndex,cost,sortValue
    var minIndex: Int = -1
    var minValue: Double = Double.MaxValue

    val costs = Array.fill[Double](centers.size)(0.0)

    for (i <- 0 until centers.size) {
      val cost = zfdistance(centers(i), point, disFunc)
      if (cost < minValue) {
        minValue = cost
        minIndex = i
      }
      costs(i) = cost
    }
    val sortCosts = costs.sortWith(_ < _)
    val sortValue = sortCosts(1) - sortCosts(0)

    Tuple3(minIndex, minValue, sortValue)
  }

  def zfdistance(v1: Vector, v2: Vector, disFunc: String): Double = {
    val dis = if (disFunc.contains("eucli")) {
      Vectors.sqdist(v1, v2)
    } else if (disFunc.contains("cos")) {
      zfCosine(v1, v2)
    } else {
      System.err.println("disFunc name error! " + disFunc)
      Vectors.sqdist(v1, v2)
    }
    dis
  }

  def zfCosine(v1: Vector, v2: Vector): Double = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and Dim(v2)" +
      s"=${v2.size}.")
    //    var squaredDistance = 0.0
    var numerator = 0.0
    var denominatorA = 0.0
    var denominatorB = 0.0
    var ans = 0.0

    (v1, v2) match {
      case (v1: SparseVector, v2: SparseVector) =>
        val v1Values = v1.values
        val v1Indices = v1.indices
        val v2Values = v2.values
        val v2Indices = v2.indices
        val nnzv1 = v1Indices.length
        val nnzv2 = v2Indices.length

        var kv1 = 0
        var kv2 = 0
        while (kv1 < nnzv1 || kv2 < nnzv2) {
          //          var score = 0.0
          var adot = 0.0

          if (kv2 >= nnzv2 || (kv1 < nnzv1 && v1Indices(kv1) < v2Indices(kv2))) {
            //            score = v1Values(kv1)
            denominatorA += math.pow(v1Values(kv1), 2)
            kv1 += 1
          } else if (kv1 >= nnzv1 || (kv2 < nnzv2 && v2Indices(kv2) < v1Indices(kv1))) {
            //            score = v2Values(kv2)
            denominatorB += math.pow(v2Values(kv2), 2)
            kv2 += 1
          } else {
            //            score = v1Values(kv1) - v2Values(kv2)
            denominatorA += math.pow(v1Values(kv1), 2)
            denominatorB += math.pow(v2Values(kv2), 2)
            adot = v1Values(kv1) * v2Values(kv2)
            kv1 += 1
            kv2 += 1
          }
          numerator += adot
          //          squaredDistance += score * score
        }
        ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))

      case (v1: SparseVector, v2: DenseVector) =>
        ans = zfCosine(v1, v2)

      case (v1: DenseVector, v2: SparseVector) =>
        ans = zfCosine(v2, v1)

      case (DenseVector(vv1), DenseVector(vv2)) =>
        var kv = 0
        val sz = vv1.length
        while (kv < sz) {
          //          val score = vv1(kv) - vv2(kv)
          //          squaredDistance += score * score
          denominatorA += math.pow(vv1(kv), 2)
          denominatorB += math.pow(vv2(kv), 2)
          numerator += vv1(kv) * vv2(kv)
          kv += 1
        }
        ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))
      case _ =>
        throw new IllegalArgumentException("Do not support vector type " + v1.getClass +
          " and " + v2.getClass)
    }
    ans
  }

  /**
    * Returns the squared distance between DenseVector and SparseVector.
    */
  def zfCosine(v1: SparseVector, v2: DenseVector): Double = {
    var kv1 = 0
    var kv2 = 0
    val indices = v1.indices
    //    var squaredDistance = 0.0
    var numerator = 0.0
    var denominatorA = 0.0
    var denominatorB = 0.0
    var ans = 0.0

    val nnzv1 = indices.length
    val nnzv2 = v2.size
    var iv1 = if (nnzv1 > 0) indices(kv1) else -1

    while (kv2 < nnzv2) {
      //      var score = 0.0
      var adot = 0.0
      if (kv2 != iv1) {
        //        score = v2(kv2)
        denominatorB += v2(kv2) * v2(kv2)
      } else {
        //        score = v1.values(kv1) - v2(kv2)
        denominatorA += v1.values(kv1) * v1.values(kv1)
        denominatorB += v2(kv2) * v2(kv2)
        adot = v1.values(kv1) * v2(kv2)
        if (kv1 < nnzv1 - 1) {
          kv1 += 1
          iv1 = indices(kv1)
        }
      }
      //      squaredDistance += score * score
      numerator += adot
      kv2 += 1
    }
    //    squaredDistance
    ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))
    ans
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("ZFKmeansPart")
    val sc = new SparkContext(conf)
    //    testKMeans(sc)
    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val test100: Array[Double] = args(5).split(",").map(_.toDouble)
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt

    val disFunc = args(8)
    val testPath = args(9)


    val rDivN = 100
    val ratioL = test100
    //    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)
    val mesb = new ArrayBuffer[Double]()
    val nnMapTimes = new ArrayBuffer[Long]()
    val nnTimes = new ArrayBuffer[Long]()
    val computePointNs = new ArrayBuffer[Long]()
    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble
      val data: RDD[LabeledPoint] = if (isSparse) {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
      } else {
        sc.textFile(dataPath, minPartN).map(s => {
          val vs = s.split(",")
          new LabeledPoint(0.0, Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble)))
        })
      }
      val train = data.map(_.features).sample(false, ratio).persist(StorageLevel.MEMORY_AND_DISK)
      train.count()
      val origCenters = initCenters(centerPath,k,isSparse,numFeatures)

      val t1 = System.currentTimeMillis
      val zfkmeans = new ZFKmeansPart(k, itN, sc)
      val centers = zfkmeans.runAlgorithm(train, origCenters, disFunc)
      val runtime = (System.currentTimeMillis - t1)

      val nnMapT = zfkmeans.mapT.value
      computePointNs += zfkmeans.computePointN.value

      val WSSSE = if (testPath.size < 2) {
        ZFKmeansPart.zfComputeCost(data.map(_.features), centers, disFunc)
      } else {
        val test: RDD[Vector] = if (isSparse) {
          MLUtils.loadLibSVMFile(sc, testPath, numFeatures).map(pt => pt.features)
        } else {
          sc.textFile(testPath).map(s => {
            val vs = s.split(",")
            Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble))
          })
        }
        ZFKmeansPart.zfComputeCost(test, centers, disFunc)

      }
      mesb += WSSSE
      nnMapTimes += nnMapT
      nnTimes += runtime

      println()
      System.out.println("ratio," + ratio + ", runTime," + runtime + ",WSSSE, " + WSSSE)
      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",nnMapT," + nnMapT + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
      val testN = if (testPath.size < 2) data.count() else sc.textFile(testPath).count()
      println("WSSSEs,[," + zfkmeans.costHis.mkString(",") + ",k," + k + ",itN," + itN + ",trainN," + train.count() / 10000.0 + ",testN," + testN / 10000.0 + ",numFeatures," + train.first().size)
      train.unpersist()

    }
    println()
    println(this.getClass.getName + ",data," + dataPath)
    println("ratio,MSE,Time,MapT,computePointsN")
    for (i <- ratioL.indices) {
      println(ratioL(i) / rDivN.toDouble + "," + mesb(i) + "," + nnTimes(i) + "," + nnMapTimes(i) + "," + computePointNs(i))
    }

  }
}
