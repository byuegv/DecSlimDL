//package AccurateML.kmeans
//
//import AccurateML.blas.{ZFBLAS, ZFUtils}
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.mllib.linalg.{Vector, Vectors}
//import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.rdd.RDD
//import org.apache.spark.storage.StorageLevel
//import org.apache.spark.{SparkConf, SparkContext}
//import redis.clients.jedis.Jedis
//
//import scala.collection.mutable.ArrayBuffer
//import scala.io.Source
//
///**
//  * Created by zhangfan on 17/1/9.
//  */
//
//
//class ZFKmeansPartSGD1(k: Int, itN: Int, redisHost: String, sc: SparkContext) extends Serializable {
//
//  val costHis = new ArrayBuffer[Double]()
//  var mapT = sc.longAccumulator
//
//  /**
//    * 对数据点进行排序,只修前ratio比例数据的centerIndex
//    *
//    * @param data
//    * @param origCenters
//    * @param ratio
//    * @return
//    */
//  def runAlgorithm1(data: RDD[Vector], origCenters: Array[Vector], ratio: Double): Array[Vector] = {
//
//    if (data.getStorageLevel == StorageLevel.NONE) {
//      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
//        + " parent RDDs are also uncached.------------------------")
//    }
//
//    val sc = data.sparkContext
//
//    val mapdata = data.map(vec => Tuple2(vec, Array(0))) // lastCenterIndex
//    mapdata.persist(StorageLevel.MEMORY_AND_DISK)
//
//    val centers = origCenters.clone() //data.takeSample(false, k)
//    val selectNs = new ArrayBuffer[Double]()
//
//
//
//    for (it <- 0 until itN) {
//      val bcCenters = sc.broadcast(centers)
//      val costAccum = sc.doubleAccumulator
//      val bcIt = sc.broadcast(it)
//
//      val totalContibs = mapdata.mapPartitions { points =>
//        val (it1, it2) = points.duplicate
//        val iter = bcIt.value
//        val jedis = new Jedis(redisHost)
//        val nnMapT = System.currentTimeMillis()
//        val thisCenters = bcCenters.value
//        val dims = thisCenters(0).size
//        val sums = Array.fill(k)(Vectors.zeros(dims))
//        val counts = Array.fill(k)(0L)
//        val zipSortIndex = if (iter == 0) {
//          Array.range(0, it1.size)
//        } else {
//          var cenCostAB = new ArrayBuffer[Int]() //zipIndex
//          it1.zipWithIndex.foreach { it =>
//            val vec = it._1._1
//            val info = it._1._2
//            val lastCenterIndex = info(0)
//            val zipIndex = it._2
//            val (centerIndex, cost, relastCost) = ZFKmeansPart.zfFindClosestNf(vec, thisCenters, lastCenterIndex.toInt, 100)
//            val sortValue = relastCost - cost
//            if (lastCenterIndex != centerIndex)
//              cenCostAB += zipIndex
//          }
//          cenCostAB.toArray
//        }
//
//
//        var tempi = 0
//        while (it2.hasNext) {
//          val temp = it2.next()
//          val vec = temp._1
//          val info = temp._2 //lastCenterIndex
//          if (zipSortIndex.contains(tempi)) {
//            val (centerIndex, cost, relastCost) = ZFKmeansPart.zfFindClosest(vec, thisCenters, info(0))
//            val sortValue = relastCost - cost
//            info(0) = centerIndex
//            //            costAccum.add(cost)
//          }
//          val nowCenterIndex = info(0).toInt
//          ZFBLAS.axpy(1.0, vec, sums(nowCenterIndex))
//          counts(nowCenterIndex) += 1
//
//          tempi += 1
//        }
//
//        val contribs = for (i <- 0 until k) yield {
//          (i, (sums(i), counts(i)))
//        }
//
//        mapT.add((System.currentTimeMillis() - nnMapT))
//        jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
//        jedis.close()
//        contribs.iterator
//      }.reduceByKey((a, b) => {
//        ZFBLAS.axpy(1.0, a._1, b._1)
//        (b._1, a._2 + b._2)
//      }).collectAsMap()
//      bcCenters.unpersist(blocking = false)
//
//      var selectN = 0L
//      for (i <- 0 until k) {
//        val (sum, count) = totalContibs(i)
//        selectN += count
//        if (count != 0) {
//          ZFBLAS.scal(1.0 / count, sum)
//          centers(i) = sum
//        }
//      }
//      println(it + "\trunAlgorithm1 MapWSSSE: " + costAccum.value + "\tcenters: " + centers.map(_.toArray.sum).sum + "\t accMapT: " + mapT.value)
//      costHis += costAccum.value
//      selectNs += selectN
//    } //end-it
//
//    println("selectNs: " + selectNs.map(_ / 10000.0).mkString(","))
//
//    println()
//    centers
//
//  }
//
//
//  def zfComputeCost(data: RDD[Vector], centers: Array[Vector]): Double = {
//    val ans = data.map(point => ZFKmeansPartSGD1.zfFindClosest(point, centers)._2).sum()
//    ans
//  }
//
//}
//
//
//object ZFKmeansPartSGD1 {
//  def zfFindClosest(point: Vector, centers: Array[Vector]): Tuple2[Int, Double] = {
//    var minIndex: Int = -1
//    var minValue: Double = Double.MaxValue
//
//    for (i <- 0 until centers.size) {
//      val cost = Vectors.sqdist(centers(i), point)
//      if (cost < minValue) {
//        minValue = cost
//        minIndex = i
//      }
//    }
//    Tuple2(minIndex, minValue)
//  }
//
//  def main(args: Array[String]): Unit = {
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//    val conf = new SparkConf().setAppName("test nonlinear")
//    val sc = new SparkContext(conf)
//    //    testKMeans(sc)
////    hello1
//    val k = args(0).toInt //2
//    val itN = args(1).toInt //5
//    val numFeatures = args(2).toInt //102660
//    val centerPath = args(3)
//    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
//    val test100: Array[Double] = args(5).split(",").map(_.toDouble)
//    //args(6) not used
//    val isSparse = args(7).toBoolean
//    val minPartN = args(8).toInt
//
//    val redisHost = args(9) //"172.18.11.97"
//    val testPath = args(10)
//
//
//    val rDivN = 100
//    val ratioL = test100
//    //    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)
//    val mesb = new ArrayBuffer[Double]()
//    val nntimesb = new ArrayBuffer[Long]()
//    for (r <- ratioL) {
//      val ratio = r / rDivN.toDouble
//      val jedis = new Jedis(redisHost)
//      jedis.flushAll()
//
//      val data: RDD[Vector] = if (isSparse) {
//        MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN).map(point => point.features)
//      } else {
//        sc.textFile(dataPath, minPartN).map(s => Vectors.dense(s.split(",").map(_.toDouble)))
//      }
//      val train = data.sample(false, ratio).cache()
//      val origCenters = new Array[Vector](k)
//      val iter = Source.fromFile(centerPath).getLines()
//      var tempk = 0
//      while (iter.hasNext && tempk < k) {
//        val line = iter.next()
//        if (!isSparse) {
//          origCenters(tempk) = Vectors.dense(line.split(",").map(_.toDouble))
//        } else {
//          val vs = line.split("\\s+")
//          val features = vs.slice(1, vs.size).map(s => s.split(":"))
//          val indexs = features.map(arr => arr(0).toInt)
//          val values = features.map(arr => arr(1).toDouble)
//          origCenters(tempk) = Vectors.sparse(numFeatures, indexs, values)
//        }
//        tempk += 1
//      }
//
//
//      val zfkmeans = new ZFKmeansPartSGD1(k, itN, redisHost, sc)
//      val centers = zfkmeans.runAlgorithm1(train, origCenters, 1.0)
//
//
//      val nnMapT = zfkmeans.mapT.value
//      val cancelnnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
//
//      val WSSSE = if (testPath.size < 2) {
//        ZFKmeansPart.zfComputeCost(data, centers)
//      } else {
//        val test: RDD[Vector] = if (isSparse) {
//          MLUtils.loadLibSVMFile(sc, testPath, numFeatures).map(pt => pt.features)
//        } else {
//          sc.textFile(testPath).map(s => Vectors.dense(s.split(",").map(_.toDouble)))
//        }
//        ZFKmeansPart.zfComputeCost(test, centers)
//
//      }
//      mesb += WSSSE
//      nntimesb += nnMapT
//
//      println()
//      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",nnMapT," + nnMapT + "," + cancelnnMapT.sum + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
//      val testN = if (testPath.size < 2) data.count() else sc.textFile(testPath).count()
//      println(",ratio," + ratio + ",k," + k + ",itN," + itN + ",trainN," + train.count() * ratio / 10000.0 + ",testN," + testN / 10000.0 + ",numFeatures," + train.first().size)
//      System.out.println(",WSSSE, " + WSSSE + ",[" + zfkmeans.costHis.mkString(","))
//      jedis.close()
//      train.unpersist()
//
//    }
//    println()
//    println(this.getClass.getName + ",data," + dataPath)
//    println("ratio,MSE,nnMapT")
//    for (i <- ratioL.indices) {
//      println(ratioL(i) / rDivN.toDouble + "," + mesb(i) + "," + nntimesb(i))
//    }
//
//
//  }
//}
