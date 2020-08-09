package AccurateML.kmeans

import java.time.Instant

import AccurateML.blas.{ZFBLAS, ZFUtils}
import AccurateML.lsh.ZFHashLayer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, TaskContext}
import redis.clients.jedis.Jedis

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer


/**
  * Created by zhangfan on 17/1/12.
  */


class ZFIncreSvdKmeansLayerTask(k: Int, itN: Int, redisHost: String, testData: RDD[LabeledPoint]) extends Serializable {


  val mapT = testData.sparkContext.longAccumulator
  val zipMapT = testData.sparkContext.longAccumulator
  val pointMapT = testData.sparkContext.longAccumulator
  var wsseT = 0L
  val computeZipN = testData.sparkContext.longAccumulator
  val computePointN = testData.sparkContext.longAccumulator

  val lastStageIdAcc = testData.sparkContext.collectionAccumulator[Long]
  val taskIdArr = new ArrayBuffer[Long]
  val executorRunTimeArr = new ArrayBuffer[Long]
  val executorIdArr = new ArrayBuffer[String]()
  val hostArr = new ArrayBuffer[String]()

  def zftestCost(centers: Array[Vector], disFunc: String): Double = {
    val wssse = ZFKmeansPart.zfComputeCost(testData.map(lp => lp.features), centers, disFunc)
    wssse
  }


  /**
    * sortValue |cost - lastCost|
    *
    * @param data
    * @param origCenters
    * @param ratio
    * @param disFunc
    * @return
    */
  def runAlgorithm(data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]])], origCenters: Array[Vector], ratio: Double, disFunc: String, restAPI: String): Array[Vector] = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }
    val runStartTime = System.currentTimeMillis()
    var nowData: RDD[Tuple2[Array[ArrayBuffer[(LabeledPoint, Array[Double])]], Array[ArrayBuffer[Array[Int]]]]] = data.mapPartitions(objectPoints => {
      val nnMapT = System.currentTimeMillis()
      val ans = new ArrayBuffer[Tuple2[Array[ArrayBuffer[(LabeledPoint, Array[Double])]], Array[ArrayBuffer[Array[Int]]]]]()
      while (objectPoints.hasNext) {
        val o: (Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]]) = objectPoints.next()
        ans += Tuple2(o._1.map(lps => lps.map(Tuple2(_, Array(0.0, 0.0, 0.0)))), o._2)
      }
      mapT.add((System.currentTimeMillis() - nnMapT))
      ans.iterator
    })
    var newData = nowData.map(t => t)
    val centers = origCenters.clone()
    val numPartition = nowData.getNumPartitions
    var accItTime = System.currentTimeMillis() - runStartTime

    /**
      * 对于稀疏数据在第一次迭代时间最短,因为center稀疏,计算的属性个数少
      *
      * 设定只有两层压缩点,先计算全部第0层压缩点的sortValue排序,然后如果时间允许（根据上一次记录的平均时间决定）依序计算第1层压缩点的sortValue,
      * 然后两层压缩点的sortValue一起排序,按序展开原始点（直到运行时间>95%记录时间）
      *
      */
    for (it <- 0 until itN) {
      val itStartTime = System.currentTimeMillis()
      val bcCenters = testData.sparkContext.broadcast(centers)
      val bcIt = testData.sparkContext.broadcast(it)

      val itMapTs = testData.sparkContext.collectionAccumulator[Long]
      if (it > 1) {
        //get completed stage task runtime , dont record the first it, because the first time is much less because of sparse data
        val c = System.currentTimeMillis()
        taskIdArr.clear()
        executorIdArr.clear()
        hostArr.clear()
        executorRunTimeArr.clear()
        getCompletedTaskRunTime(restAPI + "/" + lastStageIdAcc.value.last, numPartition)
        println(it + ", getCompletedTaskRunTime :" + (System.currentTimeMillis() - c))
      }
      val continueTime = if (it > 1) {
        val sortV = executorRunTimeArr.sortWith(_ < _)
        sortV(((sortV.size - 1) * 0.95).toInt)
      } else {
        0
      }
      val meanTime = executorRunTimeArr.sum.toDouble / executorRunTimeArr.size

      println(bcIt.value + ", continueTime, " + continueTime)
      val bcContinueTime = testData.sparkContext.broadcast(continueTime)
      val bcMeanTime = testData.sparkContext.broadcast(meanTime)
      lastStageIdAcc.reset()

      newData = nowData.mapPartitions { opit =>
        val nnMapT = System.currentTimeMillis()
        val partData: Array[(Array[ArrayBuffer[(LabeledPoint, Array[Double])]], Array[ArrayBuffer[Array[Int]]])] = opit.toArray
        val thisCenters = bcCenters.value
        val bcit = bcIt.value
        val ctx = TaskContext.get
        var tempTime = 0L
        if (lastStageIdAcc.isZero) {
          lastStageIdAcc.add(ctx.stageId())
        }
        //计算第0层压缩点sortValue
        val sortValues0 = new ArrayBuffer[Tuple2[Int, Double]]()
        tempTime = System.currentTimeMillis()
        partData.zipWithIndex.foreach(tit => {
          val pi = tit._2
          val oit = tit._1
          val zipAndPoints: Array[ArrayBuffer[(LabeledPoint, Array[Double])]] = oit._1
          zipAndPoints(0).foreach(t => {
            val zip = t._1
            val zipInfo = t._2
            val (centerIndex, cost, sortValue) = ZFKmeansPart.zfFindClosest2(zip.features, thisCenters, disFunc)
            computeZipN.add(1)
            zipInfo(0) = centerIndex
            zipInfo(1) = cost
            zipInfo(2) = sortValue
            sortValues0 += Tuple2(pi, sortValue)
          })
        })
        zipMapT.add(System.currentTimeMillis() - tempTime)

        var thisTaskLaunchTime = 0L
        if (bcit <= 1) {
          // 第0次迭代,计算全部原始点;   第1次迭代,只按照0层压缩点排序,计算前ratio%
          val pi0 = if (bcit == 0) Array.range(0, partData.size).iterator else sortValues0.sortWith(_._2 < _._2).map(_._1).slice(0, (ratio * sortValues0.size).toInt).iterator
          tempTime = System.currentTimeMillis()
          while (pi0.hasNext) {
            val pi = pi0.next()
            val oit = partData(pi)
            val zipAndPoints = oit._1
            zipAndPoints.last.foreach(pointsT => {
              val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(pointsT._1.features, thisCenters, disFunc)
              computePointN.add(1)
              pointsT._2(0) = centerIndex
              pointsT._2(1) = cost
            })
          }
          pointMapT.add(System.currentTimeMillis() - tempTime)
        } else {
          //bcit>1
          //如果时间允许,按序计算第1层压缩点sortValue
          val selectZipIndex: Map[Int, ArrayBuffer[(Double, Int, Int, Int)]] = {
            val sortValues1 = new ArrayBuffer[Tuple4[Double, Int, Int, Int]]()
            val pi0 = sortValues0.sortWith(_._2 < _._2).map(_._1).iterator

            val (launchTime, meanDiffTime) = getLaunchAndDiffTime(restAPI + "/" + ctx.stageId(), ctx.taskAttemptId(), bcMeanTime.value)
            thisTaskLaunchTime = launchTime

            val layer1Time = System.currentTimeMillis()
            while (pi0.hasNext && (System.currentTimeMillis() - layer1Time) < meanDiffTime) {
              val pi = pi0.next()
              val zipAndPoints: Array[ArrayBuffer[(LabeledPoint, Array[Double])]] = partData(pi)._1
              val lastCost = zipAndPoints(0).last._2(1)
              tempTime = System.currentTimeMillis()
              zipAndPoints(1).zipWithIndex.foreach(tt => {
                val j = tt._2
                val zip = tt._1._1
                val (_, cost, sortValue) = ZFKmeansPart.zfFindClosest2(zip.features, thisCenters, disFunc)
                computeZipN.add(1)
                sortValues1 += Tuple4(sortValue, pi, 1, j)
              })
              zipMapT.add(System.currentTimeMillis() - tempTime)
            }
            //之后只用第0层压缩点作为sortValue
            while (pi0.hasNext) {
              val pi = pi0.next()
              val zipAndPoints: Array[ArrayBuffer[(LabeledPoint, Array[Double])]] = partData(pi)._1
              sortValues1 += Tuple4(zipAndPoints(0).last._2(2), pi, 0, 0)
            }
            sortValues1.sortWith(_._1 < _._1).groupBy(_._2)
            //            sortValues1.sortWith(_._1 > _._1).slice(0, (sortValues1.size * ratio).toInt).groupBy(_._2)
          } // end-selectZipIndex


          val tempIndexs = selectZipIndex.flatMap(_._2)
          val zipI = tempIndexs.slice(0, (tempIndexs.size * ratio).toInt).iterator
          while (zipI.hasNext) {
            tempTime = System.currentTimeMillis()
            val (v, pi, i, j) = zipI.next()
            val points: ArrayBuffer[(LabeledPoint, Array[Double])] = partData(pi)._1.last
            val indexs: Array[Int] = partData(pi)._2(i)(j)
            indexs.foreach { i => {
              val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(points(i)._1.features, thisCenters, disFunc)
              computePointN.add(1)
              points(i)._2(0) = centerIndex
              points(i)._2(1) = cost
            }
            }
            pointMapT.add(System.currentTimeMillis() - tempTime)
            if ((System.currentTimeMillis() - thisTaskLaunchTime) >= bcContinueTime.value) {
              // make zipI empty
              println(bcit + ", thisTaskLaunchTime break ! " + (System.currentTimeMillis() - thisTaskLaunchTime) + " , " + bcContinueTime.value + " ,\t " + (1 - zipI.size.toDouble / selectZipIndex.map(_._2).size))
            }
          }
        }
        val c = (System.currentTimeMillis() - nnMapT)
        mapT.add(c)
        itMapTs.add(c)
        partData.iterator
      }.persist(StorageLevel.MEMORY_AND_DISK)

      val totalContibs = newData.mapPartitions(oit => {
        val nnMapT = System.currentTimeMillis()
        val dims = bcCenters.value(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        oit.map(_._1.last).foreach(arr => {
          arr.foreach(pt => {
            val centerIndex = pt._2(0).toInt
            counts(centerIndex) += 1
            ZFBLAS.axpy(1.0, pt._1.features, sums(centerIndex))
          })
        })
        //        oit.flatMap(_._1.last).foreach(pt => {
        //          val centerIndex = pt._2(0).toInt
        //          val cost = pt._2(1)
        //          counts(centerIndex) += 1
        //          ZFBLAS.axpy(1.0, pt._1.features, sums(centerIndex))
        //        })
        val contribs = for (i <- 0 until k) yield {
          (i, (sums(i), counts(i)))
        }
        mapT.add((System.currentTimeMillis() - nnMapT))
        contribs.iterator
      }).reduceByKey((a, b) => {
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
      nowData = newData.map(t => t)
      accItTime += System.currentTimeMillis() - itStartTime

      val tempT = System.currentTimeMillis()
      val WSSSE = zftestCost(centers, disFunc)
      wsseT += (System.currentTimeMillis() - tempT)
      println(it + "\tWSSSE, " + WSSSE + "\taccItTime," + accItTime + "\t accMapT, " + mapT.value + "\t wsseT, " + wsseT + "\tcomputeZipN," + computeZipN.value + ", zipMapT," + zipMapT.value + ", computePointN," + computePointN.value + ", pointMapT," + pointMapT.value)
      println(bcIt.value + " ,itMapTs, " + itMapTs.value.sortWith(_ < _).mkString(","))
    } //end-it
    //    mapdata.unpersist()
    centers
  }


  def getLaunchAndDiffTime(url: String, thisTaskId: Long, meanTime: Double): Tuple2[Long, Double] = {
    val content = scala.io.Source.fromURL(url)
    val allLines = content.mkString
    val headDetails = allLines.split("\"details\" :")
    val lines = headDetails(1).split("\n")
    val taskId = new ArrayBuffer[Long]()
    val launchTime = new ArrayBuffer[String]()
    val executorId = new ArrayBuffer[String]()
    var ii = 0
    for (line <- lines) {
      if (line.contains("\"taskId\" :")) {
        val id = line.split(":|,")(1).trim.toLong
        taskId += id
        launchTime += ""
        executorId += ""
      }
      if (line.contains("\"launchTime\" :")) {
        val s = line.split("\"")(3).replaceFirst("GMT$", "Z")
        launchTime(taskId.size - 1) = s
        if (taskId.last == thisTaskId) {
          ii = taskId.size - 1
        }
      }
      if (line.contains("\"executorId\" :")) {
        executorId(taskId.size - 1) = line.split(":|,")(1).trim
      }
    }
    assert(executorId.size == taskId.size)
    //    val thisExecutorId = executorId(thisIndex)
    //    if (sortExecutorRunTime.size == 0) true else thisTaskRuntime < sortExecutorRunTime((sortExecutorRunTime.size * empRatio).floor.toInt)
    //    val sortV = executorRunTimeArr.sortWith(_ < _)
    //    val sortV = executorRunTimeArr.zip(executorId).filter(_._2 == thisExecutorId).map(_._1).sortWith(_ < _)
    //    println("thisTaskRuntime: " + thisTaskRuntime + "\texecutorRunTimeArr: \t" + numPartition * (it - 1) + "," + xecutorRunTimeArr.size + ",[," + executorRunTimeArr.mkString(",") + "sortV: " + sortV)
    val tempT = executorRunTimeArr.zip(executorId).filter(_._2 == executorId(ii)).map(_._1)
    val diffTime = meanTime - tempT.sum.toDouble / tempT.size
    val thisLaunchTime = Instant.parse(launchTime(ii)).toEpochMilli()
    (thisLaunchTime, diffTime)
  }


  def getCompletedTaskRunTime(url: String, numPartition: Int): Unit = {
    val content = scala.io.Source.fromURL(url)
    val allLines = content.mkString

    val headDetails = allLines.split("\"details\" :")
    val head = headDetails(0).split("\n")
    val lines = headDetails(1).split("\n")

    var numCompleteTasks = 0
    var numFailedTasks = 0
    val taskId = new ArrayBuffer[Long]()
    val executorId = new ArrayBuffer[String]()
    val host = new ArrayBuffer[String]()
    //    val errorMessage = new ArrayBuffer[Boolean]()
    val executorRunTime = new ArrayBuffer[Long]()

    for (line <- head) {
      if (line.contains("\"numCompleteTasks\" :")) {
        numCompleteTasks = line.split(":|,")(1).trim.toInt
      }
      if (line.contains("\"numFailedTasks\" :")) {
        numFailedTasks = line.split(":|,")(1).trim.toInt
      }
    }
    for (line <- lines) {
      if (line.contains("\"taskId\" :")) {
        val id = line.split(":|,")(1).trim.toLong
        taskId += id
        executorId += ""
        host += ""
        //        errorMessage += false
        executorRunTime += 0L
      }
      if (line.contains("\"executorId\" :")) {
        executorId(taskId.size - 1) = line.split("\"")(3)
      }
      if (line.contains("\"host\" :")) {
        host(taskId.size - 1) = line.split(":|,")(1).trim
      }
      //      if (line.contains("\"errorMessage\" :")) {
      //        errorMessage(taskId.size - 1) = true
      //      }
      if (line.contains("\"executorRunTime\" :")) {
        executorRunTime(taskId.size - 1) = line.split(":|,")(1).trim.toLong
      }
    }

    assert(executorRunTime.size == taskId.size)
    for (i <- 0 until taskId.size) {
      //      if (!errorMessage(i)) {
      taskIdArr.add(taskId(i))
      executorIdArr.add(executorId(i))
      hostArr.add(host(i))
      executorRunTimeArr.add(executorRunTime(i))
      //      }
    }
    println(url)
    //        println("taskIdArr: \t" + taskIdArr.size + ",[," + taskIdArr.mkString(","))
    //            println("executorIdArr: \t" + executorIdArr.size + ",[," + executorIdArr.mkString(","))
    //        println("hostArr: \t" + hostArr.size + ",[," + hostArr.mkString(","))
    //            println("executorRunTimeArr: \t" + executorRunTimeArr.size + ",[," + executorRunTimeArr.mkString(","))
    executorIdArr.zip(executorRunTimeArr).groupBy(_._1).toArray.sortWith(_._1 < _._1).foreach(t => {
      val vs = t._2.map(_._2)
      println("executorId, " + t._1 + "," + vs.size + ", " + vs.sum.toDouble / vs.size + " , " + vs.sortWith(_ < _).mkString(","))
    }
    )
  }


}


object ZFIncreSvdKmeansLayerTask {

  def testKMeans(sc: SparkContext): Unit = {
    val data = sc.textFile("/Users/zhangfan/Documents/data/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //    clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
    //    val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val ratios: Array[Double] = args(5).split(",").map(_.toDouble)
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt
    val disFunc = args(8)

    val itqitN = args(9).toInt
    val itqratioN = args(10).toInt //from 1 not 0
    val upBound = args(11).toInt

    val testPath = args(12)
    val redisHost = args(13) //"172.18.11.97"
    val hashLayer = args(14).toInt


    val data: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
    } else {
      sc.textFile(dataPath, minPartN).map(s => {
        val vs = s.split("\\s+|,")
        new LabeledPoint(0.0, Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble)))
      })
    }

    val test = if (testPath.size < 2) {
      data
    } else {
      if (isSparse) {
        MLUtils.loadLibSVMFile(sc, testPath, numFeatures)
      } else {
        sc.textFile(testPath).map(s => {
          val vs = s.split("\\s+|,")
          new LabeledPoint(0.0, Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble)))
        })
      }
    }

    val jedis = new Jedis(redisHost)
    jedis.flushAll()
    val zipTime = System.currentTimeMillis()
    val oHash = new ZFHashLayer(itqitN, itqratioN, upBound, isSparse, hashLayer, redisHost, sc)
    val objectData: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]])] = data
      .mapPartitions(oHash.zfHashMap) //incrementalSVD
      .persist(StorageLevel.MEMORY_AND_DISK)
    val on = objectData.count()
    println("zipTime,\t" + (System.currentTimeMillis() - zipTime))

    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions)
    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    //    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    //    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val HashMapT = oHash.mapT.value
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    //    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    //    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    //    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    println("objectDataN," + on + ",itqbitN," + 0 + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN)
    println("readT," + readT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
    //    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


    val rDivN = 100
    val mesb = new ArrayBuffer[Double]()
    val nnMapTimes = new ArrayBuffer[Long]()
    val nnComputeZipVSPointNs = new ArrayBuffer[String]()
    val nnTimes = new ArrayBuffer[Long]()

    for (r <- ratios) {
      val ratio = r / rDivN.toDouble
      val train = objectData
      val origCenters = ZFKmeansPart.initCenters(centerPath, k, isSparse, numFeatures)

      val t1 = System.currentTimeMillis
      val zfkmeans = new ZFIncreSvdKmeansLayerTask(k, itN, redisHost, test)
      //            val restAPI = "http://" + dataPath.split("/|:")(3) + ":4040/api/v1/applications/" + sc.applicationId + "/stages"
      val restAPI = "http://" + dataPath.split("/|:")(3) + ":8088/proxy/" + sc.applicationId + "/api/v1/applications/" + sc.applicationId + "/stages"
      val centers = zfkmeans.runAlgorithm(train, origCenters, ratio, disFunc, restAPI)
      val runtime = (System.currentTimeMillis - t1) - zfkmeans.wsseT

      val nnMapT = zfkmeans.mapT.value
      val WSSSE = ZFKmeansPart.zfComputeCost(test.map(_.features), centers, disFunc)
      mesb += WSSSE
      nnMapTimes += nnMapT //nnMapT.sum
      nnComputeZipVSPointNs += zfkmeans.computeZipN.value + "," + zfkmeans.computePointN.value
      nnTimes += runtime
      println()
      println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",trainPart," + train.getNumPartitions)
      val testN = if (testPath.size < 2) data.count() else sc.textFile(testPath).count()
      println("ratio," + ratio + ", WSSSE, " + WSSSE + ", runTime," + runtime + ",wsseT, " + zfkmeans.wsseT + "," + nnComputeZipVSPointNs.last + ",nnMapT," + nnMapT + ",k," + k + ",itN," + itN + ",oN," + train.count() / 10000.0 + ",testN," + testN / 10000.0 + ",numFeatures," + data.first().features.size)
    }
    data.unpersist()

    println()
    println(this.getClass.getName + ",data," + dataPath)
    println("ratio,MSE,Time,MapT,computeZipVSPointsN")
    for (i <- ratios.indices) {
      println(ratios(i) / rDivN.toDouble + "," + mesb(i) + "," + nnTimes(i) + "," + nnMapTimes(i) + "," + nnComputeZipVSPointNs(i))
    }
  }
}
