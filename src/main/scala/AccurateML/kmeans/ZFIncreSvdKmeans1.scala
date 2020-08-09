package AccurateML.kmeans

import AccurateML.blas.{ZFBLAS, ZFUtils}
import AccurateML.lsh.ZFHash
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 17/1/12.
  */


class ZFIncreSvdKmeans1(k: Int, itN: Int, redisHost: String, testData: RDD[LabeledPoint]) extends Serializable {


  val mapT = testData.sparkContext.longAccumulator
  val zipMapT = testData.sparkContext.longAccumulator
  val pointMapT = testData.sparkContext.longAccumulator
  var wsseT = 0L
  val computeZipN = testData.sparkContext.longAccumulator
  val computePointN = testData.sparkContext.longAccumulator


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
  def runAlgorithm(data: RDD[(LabeledPoint, Array[LabeledPoint])], origCenters: Array[Vector], ratio: Double, disFunc: String): Array[Vector] = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }
    val runStartTime = System.currentTimeMillis()
    var nowData: RDD[((LabeledPoint, Array[Double]), ArrayBuffer[(LabeledPoint, Array[Double])])] = data.mapPartitions(objectPoints => {
      val nnMapT = System.currentTimeMillis()
      val ans = new ArrayBuffer[Tuple2[Tuple2[LabeledPoint, Array[Double]], ArrayBuffer[Tuple2[LabeledPoint, Array[Double]]]]]()

      while (objectPoints.hasNext) {
        val objectP = objectPoints.next()
        val zip: LabeledPoint = objectP._1
        val points = objectP._2
        val mapPoints = new ArrayBuffer[Tuple2[LabeledPoint, Array[Double]]]()
        points.foreach(point => {
          mapPoints += Tuple2(point, Array(0.0, 0.0))
        })
        ans += Tuple2(Tuple2(zip, Array(0.0, 0.0, 0.0)), mapPoints) // (lastLabel,lastCost,lastSortValue)
      }
      mapT.add((System.currentTimeMillis() - nnMapT))
      ans.iterator
    })
    var newData = nowData.map(t => t)
    val centers = origCenters.clone()
    //    println("centers -1 :\t" + centers.map(_.toArray.sum).sum)

    var accItTime = System.currentTimeMillis() - runStartTime
    for (it <- 0 until itN) {
      val itStartTime = System.currentTimeMillis()
      val bcCenters = testData.sparkContext.broadcast(centers)
//      val cancenN = testData.sparkContext.longAccumulator


      newData = nowData.mapPartitions { mapPoints =>
        val nnMapT = System.currentTimeMillis()
        val (it1, it2) = mapPoints.duplicate
        val ans = new ArrayBuffer[((LabeledPoint, Array[Double]), ArrayBuffer[(LabeledPoint, Array[Double])])]()
        val thisCenters = bcCenters.value

        var tempTime = 0L
        var cenCostAB = new ArrayBuffer[Tuple4[Int, Double, Double, Int]]() //centerIndex,cost,gap,zipIndex
        it1.zipWithIndex.foreach { mapP =>
          val zipTuple = mapP._1._1
          val zipIndex = mapP._2
          tempTime = System.currentTimeMillis()
          val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(zipTuple._1.features, thisCenters, disFunc)
          zipMapT.add(System.currentTimeMillis() - tempTime)
          computeZipN.add(1)
          zipTuple._2(0) = centerIndex
          val sortValue = math.abs(zipTuple._2(1) - cost)
          zipTuple._2(1) = cost
          //zipTuple._2(2) = sortValue
          cenCostAB += Tuple4(centerIndex, cost, sortValue, zipIndex)
        }
        val zipSortIndex = if (it == 0) {
          Array(0)
        } else {
          val sortValueWithIndex = cenCostAB.map(t4 => (t4._3, t4._4)).sortWith(_._1 > _._1)
          val zipSortIndex = sortValueWithIndex.slice(0, (ratio * sortValueWithIndex.size).toInt).map(_._2)
          zipSortIndex.toArray
        }

        it2.zipWithIndex.foreach(t => {
          val (zipTuple, pointsTuple) = t._1
          val cancel = pointsTuple.map(_._2(0)).sum
//          cancenN.add(cancel.toLong)
          if (it == 0 || zipSortIndex.contains(t._2)) {
            tempTime = System.currentTimeMillis()
            pointsTuple.foreach(pt => {
              val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(pt._1.features, thisCenters, disFunc)
              pt._2(0) = centerIndex
              pt._2(1) = cost
            })
            pointMapT.add(System.currentTimeMillis() - tempTime)
            computePointN.add(pointsTuple.size)
          }
          ans += Tuple2(zipTuple, pointsTuple)
        })
        mapT.add((System.currentTimeMillis() - nnMapT))
        ans.iterator
      }.persist(StorageLevel.MEMORY_AND_DISK)



      val totalContibs = newData.mapPartitions(pit => {
        val nnMapT = System.currentTimeMillis()
        val dims = bcCenters.value(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        pit.foreach(t => {
          val zipTuple = t._1
          val pointsTuple = t._2
          pointsTuple.foreach(pt => {
            val centerIndex = pt._2(0).toInt
            val cost = pt._2(1)
            counts(centerIndex) += 1
            ZFBLAS.axpy(1.0, pt._1.features, sums(centerIndex))
          })
        })
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

    } //end-it
    centers
  }


}


object ZFIncreSvdKmeans1 {

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
    val conf = new SparkConf().setAppName("ZFIncreSvdKmeans1")
    val sc = new SparkContext(conf)

    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val test100: Array[Double] = args(5).split(",").map(_.toDouble)
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt
    val disFunc = args(8)

    val itqitN = args(9).toInt
    val itqratioN = args(10).toInt //from 1 not 0
    val upBound = args(11).toInt

    val testPath = args(12)
    val redisHost = args(13) //"172.18.11.97"


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
    val oHash = new ZFHash(0, itqitN, itqratioN, upBound, 2, isSparse, redisHost, sc)
    val objectData: RDD[(LabeledPoint, Array[LabeledPoint])] = data
      .mapPartitions(oHash.zfHashMap) //incrementalSVD
      .map(t3 => Tuple2(t3._1(0).last, t3._1(1).toArray)).persist(StorageLevel.MEMORY_AND_DISK)
    val on = objectData.count()
    println("zipTime,\t" + (System.currentTimeMillis() - zipTime))

    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions)
    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val HashMapT = oHash.mapT.value
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    println("objectDataN," + on + ",itqbitN," + 0 + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN)
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + HashMapT + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


    val rDivN = 100
    val ratioL = test100
    val mesb = new ArrayBuffer[Double]()
    val nnMapTimes = new ArrayBuffer[Long]()
    val nnComputeZipVSPointNs = new ArrayBuffer[String]()
    val nnTimes = new ArrayBuffer[Long]()

    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble
      val train = objectData
      val origCenters = ZFKmeansPart.initCenters(centerPath, k, isSparse, numFeatures)

      val t1 = System.currentTimeMillis
      val zfkmeans = new ZFIncreSvdKmeans1(k, itN, redisHost, test)
      val centers = zfkmeans.runAlgorithm(train, origCenters, ratio, disFunc)
      val runtime = (System.currentTimeMillis - t1) - zfkmeans.wsseT

      val nnMapT = zfkmeans.mapT.value
      val WSSSE = ZFKmeansPart.zfComputeCost(test.map(_.features), centers, disFunc)
      mesb += WSSSE
      nnMapTimes += nnMapT //nnMapT.sum
      nnComputeZipVSPointNs += zfkmeans.computeZipN.value + "," + zfkmeans.computePointN.value
      nnTimes += runtime
      println()
      println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",trainPart," + train.getNumPartitions)
      //      println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",trainPart," + train.getNumPartitions + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
      val testN = if (testPath.size < 2) data.count() else sc.textFile(testPath).count()
      println("ratio," + ratio + ", WSSSE, " + WSSSE + ", runTime," + runtime + ",wsseT, " + zfkmeans.wsseT + "," + nnComputeZipVSPointNs.last + ",nnMapT," + nnMapT + ",k," + k + ",itN," + itN + ",oN," + train.count() / 10000.0 + ",testN," + testN / 10000.0 + ",numFeatures," + data.first().features.size)
    }
    data.unpersist()

    println()
    println(this.getClass.getName + ",data," + dataPath)
    println("ratio,MSE,Time,MapT,computeZipVSPointsN")
    for (i <- ratioL.indices) {
      println(ratioL(i) / rDivN.toDouble + "," + mesb(i) + "," + nnTimes(i) + "," + nnMapTimes(i) + "," + nnComputeZipVSPointNs(i))
    }
  }
}
