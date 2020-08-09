package AccurateML.kmeans

import AccurateML.blas.ZFBLAS
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
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


class ZFKmeansPartStatistic(k: Int, itN: Int, sc: SparkContext) extends Serializable {

  val costHis = new ArrayBuffer[Double]()
  val mapT = sc.longAccumulator
  val changeCenterN = new ArrayBuffer[Long]()

  def runAlgorithm(data: RDD[(Vector, Array[Int])], origCenters: Array[Vector], disFunc: String): Array[Vector] = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }
    val sc = data.sparkContext

    val centers = origCenters.clone() //data.takeSample(false, k)
    println("centers -1 :\t" + centers.map(_.toArray.sum).sum)
    var newData: RDD[(Vector, Array[Int])] = data.map(line => line)
    var nowData: RDD[(Vector, Array[Int])] = data.map(line => line)
    //    println("newData sum: " + newData.map(_._2(0)).sum() + "\t nowData sum " + nowData.map(_._2(0)).sum())

    for (it <- 0 until itN) {
      val bcCenters = sc.broadcast(centers)
      val costAccum = sc.doubleAccumulator
      val computeN = sc.longAccumulator
      val changeN = sc.longAccumulator

      newData = nowData.mapPartitions { points =>
        val nnMapT = System.currentTimeMillis()
        val thisCenters = bcCenters.value
        val dims = thisCenters(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        val ans = new ArrayBuffer[(Vector, Array[Int])]()
        points.foreach { point =>
          val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point._1, thisCenters, disFunc)
          counts(centerIndex) += 1
          ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
          if (centerIndex != point._2(0)) {
            changeN.add(1)
          }
          costAccum.add(cost)
          computeN.add(1)
          point._2(0) = centerIndex
          ans += point
        }
        ans.iterator
      }.persist(StorageLevel.MEMORY_AND_DISK)
      //      println(it + "\tnewData sum: " + newData.map(_._2(0)).sum() + "\t nowData sum " + nowData.map(_._2(0)).sum())

      val totalContibs = newData.mapPartitions { points =>
        val nnMapT = System.currentTimeMillis()
        val thisCenters = bcCenters.value
        val dims = thisCenters(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        points.foreach { point =>
          val centerIndex = point._2(0)
          counts(centerIndex) += 1
          ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
          point._2(0) = centerIndex
        }
        val contribs = for (i <- 0 until k) yield {
          (i, (sums(i), counts(i)))
        }

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

      nowData = newData.map(t => t)
      println()
      changeCenterN += changeN.value
      println(it + "\tMapWSSSE: " + costAccum.value + "\tcomputeN " + computeN.value + "\tcenters: " + centers.map(_.toArray.sum).sum + "\t accMapT: " + mapT.value + "\t changeN :" + changeN.value)
      costHis += costAccum.value
    } //end-it
    centers
  }


}


object ZFKmeansPartStatistic {


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
    val changeNs = new ArrayBuffer[ArrayBuffer[Long]]()
    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble

      val data: RDD[LabeledPoint] = if (isSparse) {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
      } else {
        sc.textFile(dataPath, minPartN).map(s => {
          val vs = s.split(",|\\s+")
          new LabeledPoint(0.0, Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble)))
        })
      }
      val train = data.map(_.features).sample(false, ratio).map(v => (v, Array(-1))).persist(StorageLevel.MEMORY_AND_DISK)
      train.count()
      val origCenters = new Array[Vector](k)
      val iter = Source.fromFile(centerPath).getLines()
      var tempk = 0
      while (iter.hasNext && tempk < k) {
        val line = iter.next()
        if (!isSparse) {
          val vs = line.split(",|\\s+").map(_.toDouble)
          origCenters(tempk) = Vectors.dense(vs.slice(0, vs.size - 1))
        } else {
          val vs = line.split(",|\\s+")
          val features = vs.slice(1, vs.size).map(s => s.split(":"))
          val indexs = features.map(arr => arr(0).toInt)
          val values = features.map(arr => arr(1).toDouble)
          origCenters(tempk) = Vectors.sparse(numFeatures, indexs, values)
        }
        tempk += 1
      }
      //      for (c <- 0 until k) {
      //        val a = Array.fill(numFeatures)(0.0)
      //        for (i <- 0 until numFeatures)
      //          a(i) = Random.nextDouble()
      //        origCenters(c) = Vectors.dense(a)
      //      }

      val t1 = System.currentTimeMillis
      val zfkmeans = new ZFKmeansPartStatistic(k, itN, sc)
      val centers = zfkmeans.runAlgorithm(train, origCenters, disFunc)
      val runtime = (System.currentTimeMillis - t1)

      val nnMapT = zfkmeans.mapT.value

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
      changeNs += zfkmeans.changeCenterN
      println()
      System.out.println("ratio," + ratio + ", runTime," + runtime + ",WSSSE, " + WSSSE + ",changeN, " + zfkmeans.changeCenterN.sum + ",[," + zfkmeans.changeCenterN.mkString(","))
      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",nnMapT," + nnMapT + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
      val testN = if (testPath.size < 2) data.count() else sc.textFile(testPath).count()
      println("WSSSEs,[," + zfkmeans.costHis.mkString(",") + ",k," + k + ",itN," + itN + ",trainN," + train.count() / 10000.0 + ",testN," + testN / 10000.0 + ",numFeatures," + train.first()._1.size)
      train.unpersist()

    }
    println()
    println(this.getClass.getName + ",data," + dataPath)
    println("ratio,MSE,Time,MapT")
    for (i <- ratioL.indices) {
      println()
      println(ratioL(i) / rDivN.toDouble + "," + mesb(i) + "," + nnTimes(i) + "," + nnMapTimes(i))
      println("changeN: " + changeNs(i).sum + ",[," + changeNs(i).mkString(","))
    }

  }
}
