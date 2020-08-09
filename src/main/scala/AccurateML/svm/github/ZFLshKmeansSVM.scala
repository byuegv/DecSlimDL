package AccurateML.svm.github

import com.lendap.spark.lsh.ZFLshRoundIndexRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 17/8/25.
  */


object ZFLshKmeansSVM {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")



    ///Users/zhangfan/Documents/data/a9a - 0 0 10,50,100 true 123 1
    val dataPath = args(0)
    val testPath = args(1)
    val packN = args(2).toInt
    val itN = args(3).toInt
    val ratioL = args(4).split(",").map(_.toDouble)
    val isSparse = args(5).toBoolean
    val numFeature = args(6).toInt
    val kernel = args(7)
    val step = args(8).toDouble
    val gamma = args(9).toDouble
    val minPartN = args(10).toInt
    val checkPointDir = args(11) //hdfs://172.18.11.90:9000/checkpoint

    val lshPerBucketN = args(12).toInt //should be cancel
    val lshRoundN = args(13).toInt
    val kmeansItN = args(14).toInt
    val perCheckN = args(15).toInt

    println("kernel, " + kernel + "\tstep, " + step + "\tgamma, " + gamma + "\tperBucketN, " + lshPerBucketN + "\tlshItN, " + lshRoundN + "\tkmeansItN, " + kmeansItN)

    val splitChar = ","
    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features)) // must be sparse
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }


    //zip
    val zipTime = System.currentTimeMillis()
    var indexedData: IndexedRDD[Long, (LabeledPoint, Double)] = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })

    val lshData = indexedData.map { case (id, (lp, beta)) => (id, lp.features) }
    val zfRoundLsh = new ZFLshRoundIndexRDD(lshData, numFeature, lshRoundN, lshPerBucketN)
    val lshZipIndex = zfRoundLsh.zfRoundLsh().collect()

    val bigZipIndex: Array[Array[Long]] = lshZipIndex.filter(_.size > lshPerBucketN / 2)
    val litZipIndex0: Array[Array[Long]] = lshZipIndex.filter(_.size <= lshPerBucketN / 2)
    val tempZipIndex: Array[Long] = litZipIndex0.flatMap(_.toIterator)
    println("lshTime: " + (System.currentTimeMillis() - zipTime))

    //    val littleZipIndex: Array[Array[Long]] = if (tempZipIndex.size <= lshPerBucketN) {
    //      Array(tempZipIndex)
    //    } else {
    //      val kmeansData: RDD[(Long, (LabeledPoint, Double))] = indexedData.filter(p => tempZipIndex.contains(p._1)).map(p => p).persist(StorageLevel.MEMORY_AND_DISK)
    //      //      val kmeansData = indexedData.multiget(tempZipIndex).toArray
    //      val K = tempZipIndex.size / lshPerBucketN
    //      val kmeansTrainData = kmeansData.map(_._2._1.features).persist(StorageLevel.MEMORY_AND_DISK)
    //      val kmodel = KMeans.train(kmeansTrainData, K, kmeansItN - 1)
    //      val temp: Array[(Int, Long)] = kmeansData.map(t3 => {
    //        val centerId = kmodel.predict(t3._2._1.features)
    //        (centerId, t3._1)
    //      }).collect()
    //      val longIndexs = new Array[ArrayBuffer[Long]](K)
    //      longIndexs.indices.foreach(i => longIndexs(i) = new ArrayBuffer[Long]())
    //      for (i <- 0 until temp.size) {
    //        val centerId = temp(i)._1
    //        longIndexs(centerId) += temp(i)._2
    //      }
    //      longIndexs.filter(_.size > 0).map(_.toArray)
    //    }
    //    val zipIndex: RDD[Array[Long]] = if (tempZipIndex.size > 0) {
    //      //avoid empty
    //      sc.parallelize(bigZipIndex ++ littleZipIndex)
    //    } else {
    //      sc.parallelize(bigZipIndex)
    //    }
    val zipIndex = sc.parallelize(lshZipIndex)


    val zipData = zipIndex.zipWithUniqueId().map { case (ids, zipId) => {
      val lp: LabeledPoint = new LabeledPoint(0.0, Vectors.zeros(numFeature))
      (zipId, Tuple3(lp, 0.0D, ids))
    }
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val cancelz = zipData.collect()
    val cancelzN: Array[Int] = cancelz.map(_._2._3.size).sortWith(_ < _)
    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))
    val realPerBucketN = (cancelzN.sum / cancelzN.size.toDouble).toInt
    println("zipN:\t" + cancelz.size + "\t" + realPerBucketN + "\t" + cancelzN.slice(0, 10).mkString(",") + "\t" + cancelzN.slice(cancelzN.size - 10, cancelzN.size).mkString(","))


    val tempP = new ArrayBuffer[String]()
    for (r <- ratioL) {
      val ratio = r / 100.0
      val s = checkPointDir splitAt (checkPointDir lastIndexOf ("/"))
      val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
      val tPath = new org.apache.hadoop.fs.Path(s._2)
      if (fs.exists(tPath)) {
        fs.delete(tPath, true) // isRecusrive= true
      }
      fs.mkdirs(new Path(s._2))
      fs.close()

      val splits = indexedData.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) IndexedRDD(indexedData.sample(false, ratio)).persist(StorageLevel.MEMORY_AND_DISK) else IndexedRDD(splits(0).sample(false, ratio)).persist(StorageLevel.MEMORY_AND_DISK)
      val test: RDD[LabeledPoint] = if (testPath.size > 3) {
        if (!isSparse) {
          sc.textFile(testPath, minPartN).map(line => {
            val vs = line.split(splitChar).map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1).map(_._2._1)

      println("partN:\t" + train.getNumPartitions)
      val m = train.count()
      //      num_iter = (2 * m).toInt
      val t1 = System.currentTimeMillis
      val svm = new ZFHashSVMTrain(train, zipData.collect(), test, step, kernel, gamma, checkPointDir) //gamma = 0.1
      //      val svm = new ZFZipKernelSVM(train, zipData, 1.0 / m, kernel, gamma, checkPointDir) //gamma = 0.1
      svm.train(itN * packN, packN, perCheckN)
      //      svm.train(itN * packN, packN, realPerBucketN, perCheckN)
      val t2 = System.currentTimeMillis
      val runtime = (t2 - t1)

      println("model count:\t" + svm.model.count())
      println("Ratio\tDataN\titN\tpackN\tT")
      var ss = ratio + "\t" + m + "\t" + itN + "\t" + packN + "\t" + runtime + "\n"

      System.out.println(ss)

      val testT = System.currentTimeMillis()
      val acc = svm.getAccuracy(test)
      println("ACC:\t" + acc + "\ttestT: " + (System.currentTimeMillis() - testT))

      tempP += ss
      data.unpersist()
      train.unpersist()
      val fs2 = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
      fs2.delete(new org.apache.hadoop.fs.Path(s._2), true) // isRecusrive= true
      fs2.close()
    }
    println("kernel, " + kernel + "\tstep, " + step + "\tgamma, " + gamma + "\tperBucketN, " + lshPerBucketN + "\tlshItN, " + lshRoundN + "\tkmeansItN, " + kmeansItN)

    //    println("Ratio\tDataN\titN\tpackN\tACC\tT")
    //    println(tempP.mkString("\n"))

  }
}
