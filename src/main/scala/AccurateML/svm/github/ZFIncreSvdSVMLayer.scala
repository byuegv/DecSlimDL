package AccurateML.svm.github

import AccurateML.lsh.ZFHashLayer
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.Array._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by zhangfan on 17/8/24.
  */
class ZFIncreSvdSVMLayer(indexedData: IndexedRDD[Long, (LabeledPoint, Double)],
                         azipData: Array[Array[(LabeledPoint, Array[Long])]],
                         test_data: RDD[LabeledPoint],
                         lambda_s: Double,
                         kernel: String,
                         gamma: Double,
                         checkpoint_dir: String) extends java.io.Serializable {

  val lambda = lambda_s
  var s = 1D
  var zip_s = s
  //Currently only rbf kernel is implemented
  var kernel_func = KernelSVM.zfSvmKerkel(gamma, kernel)
  //  var kernel_func_zf = new ZFRbfKernelFunc(gamma,training_data.first().features.size/10)
  var model = indexedData.map(x => x._2)
  val test = test_data
  var testTime = 0L
  var zipData = azipData

  var cancel1 = 0
  var cancel2 = 0


  //  sc.setCheckpointDir(checkpoint_dir)

  /** method train, based on P-Pack algorithm **/
  def train(num_iter: Int, pack_size: Int = 1, perCheckN: Int = 10) {
    //Free the current model from memory
    model.unpersist()
    //Initialization
    var working_data = indexedData
    working_data.persist(StorageLevel.MEMORY_AND_DISK)
    var norm = 0D
    var alpha = 0D
    var t = 1
    var zip_norm = 0D
    var zip_alpha = 0D
    var zip_t = 1

    val broad_kernel_func = indexedData.sparkContext.broadcast(kernel_func)

    var itTimeSum = 0L
    var takeSampleTime = 0L
    var ypTime = 0L
    var computeTime = 0L
    var multiputTime = 0L
    var checkpointTime = 0L

    var zipTime = 0L
    var zipSampleTime = 0L
    var zipYpTime = 0L
    var zipComputeTime = 0L
    var selectIndexTime = 0L

    for (num_of_updates <- 1 to num_iter) {

      val zipT = System.currentTimeMillis()
      var selectCount = 0
      val selectZipIndex = new ArrayBuffer[Tuple2[Int, Int]]()
      val zipSample = new ArrayBuffer[(LabeledPoint, Array[Long])]()
      while (selectCount < pack_size) {
        val r0 = Random.nextInt(zipData.size)
        val r1 = Random.nextInt(zipData(r0).size - 1) + 1 // 0 is the 0_zip
        if (!(selectZipIndex contains(r0, r1))) {
          zipSample += zipData(r0)(r1)
          selectZipIndex += Tuple2(r0, r1)
          selectCount += zipData(r0)(r1)._2.size
        }
      }
      zipSampleTime += System.currentTimeMillis() - zipT


      val zipYpT = System.currentTimeMillis()
      val zipYp: Array[Double] = {
        val broad_zipSample = indexedData.sparkContext.broadcast(zipSample)
        working_data.sample(false, 0.3).mapPartitions(pit => {
          val zips = broad_zipSample.value.toArray
          pit.map { case (k, v) => zips.map(x => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._1.features))) }
        }).reduce((aa, bb) => {
          for (i <- 0 until aa.size)
            aa(i) += bb(i)
          aa
        })
      }
      zipYpTime += System.currentTimeMillis() - zipYpT

      val zipComputeT = System.currentTimeMillis()
      var zip_local_set = Map[Tuple2[Int, Int], (LabeledPoint, Double)]()
      val zip_pair_idx = indexedData.sparkContext.parallelize(range(0, zipSample.size).flatMap(x => (range(x, zipSample.size).map(y => (x, y)))))
      val zip_inner_prod = zip_pair_idx.map(x => (x, broad_kernel_func.value.evaluate(zipSample(x._1)._1.features, zipSample(x._2)._1.features))).collectAsMap()

      val zipAlpha: Array[Array[Double]] = zipData.map(arr => {
        Array.fill(arr.size)(0.0)
      })
      val zipY = zipSample.map(_._1.label)

      for (i <- 0 until zipSample.size) {
        zip_t = zip_t + 1
        zip_s = (1 - 1D / (zip_t)) * zip_s
        for (j <- (i + 1) until zipSample.size) {
          zipYp(j) = (1 - 1D / (zip_t)) * zipYp(j)
        }
        if (zipY(i) * zipYp(i) < 1) {
          zip_norm = zip_norm + (2 * zipY(i)) / (lambda * zip_t) * zipYp(i) + math.pow((zipY(i) / (lambda * zip_t)), 2) * zip_inner_prod((i, i))
          val (zi, zj) = selectZipIndex(i)
          zip_alpha = zipAlpha(zi)(zj)
          zip_local_set = zip_local_set + (selectZipIndex(i) ->(zipSample(i)._1, zip_alpha + (1 / (lambda * zip_t * zip_s))))
          for (j <- (i + 1) to (zipSample.size - 1)) {
            zipYp(j) = zipYp(j) + zipY(j) / (lambda * zip_t) * zip_inner_prod((i, j))
          }
          if (zip_norm > (1 / lambda)) {
            zip_s = zip_s * (1 / math.sqrt(lambda * zip_norm))
            zip_norm = (1 / lambda)
            for (j <- (i + 1) to (zipSample.size - 1)) {
              zipYp(j) = zipYp(j) / math.sqrt(lambda * zip_norm)
            }
          }
        }
      }
      zipComputeTime += System.currentTimeMillis() - zipComputeT

      val selectIndexT = System.currentTimeMillis()
      zip_local_set.foreach { case (k, v) => {
        val (zi, zj) = k
        if (zj > 0) zipAlpha(zi)(0) += v._2 - zipAlpha(zi)(zj)
        zipAlpha(zi)(zj) = v._2
      }
      }
      val selectZipAlpha = selectZipIndex.map { case (zi, zj) => zipAlpha(zi)(zj) }
      val temp = if (selectZipAlpha.sum == 0) selectZipIndex.zip(selectZipAlpha) else selectZipIndex.zip(selectZipAlpha).filter(_._2 > 0)
      //cancel
      val cancelAllIndex = selectZipIndex.map(t => {
        val (zi, zj) = t
        val ans: Array[Long] = if (zj == 0) {
          zipData(zi).flatMap(_._2)
        } else {
          zipData(zi)(zj)._2
        }
        ans.size
      }).sum
      cancel1 += cancelAllIndex
      val cancelAfterZipFilterIndex = temp.map(_._1).map(t => {
        val (zi, zj) = t
        val ans: Array[Long] = if (zj == 0) {
          zipData(zi).flatMap(_._2)
        } else {
          zipData(zi)(zj)._2
        }
        ans.size
      }).sum
      cancel2 += cancelAfterZipFilterIndex
      //cancel-end
      val indexs: Array[Long] = {
        val allIndexs = temp.flatMap(t => {
          val (zi, zj) = t._1
          val ans: Array[Long] = if (zj == 0) {
            zipData(zi).flatMap(_._2)
          } else {
            zipData(zi)(zj)._2
          }
          ans
        })
        Random.shuffle(allIndexs.toList).take(if (selectZipAlpha.sum == 0) 1 else pack_size).toArray
      }
      selectIndexTime += System.currentTimeMillis() - selectIndexT
      zipTime += System.currentTimeMillis() - zipT

      val itStartTime = System.currentTimeMillis()
      val sample: Array[(Long, (LabeledPoint, Double))] = working_data.multiget(indexs).toArray
      println(num_of_updates + " zipSampleN, " + zipSample.size + " sampleN, " + sample.size)
      val sampleN = sample.size
      takeSampleTime += (System.currentTimeMillis() - itStartTime)

      /** how to caculate yp filter or not(original) */
      val ypT = System.currentTimeMillis()
      val yp = {
        val broad_sample = working_data.sparkContext.broadcast(sample)
        working_data.mapPartitions(pit => {
          val points = broad_sample.value.map(_._2)
          pit.map { case (k, v) => points.map(x => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._1.features))) }
        }).reduce((aa, bb) => {
          for (i <- 0 until aa.size)
            aa(i) += bb(i)
          aa
        })
      }
      ypTime += System.currentTimeMillis() - ypT

      val tempComputeTime = System.currentTimeMillis()
      val y = sample.map(x => x._2._1.label)
      var local_set = Map[Long, (LabeledPoint, Double)]()
      val pair_idx = indexedData.sparkContext.parallelize(range(0, sampleN).flatMap(x => (range(x, sampleN).map(y => (x, y)))))
      val inner_prod = pair_idx.map(x => (x, broad_kernel_func.value.evaluate(sample(x._1)._2._1.features, sample(x._2)._2._1.features))).collectAsMap()
      //      // Compute sub gradients
      for (i <- 0 until sampleN) {
        t = t + 1
        s = (1 - 1D / (t)) * s
        for (j <- (i + 1) until (sampleN)) {
          yp(j) = (1 - 1D / (t)) * yp(j)
        }
        if (y(i) * yp(i) < 1) {
          norm = norm + (2 * y(i)) / (lambda * t) * yp(i) + math.pow((y(i) / (lambda * t)), 2) * inner_prod((i, i))
          alpha = sample(i)._2._2
          local_set = local_set + (sample(i)._1 ->(sample(i)._2._1, alpha + (1 / (lambda * t * s))))
          for (j <- (i + 1) to (sampleN - 1)) {
            yp(j) = yp(j) + y(j) / (lambda * t) * inner_prod((i, j))
          }

          if (norm > (1 / lambda)) {
            s = s * (1 / math.sqrt(lambda * norm))
            norm = (1 / lambda)
            for (j <- (i + 1) to (sampleN - 1)) {
              yp(j) = yp(j) / math.sqrt(lambda * norm)
            }
          }
        }
      }
      computeTime += System.currentTimeMillis() - tempComputeTime
      val tempMultiputTime = System.currentTimeMillis()
      val to_forget = working_data
      //      val cancelN = (sample.size * 0.5).toInt
      //      val cancelMap: Map[Long, (LabeledPoint, Double)] = local_set.toArray.sortBy(t => math.abs(t._2._2)).slice(0, cancelN).toMap
      //      working_data = working_data.multiput(cancelMap).persist(StorageLevel.MEMORY_AND_DISK)
      working_data = working_data.multiput(local_set).persist(StorageLevel.MEMORY_AND_DISK)
      working_data.count() //cancel
      to_forget.unpersist()
      multiputTime += System.currentTimeMillis() - tempMultiputTime

      //checkpoint
      val checkpointT = System.currentTimeMillis()
      if (num_of_updates % perCheckN == 0) {
        //        println(num_of_updates + "\tcheckpoint" + "\tworking-data partN:\t" + working_data.getNumPartitions)
        //        val s = checkpoint_dir splitAt (checkpoint_dir lastIndexOf ("/"))
        //        val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
        //        val lastN = num_of_updates - perCheckN
        //        working_data.map(t => t).saveAsObjectFile(checkpoint_dir + "/working_data/" + num_of_updates)
        //        if (lastN > 0) {
        //          fs.delete(new Path(checkpoint_dir + "/working_data/" + lastN), true)
        //        }
        //        working_data = IndexedRDD(sc.objectFile[(Long, (LabeledPoint, Double))](checkpoint_dir + "/working_data/" + num_of_updates))
        //        working_data.persist(StorageLevel.MEMORY_AND_DISK)
        //        working_data.count() //cancel
        //        fs.close()
        indexedData.sparkContext.setCheckpointDir(checkpoint_dir + "/" + num_of_updates)
        working_data.checkpoint() //会增大hdfs磁盘使用量,超过一定阈值报错 safe mode cant change,所以要手动删除文件
        working_data.count()
        val s = checkpoint_dir splitAt (checkpoint_dir lastIndexOf ("/"))
        val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), indexedData.sparkContext.hadoopConfiguration)
        val lastN = num_of_updates - perCheckN
        if (lastN > 0) {
          fs.delete(new Path(checkpoint_dir + "/" + lastN), true)
        }
      }
      //get acc
      checkpointTime += System.currentTimeMillis() - checkpointT
      itTimeSum += System.currentTimeMillis() - itStartTime
      val testT = System.currentTimeMillis()
      if (num_of_updates % 10 == 0) {
        model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
        val acc = getAccuracy(test)
        println("itN: " + num_of_updates + " * " + sampleN + "\t ACC:" + acc + "\t SVMModelN:" + model.count() + "\t Time:" + (zipTime + itTimeSum) + " ( " + zipTime + ", " + itTimeSum + " ) " + "\t takeSampleT: " + takeSampleTime + "\t ypT: " + ypTime + "\t computeTime:" + computeTime + "\t multiputTime: " + multiputTime + "\t checkpointTime: " + checkpointTime)
        println("\t zipSampleTime:" + zipSampleTime + "\t zipYpTime: " + zipYpTime + "\t zipComputeTime: " + zipComputeTime + "\t selectIndexTime: " + selectIndexTime)
        println("itN: " + num_of_updates + " ,zipALL, " + cancel1 + ",afterZipFilter," + cancel2)
      }
      testTime += System.currentTimeMillis() - testT
    }


    //keep the effective part of the model
    model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
    working_data.unpersist()

  }

  /** getting the number of support vectors of the trained model */
  def getNumSupportVectors(): Long = {
    model.count()
  }

  /** make a prediction on a single data point */
  def predict(data: LabeledPoint): Double = {
    s * (model.map { case (k, v) => v * k.label * kernel_func.evaluate(data.features, k.features) }.reduce((a, b) => a + b))
  }

  /** Evaluate the accuracy given the test set as a local array */
  def getAccuracy(data: Array[LabeledPoint]): Double = {
    val N_c = data.map(x => (predict(x) * x.label)).count(x => x > 0)
    val N = data.count(x => true)
    (N_c.toDouble / N)

  }

  def getAccuracy(data: RDD[LabeledPoint]): Double = {
    val arrModel = model.collect()
    val N_c = data.map(x => (s * (arrModel.map { case (k, v) => v * k.label * kernel_func.evaluate(x.features, k.features) }.reduce((a, b) => a + b)) * x.label)).filter(_ > 0).count()
    val N = data.count()
    (N_c.toDouble / N)

  }

}

object ZFIncreSvdSVMLayer {
  /**
    * 目前只支持ratio=100%,其它有问题
    *
    * @param args
    */
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)

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
    val perCheckN = args(12).toInt

    val itqitN = args(13).toInt
    val itqratioN = args(14).toInt //from 1 not 0
    val upBound = args(15).toInt
    val redisHost = args(16) //"172.18.11.97"


    val splitChar = ","
    println("kernel, " + kernel + "\tstep, " + step + "\tgamma, " + gamma ++ "\t packN, " + packN + "\t upBound, " + upBound)

    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features).toSparse.asInstanceOf[SparseVector]) // must be sparse
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }

    val zipTime = System.currentTimeMillis()
    val indexedData: IndexedRDD[Long, (LabeledPoint, Double)] = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })
    val oHash = new ZFHashLayer(itqitN, itqratioN, upBound, isSparse, 2, redisHost, sc)
    //* zipData(i) = [ (lp,nullArr),(lp,indexArr),...,(lp,indexArr) ] the first element is the 0_th zip, the follows are 1_th
    val zipData: Array[Array[(LabeledPoint, Array[Long])]] = indexedData.mapPartitions(oHash.zfHashMapIndexedRDD2).collect()
    //    val cancel0 = zipData.flatMap(arr => arr.slice(1, arr.size).map(_._2.size)).sortWith(_ < _)
    //    println("zip1Size, " + cancel0.slice(0, 10).mkString(",") + ",,," + cancel0.slice(cancel0.size - 10, cancel0.size).mkString(","))
    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))

    val tempP = new ArrayBuffer[String]()
    for (r <- ratioL) {

      val s = checkPointDir splitAt (checkPointDir lastIndexOf ("/"))
      //      val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
      //      val tPath = new org.apache.hadoop.fs.Path(s._2)
      //      if (fs.exists(tPath)) {
      //        fs.delete(tPath, true) // isRecusrive= true
      //      }
      //      fs.mkdirs(new Path(s._2))
      //      fs.close()


      val ratio = r / 100.0
      val splits = indexedData.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) IndexedRDD(indexedData.sample(false, ratio)) else IndexedRDD(splits(0).sample(false, ratio))
      val test = if (testPath.size > 3) {
        if (!isSparse) {
          sc.textFile(testPath).map(line => {
            val vs = line.split(splitChar).map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1).map(_._2._1)
      println("partN:\t" + train.getNumPartitions)
      val t1 = System.currentTimeMillis
      val svm = new ZFIncreSvdSVMLayer(train, zipData, test, step, kernel, gamma, checkPointDir) //gamma = 0.1
      svm.train(itN, packN, perCheckN)
      val t2 = System.currentTimeMillis
      val runtime = (t2 - t1) - svm.testTime

      println("model count:\t" + svm.model.count())
      println("Ratio\titN\tpackN\tACC\tT")

      val acc = svm.getAccuracy(test)
      var ss = ratio + "\t" + itN + "*" + packN + "\t" + acc + "\t" + runtime + "\n"

      System.out.println(ss)
      tempP += ss
      data.unpersist()
      train.unpersist()

    }
    println("Ratio\titN\tpackN\tACC\tT")
    println(tempP.mkString("\n"))

  }
}
