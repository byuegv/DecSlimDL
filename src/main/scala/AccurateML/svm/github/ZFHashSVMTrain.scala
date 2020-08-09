package AccurateML.svm.github

import AccurateML.blas.ZFBLAS
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.Array._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by zhangfan on 17/8/21.
  */

class ZFHashSVMTrain(indexedData: IndexedRDD[Long, (LabeledPoint, Double)],
                     azipData: Array[(Long, Tuple3[LabeledPoint, Double, Array[Long]])],
                     test_data: RDD[LabeledPoint],
                     lambda_s: Double,
                     kernel: String,
                     gamma: Double,
                     checkpoint_dir: String) extends java.io.Serializable {

  /** Initialization */
  var lambda = lambda_s
  //Currently only rbf kernel is implemented
  var kernel_func = KernelSVM.zfSvmKerkel(gamma, kernel)
  //  var kernel_func_zf = new ZFRbfKernelFunc(gamma,training_data.first().features.size/10)
  var model = indexedData.map(x => (x._2._1, 0D))
  val test = test_data
  var s = 1D
  var zipData = azipData
  val sc = indexedData.sparkContext
  //  sc.setCheckpointDir(checkpoint_dir)

  /** method train, based on P-Pack algorithm **/
  def train(num_iter: Int, pack_size: Int = 1, perCheckN: Int = 10) {
    //Free the current model from memory
    model.unpersist()
    //Initialization
    var working_data: IndexedRDD[Long, (LabeledPoint, Double)] = indexedData
    working_data.persist(StorageLevel.MEMORY_AND_DISK)
    var norm = 0D
    var alpha = 0D
    var t = 1
    var i = 0
    var j = 0

    val pair_idx = sc.parallelize(range(0, pack_size).flatMap(x => (range(x, pack_size).map(y => (x, y)))))
    val broad_kernel_func = sc.broadcast(kernel_func)

    // Training the model with pack updating
    var itTimeSum = 0L
    var ypTime = 0L
    var computeTime = 0L
    var multiputTime = 0L
    var mutiget1Time = 0L
    var mutiget2Time = 0L
    var zipFilterTime = 0L


    for (num_of_updates <- 1 to num_iter) {
      val itStartTime = System.currentTimeMillis()
      val sample: Array[(Long, (LabeledPoint, Double))] = {
        val zipIndex = new ArrayBuffer[Long]() //zipData.map(_._2._3).takeSample(false, pack_size).flatMap(arr => arr).slice(0, pack_size)
        val zipSet = new mutable.HashSet[Int]()
        while (zipIndex.size < pack_size) {
          val r = Random.nextInt(zipData.size)
          if (!zipSet(r)) {
            zipIndex ++= zipData(r)._2._3
            zipSet.add(r)
          }

        }
        val temp1 = System.currentTimeMillis()
        val temp = working_data.multiget(zipIndex.slice(0, pack_size).toArray).toArray
        mutiget1Time += System.currentTimeMillis() - temp1
        temp
      }
      //      zipTime += (System.currentTimeMillis() - itStartTime)
      //            var sample = working_data.takeSample(true, pack_size)
      val broad_sample = sc.broadcast(sample)

      val yp = if (t == 1) {
        Array.fill(pack_size)(0.0)
      } else {
        val tempf = System.currentTimeMillis()
        val selectDataIndex = zipData.flatMap { case (centerIndex, (lp, beta, ab)) => if (beta != 0) ab else new Array[Long](0) }
        zipFilterTime += System.currentTimeMillis() - tempf
        println(t / pack_size + "\tselect Points N: " + selectDataIndex.size)
        val temp2 = System.currentTimeMillis()
        val selectData = working_data.multiget(selectDataIndex) //may out of memory
        mutiget2Time += System.currentTimeMillis() - temp2
        broad_sample.value.map(x => (selectData.map { case (k, v) => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features)) }.reduce((a, b) => a + b)))
      }
      ypTime += (System.currentTimeMillis() - itStartTime)
      //      var yp = broad_sample.value.map(x => (working_data.map { case (k, v) => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features)) }.reduce((a, b) => a + b)))
      val tempComputeTime = System.currentTimeMillis()
      val y = sample.map(x => x._2._1.label)
      var local_set = Map[Long, (LabeledPoint, Double)]()
      val inner_prod = pair_idx.map(x => (x, broad_kernel_func.value.evaluate(sample(x._1)._2._1.features, sample(x._2)._2._1.features))).collectAsMap()
      //      // Compute sub gradients
      for (i <- 0 until pack_size) {
        t = t + 1
        s = (1 - 1D / (t)) * s
        for (j <- (i + 1) until (pack_size)) {
          yp(j) = (1 - 1D / (t)) * yp(j)
        }
        if (y(i) * yp(i) < 1) {
          norm = norm + (2 * y(i)) / (lambda * t) * yp(i) + math.pow((y(i) / (lambda * t)), 2) * inner_prod((i, i))
          alpha = sample(i)._2._2
          local_set = local_set + (sample(i)._1 ->(sample(i)._2._1, alpha + (1 / (lambda * t * s))))

          for (j <- (i + 1) to (pack_size - 1)) {
            yp(j) = yp(j) + y(j) / (lambda * t) * inner_prod((i, j))
          }

          if (norm > (1 / lambda)) {
            s = s * (1 / math.sqrt(lambda * norm))
            norm = (1 / lambda)
            for (j <- (i + 1) to (pack_size - 1)) {
              yp(j) = yp(j) / math.sqrt(lambda * norm)
            }
          }
        }
      }
      computeTime += System.currentTimeMillis() - tempComputeTime

      val tempMultiputTime = System.currentTimeMillis()
      val to_forget = working_data
      working_data = working_data.multiput(local_set)
      multiputTime += System.currentTimeMillis() - tempMultiputTime

      zipData = zipData.map { case (centerId, (lp, beta, ab)) => {
        var sum = beta
        for ((k, v) <- local_set) {
          if (ab.contains(k)) {
            sum += v._2
          }
        }
        Tuple2(centerId, Tuple3(lp, sum, ab))
      }
      }
      //      zipTime += (System.currentTimeMillis() - tempZipTime)
      to_forget.unpersist()
      //checkpoint
      if (num_of_updates % perCheckN == 0) {
        println(num_of_updates + "\tcheckpoint" + "\tworking-data partN:\t" + working_data.getNumPartitions)
        //        working_data.checkpoint()
        //        zipData.checkpoint()
        val s = checkpoint_dir splitAt (checkpoint_dir lastIndexOf ("/"))
        val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
        val lastN = num_of_updates - perCheckN
        working_data.map(t => t).saveAsObjectFile(checkpoint_dir + "/working_data/" + num_of_updates)
        if (lastN > 0) {
          fs.delete(new Path(checkpoint_dir + "/working_data/" + lastN), true)
        }
        working_data = IndexedRDD(sc.objectFile[(Long, (LabeledPoint, Double))](checkpoint_dir + "/working_data/" + num_of_updates))
        fs.close()
      }


      itTimeSum += System.currentTimeMillis() - itStartTime
      if (num_of_updates % 500 == 0) {
        model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
        val acc = getAccuracy(test)
        println("itN:" + num_of_updates + " * " + pack_size + "\tACC: " + acc + "\tSVMModelN: " + model.count() + "\tTime: " + itTimeSum + "\typT: " + ypTime + "\t{ zipFilterT: " + zipFilterTime + "\tget1T: " + mutiget1Time + "\t get2T:" + mutiget2Time + "}\tcomputeT:" + computeTime + "\t multiputT: " + multiputTime)
      }
    }


    //keep the effective part of the model
    model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
    working_data.unpersist()
    //    println(" case (k, v) => (  v._2 + x._2._2 ) }.reduce((a, b) => a + b)))")


  }

  /** getting the number of support vectors of the trained model */
  def getNumSupportVectors(): Long = {
    model.count()
  }

  /** make a prediction on a single data point */
  def predict(data: LabeledPoint): Double = {

    //    model.map { case (k, v) =>
    //      val tempv = v
    //      val tempk = k.label
    //      val tempdot = kernel_func.evaluate(data.features, k.features)
    //      val ans = tempv * tempk * tempdot
    //      v * k.label * kernel_func.evaluate(data.features, k.features)
    //    }
    //      .reduce((a, b) => a + b)

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

  /** reset the regularization lambda */
  def setLambda(new_lambda: Double) {
    lambda = new_lambda
  }

}

object ZFHashSVMTrain {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val logFile = "README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)


    // /Users/zhangfan/Documents/data/zflittle /Users/zhangfan/Documents/data/zflittle 1 3 100 false 4 1 /Users/zhangfan/Documents/cancel 2 2
    val dataPath = args(0)
    val testPath = args(1)
    val packN = args(2).toInt
    val itN = args(3).toInt
    val ratioL = args(4).split(",").map(_.toDouble)
    val isSparse = args(5).toBoolean
    val numFeature = args(6).toInt
    val minPartN = args(7).toInt
    val checkPointDir = args(8)

    val kmeansK = args(9).toInt
    val kmeansItN = args(10).toInt



    val splitChar = ","

    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features))
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }



    //kmeans
    val zipTime = System.currentTimeMillis()
    val kmeansData = data.map(lp => lp.features)
    val clusters = KMeans.train(kmeansData, kmeansK, kmeansItN)
    // IndexedRDD[Long, Tuple2[LabeledPoint, Double]]
    var indexedData = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })
    /** zipData [centerId,(lp,0.0b,Array(pointIndex))] */
    val zipData: RDD[(Long, Tuple3[LabeledPoint, Double, Array[Long]])] = indexedData
      .map { case (id, lpb) => Tuple2(clusters.predict(lpb._1.features).toLong, Tuple2(id, lpb._1)) }
      .aggregateByKey(Tuple3(Array(0.0), Vectors.zeros(numFeature), new ArrayBuffer[Long]()))(
        seqOp = (u, v) => {
          //v (dataId,b)
          u._1(0) += v._2.label
          ZFBLAS.axpy(1.0, v._2.features, u._2)
          u._3 += v._1
          u
        },
        combOp = (u1, u2) => {
          u1._1(0) += u2._1.last
          ZFBLAS.axpy(1.0, u2._2, u1._2)
          u1._3 ++= u2._3
          u1
        }
      )
      .map(t4 => {
        val n = t4._2._3.size
        val vec = t4._2._2
        ZFBLAS.scal(1.0 / n, vec)
        val lp = new LabeledPoint(t4._2._1.last / n, vec)
        Tuple2(t4._1, Tuple3(lp, 0.0D, t4._2._3.toArray))
      })
    val cancel = zipData.collect()
    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))
    println("zipN: " + cancel.size + "\t[,\t" + cancel.slice(0, 10).map(_._2._3.size).mkString(","))

    val tempP = new ArrayBuffer[String]()
    for (r <- ratioL) {
      val ratio = r / 100.0
      val splits = indexedData.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) IndexedRDD(indexedData.sample(false, ratio)) else IndexedRDD(splits(0).sample(false, ratio))
      val testData: RDD[LabeledPoint] = if (testPath.size > 3) {
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
      val m = train.count()
      //      var num_iter = 0
      //      num_iter = (2 * m).toInt
      val t1 = System.currentTimeMillis
      val svm = new ZFHashSVMTrain(train, zipData.collect(), testData, 1.0 / m, "rbf", 0.1, checkPointDir)
      svm.train(itN * packN, packN)
      val t2 = System.currentTimeMillis
      val runtime = (t2 - t1)

      println("model count:\t" + svm.model.count())
      println("Ratio\tDataN\titN\tpackN\tACC\tT")

      val acc = svm.getAccuracy(testData)
      var ss = ratio + "\t" + m + "\t" + (itN * packN) + "\t" + packN + "\t" + acc + "\t" + runtime + "\n"
      System.out.println(ss)
      tempP += ss
      data.unpersist()
      train.unpersist()

    }
    println("Ratio\tDataN\titN\tpackN\tACC\tT")
    println(tempP.mkString("\n"))

  }
}
