package AccurateML.svm.github

/*
 * Kernel SVM: the class for kernelized SVM on Spark
 * Using SGD
 * Usage example:
    //data = some rdd of LabeledPoint
    //setup amodel by regietering training data, specifying lambda,
    //specifying kernel and kernel parameters
    val model = new github.KernelSVM(data_train, 1.0, "rbf", 1.0)
    //train the model by specifying # of iterations and packing size
    model.train(1000,10)
 */

import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.Array._
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer


class KernelSVM(training_data: RDD[LabeledPoint], test_data: RDD[LabeledPoint], lambda_s: Double, kernel: String = "rbf", gamma: Double = 1D, checkpoint_dir: String = "/home/ubuntu/") extends java.io.Serializable {
  /** Initialization */
  val lambda = lambda_s
  //Currently only rbf kernel is implemented
  var kernel_func = KernelSVM.zfSvmKerkel(gamma, kernel)
  //  var kernel_func_zf = new ZFRbfKernelFunc(gamma,training_data.first().features.size/10)
  var model = training_data.map(x => (x, 0D))
  var data = training_data
  val test = test_data
  var s = 1D
  var testTime = 0L
  //  sc.setCheckpointDir(checkpoint_dir)

  /** method train, based on P-Pack algorithm **/
  def train(num_iter: Int, pack_size: Int = 1, perCheckN: Int = 10) {
    //Free the current model from memory
    model.unpersist()
    //Initialization
    var working_data: IndexedRDD[Long, (LabeledPoint, Double)] = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })
    working_data.persist(StorageLevel.MEMORY_AND_DISK)
    var norm = 0D
    var alpha = 0D
    var t = 1
    var i = 0
    var j = 0

    val pair_idx = working_data.sparkContext.parallelize(range(0, pack_size).flatMap(x => (range(x, pack_size).map(y => (x, y)))))
    val broad_kernel_func = working_data.sparkContext.broadcast(kernel_func)
    //    val broad_kernel_func_zf = data.sparkContext.broadcast(kernel_func_zf)

    // Training the model with pack updating
    var itTimeSum = 0L
    var ypTime = 0L
    var takeSampleTime = 0L
    var computeTime = 0L
    var multiputTime = 0L
    var checkpointTime = 0L

    val ypPartTime = working_data.sparkContext.collectionAccumulator[Long]

    for (num_of_updates <- 1 to num_iter) {

      val itStartTime = System.currentTimeMillis()
      val sample = working_data.takeSample(true, pack_size)
      takeSampleTime += (System.currentTimeMillis() - itStartTime)

      /** how to caculate yp filter or not(original) */
      val ypT = System.currentTimeMillis()
      val yp = {
        val broad_sample = working_data.sparkContext.broadcast(sample)
        working_data.mapPartitions(pit => {
          val samples = broad_sample.value.map(_._2)
          val ans = Array.fill(samples.size)(0.0)

          while (pit.hasNext) {
            val (k, v) = pit.next()
            val aa = samples.map(x => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._1.features)))
            for (i <- 0 until ans.size) ans(i) += aa(i)
          }
          Array(ans).iterator
        }).reduce((aa, bb) => {
          for (i <- 0 until aa.size)
            aa(i) += bb(i)
          aa
        })
      }
      //      val yp = broad_sample.value.map(x => (working_data.map { case (k, v) => if(v._2==0) 0.0 else (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features)) }.reduce((a, b) => a + b)))
      //      val yp = broad_sample.value.map(x => (working_data.map { case (k, v) => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features)) }.reduce((a, b) => a + b)))
      ypTime += System.currentTimeMillis() - ypT

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
        val tempTime = System.currentTimeMillis()
        if (y(i) * yp(i) < 1) {
          norm = norm + (2 * y(i)) / (lambda * t) * yp(i) + math.pow((y(i) / (lambda * t)), 2) * inner_prod((i, i))
          alpha = sample(i)._2._2
          local_set = local_set + (sample(i)._1 ->(sample(i)._2._1, alpha + (1 / (lambda * t * s))))
          //          println("beta: "+(alpha + (1 / (lambda * t * s)))+"\talpha: "+alpha +"\t lts: "+lambda * t * s+"\t"+lambda+","+t+","+s )
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
//      val to_forget = working_data
      //      val cancelN = (sample.size * 0.5).toInt
      //      val cancelMap: Map[Long, (LabeledPoint, Double)] = local_set.toArray.sortBy(t => math.abs(t._2._2)).slice(0, cancelN).toMap
      //      working_data = working_data.multiput(cancelMap).persist(StorageLevel.MEMORY_AND_DISK)
      working_data = working_data.multiput(local_set).persist(StorageLevel.MEMORY_AND_DISK)
      //      println("localset "+local_set.values.map(_._2).mkString(","))
      working_data.count() //cancel
//      to_forget.unpersist()
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
        working_data.sparkContext.setCheckpointDir(checkpoint_dir + "/" + num_of_updates)
        working_data.checkpoint() //会增大hdfs磁盘使用量,超过一定阈值报错 safe mode cant change,所以要手动删除文件
        working_data.count()
        val s = checkpoint_dir splitAt (checkpoint_dir lastIndexOf ("/"))
        val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), working_data.sparkContext.hadoopConfiguration)
        val lastN = num_of_updates - perCheckN
        if (lastN > 0) {
          fs.delete(new Path(checkpoint_dir + "/" + lastN), true)
        }
      }
      //get acc
      checkpointTime += System.currentTimeMillis() - checkpointT
      itTimeSum += System.currentTimeMillis() - itStartTime
      val testT = System.currentTimeMillis()
      println(" ypT: " + ypTime + "\t ypPartT: " + ypPartTime.value.sum)
      if (num_of_updates % 10 == 0) {
        model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
        val acc = getAccuracy(test)
        println("itN: " + num_of_updates + " * " + sample.size + "\t ACC:" + acc + "\t SVMModelN:" + model.count() + "\t Time:" + itTimeSum + "\t takeSampleT: " + takeSampleTime + "\t ypT: " + ypTime + "\t ypPartT: " + ypPartTime.value.sum + "\t computeTime:" + computeTime + "\t multiputTime: " + multiputTime + "\t checkpointTime: " + checkpointTime)
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

  /** register a new training data */
  def registerData(new_training_data: RDD[LabeledPoint]) {
    data = new_training_data
  }

  /** reset the regularization lambda */
  def setLambda(new_lambda: Double) {
    //    lambda = new_lambda
  }

}


object KernelSVM {
  //old name TestKernelSVM
  def zfSvmKerkel(gamma: Double, kernel: String): Kernels = {
    if (kernel.contains("rbf")) {
      new RbfKernelFunc(gamma)
    } else if (kernel.contains("poly")) {
      new PolynomialKernelFunc(gamma)
    } else if (kernel.contains("linear")) {
      new LinearKernelFunc(gamma)
    } else if (kernel.contains("tanh")) {
      new TanhKernelFunc(gamma)
    } else {
      System.err.println("Kernel name error!\t" + kernel)
      new RbfKernelFunc(gamma)
    }
  }

  def main(args: Array[String]) {


    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val logFile = "README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")


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
    val checkPointDir = args(11)
    val perCheckN = args(12).toInt

    val splitChar = ","
    println("kernel, " + kernel + "\tstep, " + step + "\tgamma, " + gamma ++ "\t packN, " + packN)


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

      val data = if (!isSparse) {
        sc.textFile(dataPath, minPartN).map(line => {
          val vs = line.split(splitChar).map(_.toDouble)
          val features = vs.slice(0, vs.size - 1)
          LabeledPoint(vs.last, Vectors.dense(features))
        })
      } else {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
      }
      val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) data.sample(false, ratio) else splits(0).sample(false, ratio)
      //      val m = train.count()
      println("partN:\t" + train.getNumPartitions)
      val test = if (testPath.size > 3) {
        if (!isSparse) {
          sc.textFile(testPath, minPartN).map(line => {
            val vs = line.split(splitChar).map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1)


      val t1 = System.currentTimeMillis
      val svm = new KernelSVM(train, test, step, kernel, gamma, checkPointDir) //gamma=0.1
      //      val svm = new KernelSVM(train, 1.0 / m, kernel, gamma, checkPointDir) //gamma=0.1
      svm.train(itN, packN, perCheckN)
      val t2 = System.currentTimeMillis
      val trainT = (t2 - t1) - svm.testTime
      //      val svmModelPath = s._1 + "/svmModel/" + dataPath.split("/").last + "-" + itN + "-" + packN + "-" + kernel + "-" + gamma + "-" + Random.nextInt()
      val cancel = svm.model.map(_._2).collect().sortWith(_ < _)
      println(svm.getClass.getName + " model count:\t " + svm.model.count() + "\t[," + cancel.slice(0, 10).mkString(",") + ",,," + cancel.slice(cancel.size - 10, cancel.size).mkString(","))
      val acc = svm.getAccuracy(test)

      println("Ratio \t ACC \t itN \t packN \t T")
      var ss = ratio + "\t" + acc + "\t" + itN + "\t" + packN + "\t" + trainT + "\t" + testPath + "\n"
      System.out.println(ss)
      tempP += ss
      data.unpersist()
      train.unpersist()
      val fs2 = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
      fs2.delete(new org.apache.hadoop.fs.Path(s._2), true) // isRecusrive= true
      fs2.close()
    }
    //    println(tempP.mkString("\n"))
  }
}

