//package AccurateML.svm.github
//
//import java.io.PrintWriter
//
//import AccurateML.blas.ZFBLAS
//import AccurateML.svm.github.zfobjectSVM
//import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
//import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
//import org.apache.hadoop.fs.Path
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.mllib.clustering.KMeans
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.util.MLUtils
//import org.apache.spark.rdd.RDD
//import org.apache.spark.{SparkConf, SparkContext}
//
//import scala.Array._
//import scala.collection.mutable
//import scala.collection.mutable.ArrayBuffer
//import scala.util.Random
//import java.io.File
//
///**
//  * Created by zhangfan on 17/8/21.
//  */
//
//class ZFKmeansSVM_WR(indexedData: IndexedRDD[Long, (LabeledPoint, Double)],
//                     azipData: Array[(Long, Tuple3[LabeledPoint, Double, Array[Long]])],
//                     test_data: RDD[LabeledPoint],
//                     lambda_s: Double,
//                     kernel: String,
//                     gamma: Double,
//                     checkpoint_dir: String) extends java.io.Serializable {
//
//  /** Initialization */
//  var lambda = lambda_s
//  //Currently only rbf kernel is implemented
//  var kernel_func = if (kernel.contains("rbf")) {
//    new RbfKernelFunc(gamma)
//  } else {
//    new PolynomialKernelFunc(gamma)
//  }
//  //  var kernel_func_zf = new ZFRbfKernelFunc(gamma,training_data.first().features.size/10)
//  var model = indexedData.map(x => (x._2._1, 0D))  //RDD[LabeledPoint,Double]³õÊ¼»¯Îª0
//  val test = test_data //RDD[LabeledPoint]
//  var s = 1D
//  var zipData = azipData
//  val sc = indexedData.sparkContext
//  //  sc.setCheckpointDir(checkpoint_dir)
//
//  /** method train, based on P-Pack algorithm **/
//  def train(num_iter: Int, pack_size: Int = 1, perCheckN: Int = 10) {
//    //Free the current model from memory
//    model.unpersist() //½«RDD±ê¼ÇÎª·Ç³Ö¾ÃÐÔ£¬²¢´ÓÄÚ´æºÍ´ÅÅÌÖÐÉ¾³ýËùÓÐµÄ¿é
//    //Initialization
//    var working_data: IndexedRDD[Long, (LabeledPoint, Double)] = indexedData
//    var norm = 0D
//    var alpha = 0D
//    var t = 1
//    var i = 0
//    var j = 0
//    /**
//      * ÓÉÓÚpair_idx(i,j)ºÍpair(j,i)×îºóÇóµÄ½á¹ûÊÇÒ»ÑùµÄËùÒÔÖ»±£´æÆäÖÐÒ»¸ö
//      * (0, 0)(0, 1)(0, 2)(0, 3)(0, 4)(0, 5)(0, 6)(0, 7)(0, 8)(0, 9)
//      * (1, 1)(1, 2)(1, 3)(1, 4)(1, 5)(1, 6)(1, 7)(1, 8)(1, 9)
//      * (2, 2)(2, 3)(2, 4)(2, 5)(2, 6)(2, 7)(2, 8)(2, 9)
//      * (3, 3)(3, 4)(3, 5)(3, 6)(3, 7)(3, 8)(3, 9)
//      * (4, 4)(4, 5)(4, 6)(4, 7)(4, 8)(4, 9)
//      * (5, 5)(5, 6)(5, 7)(5, 8)(5, 9)
//      * (6, 6)(6, 7)(6, 8)(6, 9)
//      * (7, 7)(7, 8)(7, 9)
//      * (8, 8)(8, 9)
//      * (9, 9)
//      */
//    val pair_idx = sc.parallelize(range(0, pack_size).flatMap(x => range(x, pack_size).map(y => (x, y))))
//    val broad_kernel_func = sc.broadcast(kernel_func)
//    // Training the model with pack updating
//    var itTimeSum = 0L
//    var ypTime = 0L
//    var computeTime = 0L
//    var multiputTime = 0L
//    var mutiget1Time = 0L
//    var mutiget2Time = 0L
//    var zipFilterTime = 0L
//    var WR:Long = 0L
//    var mutigetWR:Long = 0L
//    var mutigetWR1:Long = 0L
//    var WRSelectDataSTime = 0L
//    var WRSelectDataETime = 0L
//    val writer = new PrintWriter(new File("/home/wangrui/important/SVMtest.txt" ))
//    var WRSelectData: Map[Long, (LabeledPoint, Double)] = Map()
//    var flags: Array[Int] = Array.fill(zipData.length)(0)
//    //    println("zipData.length:" + zipData.length)
//    var WRselectDataIndex = Array.fill(zipData.length)(0L)
//    var b = 0
//    var num = 0
//    for (num_of_updates <- 1 to num_iter) {
//      num = num + 1
//      println("iter:" + num)
//      writer.write("iter:" + num + "\n")
//      val itStartTime = System.currentTimeMillis()
//      val sample: Array[(Long, (LabeledPoint, Double))] = {
//        val zipIndex = new ArrayBuffer[Long]() //zipData.map(_._2._3).takeSample(false, pack_size).flatMap(arr => arr).slice(0, pack_size)
//        val zipSet = new mutable.HashSet[Int]()
//        while (zipIndex.size < pack_size) {
//          /**
//            * rµÄÖµÖ¸µÄÊÇ´Ó0µ½zipData.length³¤¶ÈÖÐÈÎÒâÒ»¸öÖµ£¬±ÈÈçËµÈç¹ûzipData.length³¤¶ÈÊÇ10£¬Ôòr¿ÉÄÜÊÇ9£¬¿ÉÄÜÊÇ6
//            */
//          val r = Random.nextInt(zipData.length)
//          val len = zipData.length
//          //        println("r:" + r + " lenL:" + len)
//          if (!zipSet(r)) {
//            zipIndex ++= zipData(r)._2._3
//            zipSet.add(r)
//          }
//        }
//        //        val temp1 = System.currentTimeMillis()
//        //        val temp = working_data.multiget(zipIndex.slice(0, pack_size).toArray).toArray
//        ////wr test
//        //        temp.foreach{t =>
//        //          println("temp:"+t)
//        //          println("temp_index" + t._2._1.features)
//        //          println("temp_index1" + t._2._1.features(1))
//        //          println("long:"+t._1 + " LabelPoint:" +t._2._1.label)}
//        //          mutiget1Time += System.currentTimeMillis() - temp1
//        //          println("mutiget1Time:" +mutiget1Time)
//        //        println("working_data.getNumPartitions" + working_data.getNumPartitions)
//        //        working_data.saveAsTextFile("/home/wangrui/work_data1.txt")
//        mutigetWR = System.currentTimeMillis()
//        val tempX:Array[(Long, (LabeledPoint, Double))] = new Array[(Long, (LabeledPoint, Double))](pack_size)
//        for (i: Int <- 0 until pack_size) {
//          tempX(i) = (zipIndex(i), working_data.get(zipIndex(i)).toArray.last)
//        }
//        //        tempX.foreach{t =>
//        //          println("tempWR:"+ t)
//        //          println("longWR:"+t._1 + " LabelPointWR:" +t._2._1.label)}
//        //        val work:
//        WR += System.currentTimeMillis() - mutigetWR
//        //        println("mutigetWR:" + WR)
//        writer.write("mutigetWR:" + WR + "\n")
//        //        println("tempX.equals(temp):" + temp(0).equals(tempX(0)))
//        //        writer.write("tempX.equals(temp):" + temp(0).equals(tempX(0)) + "\n")
//        tempX
//      }
//      //     zipTime += (System.currentTimeMillis() - itStartTime)
//      //            var sample = working_data.takeSample(true, pack_size)
//      val broad_sample = sc.broadcast(sample)
//      val yp = if (t == 1) {
//        Array.fill(pack_size)(0.0)
//      } else {
//        val tempf = System.currentTimeMillis()
//        //        val selectDataIndex =  zipData.flatMap { case (centerIndex, (lp, beta, ab)) => if (beta != 0)
//        //          {
//        //          //  print(centerIndex + " ")
//        //            ab
//        //          } else new Array[Long](0) }
//        //        println()
//        ////        zipFilterTime += System.currentTimeMillis() - tempf
//        //        println("select Points N: " + selectDataIndex.length)
//        ////        val temp2 = System.currentTimeMillis()
//        //        val selectData = working_data.multiget(selectDataIndex) //may out of memory
//        //        selectData.foreach{t =>
//        //                    count = count + 1
//        //                   if(count <= 10){
//        //                     println("temp:"+ t)
//        //                     println("long:"+t._1 + " LabelPoint:" +t._2._1.label)
//        //                   }
//        //        }
//        //        mutiget2Time += System.currentTimeMillis() - temp2
//        //       println("selectData mutiget2Time:"+ mutiget2Time)
//        //   val tempf = System.currentTimeMillis()
//        //        val flag = Array.fill(zipData.length)(0)
//        //        val WRselectDataIndex1 = zipData.flatMap { case (centerIndex, (lp, beta, ab)) => {
//        //       if (beta != 0 && (flags(centerIndex.toInt) == 0)) {
//        //            flags(centerIndex.toInt) = 1
//        //            ab
//        //          } else {
//        //            new Array[Long](0)
//        //       }
//        //          }
//        //        }
//        val WRselectDataIndex1 = zipData.flatMap { case (centerIndex, (lp, beta, ab)) => {
//          if (beta != 0) {
//            ab
//          } else {
//            new Array[Long](0)
//          }
//        }
//        }
//        //        var c = 0
//        //        flags.foreach{flag => if(flag == 1)
//        //          print(c + " ")
//        //          c = c + 1}
//        //        println()
//        val WRselectDataIndex2 = if(WRselectDataIndex1.diff(WRselectDataIndex)!=null){
//          WRselectDataIndex1.diff(WRselectDataIndex)
//        }else{
//          new Array[Long](0)
//        }
//        //        val WRselectDataIndex3 = if(WRselectDataIndex != null) {
//        //          WRselectDataIndex.filter(X => WRselectDataIndex2.contains(X))
//        //        }else{
//        //          new Array[Long](0)
//        //        }
//        //        println("WRselectDataIndex3:" + (WRselectDataIndex3.length == 0))
//        zipFilterTime += System.currentTimeMillis() - tempf
//        WRSelectDataSTime = System.currentTimeMillis()
//        WRselectDataIndex = WRselectDataIndex1
//        //        for(i <- 0 until WRselectDataIndex3.length){
//        //            var j = WRselectDataIndex3(i)
//        //             println("j" + j)
//        //             WRSelectData.updated(j, working_data.get(j))
//        //        }
//        WRSelectData = WRSelectData ++ working_data.multiget(WRselectDataIndex2)
//        WRSelectDataETime += System.currentTimeMillis() - WRSelectDataSTime
//        println("WRSelectDataETime:" + WRSelectDataETime)
//        //        writer.write("WRSelectDataETime:" + WRSelectDataETime + "\n")
//        //        println(" WRSelectDataIndex size equal selectDataIndex size:" +(WRselectDataIndex.size == selectDataIndex.size))
//        //        writer.write(" WRSelectDataIndex size equal selectDataIndex size:" +(WRselectDataIndex.size == selectDataIndex.size) + "\n")
//        //        println(" WRSelectData size:" + WRSelectData.size)
//
//        //        if(!WRSelectData.equals(selectData) && b < 5){
//        //          b = b + 1
//        //          println("----------------------")
//        //          writer.write("----------------------\n")
//        //          WRSelectData.foreach(point =>
//        //            if(point._2._2 != 0)
//        //            writer.write("long:" + point._1 + " double:" + point._2._2 + "       "))
//        //          println("*********")
//        //          writer.write("\n*********\n")
//        //          selectData.foreach(point =>
//        //            if(point._2._2 != 0)
//        //              writer.write("long:" + point._1 + " double:" + point._2._2 + "       "))
//        //          println("----------------------")
//        //          writer.write("\n----------------------\n")
//        //        }
//        //        println("WRSelectData.equals(selectData):" + WRSelectData.equals(selectData))
//        //        val lenWR = selectDataIndex.length
//        //        println("lenR" + lenWR)
//        //        var time1 = System.currentTimeMillis()
//        //        var arrayWR:Array[(Long, (LabeledPoint, Double))] = new Array[(Long, (LabeledPoint, Double))](lenWR)
//        //        for (i: Int <- 0 until lenWR) {
//        //          arrayWR(i) = (selectDataIndex(i), working_data.get(selectDataIndex(i)).toArray.last)
//        //          println("--------------------------------------------")
//        //        }
//        //        arrayWR.foreach{t =>
//        //          println("tempWR:"+ t)
//        //          println("longWR:"+t._1 + " LabelPointWR:" +t._2._1.label)}
//        //        mutigetWR1 += System.currentTimeMillis() - time1
//        //        println("mutigetWR1" + mutigetWR1)
//
//        broad_sample.value.map(x => WRSelectData.map { case (k, v) => v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features) }.reduce((a, b) => a + b))
//      }
//      ypTime += (System.currentTimeMillis() - itStartTime)
//      //      println("ypTime is" + ypTime)
//      writer.write("ypTime is" + ypTime + "\n")
//      //      var yp = broad_sample.value.map(x => (working_data.map { case (k, v) => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features)) }.reduce((a, b) => a + b)))
//      val tempComputeTime = System.currentTimeMillis()
//      val y = sample.map(x => x._2._1.label)
//      var local_set = Map[Long, (LabeledPoint, Double)]()
//      val inner_prod = pair_idx.map(x => (x, broad_kernel_func.value.evaluate(sample(x._1)._2._1.features, sample(x._2)._2._1.features))).collectAsMap()
//      // Compute sub gradients
//      for (i <- 0 until pack_size) {
//        t = t + 1
//        s = (1 - 1D / t) * s
//        for (j <- (i + 1) until pack_size) {
//          yp(j) = (1 - 1D / t) * yp(j)
//        }
//        if (y(i) * yp(i) < 1) {
//          norm = norm + (2 * y(i)) / (lambda * t) * yp(i) + math.pow(y(i) / (lambda * t), 2) * inner_prod((i, i))
//          alpha = sample(i)._2._2
//          local_set = local_set + (sample(i)._1 ->(sample(i)._2._1, alpha + (1 / (lambda * t * s))))
//
//          for (j <- (i + 1)until pack_size) {
//            yp(j) = yp(j) + y(j) / (lambda * t) * inner_prod((i, j))
//          }
//
//          if (norm > (1 / lambda)) {
//            s = s * (1 / math.sqrt(lambda * norm))
//            norm = 1 / lambda
//            for (j <- (i + 1) until pack_size) {
//              yp(j) = yp(j) / math.sqrt(lambda * norm)
//            }
//          }
//        }
//      }
//      computeTime += System.currentTimeMillis() - tempComputeTime
//      //      println("computeTime:" + computeTime)
//      writer.write("computeTime:" + computeTime + "\n")
//      val tempMultiputTime = System.currentTimeMillis()
//      val to_forget = working_data
//      working_data = working_data.multiput(local_set)/*¸üÐÂÒ»×éKeyµÄValue*/
//      WRSelectData.foreach(point => {
//        val b = local_set.get(point._1).toArray
//        val c = Array(point._2)
//        if(b != null && b.length > 0){
//          if(!b(0).equals(c(0))){
//            WRSelectData += (point._1 -> b(0))
//          }
//        }
//      })
//      println()
//      multiputTime += System.currentTimeMillis() - tempMultiputTime
//      zipData = zipData.map { case (centerId, (lp, beta, ab)) => {
//        var sum = beta
//        for ((k, v) <- local_set) {
//          if (ab.contains(k)) {
//            sum += v._2
//          }
//        }
//        Tuple2(centerId, Tuple3(lp, sum, ab))
//      }
//      }
//      //      zipTime += (System.currentTimeMillis() - tempZipTime)
//      to_forget.unpersist()
//      //checkpoint
//      if (num_of_updates % perCheckN == 0) {
//        println(num_of_updates + "\tcheckpoint" + "\tworking-data partN:\t" + working_data.getNumPartitions)
//        //        working_data.checkpoint()
//        //        zipData.checkpoint()
//        val s = checkpoint_dir splitAt (checkpoint_dir lastIndexOf "/")
//        val fs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s._1), sc.hadoopConfiguration)
//        val lastN = num_of_updates - perCheckN
//
//        working_data.map(t => t).saveAsObjectFile(checkpoint_dir + "/working_data/" + num_of_updates)
//        if (lastN > 0) {
//          fs.delete(new Path(checkpoint_dir + "/working_data/" + lastN), true)
//        }
//        working_data = IndexedRDD(sc.objectFile[(Long, (LabeledPoint, Double))](checkpoint_dir + "/working_data/" + num_of_updates))
//
//        fs.close()
//      }
//
//
//      itTimeSum += System.currentTimeMillis() - itStartTime
//      if (num_of_updates % 500 == 0) {
//        model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => (v > 0) }.cache()
//        val zfObjectSVM = new zfobjectSVM(kernel, s, gamma, model.collect())
//        val acc = zfObjectSVM.getAccuracy(test)
//        println("itN:" + num_of_updates + " * " + pack_size + "\tACC: " + acc + "\tSVMModelN: " + model.count() + "\tTime: " + itTimeSum+"\typT: " + ypTime +"\t{ zipFilterT: "+zipFilterTime+"\tget1T: " + WR + "\t get2T:" + WRSelectDataETime + "}\tcomputeT:" + computeTime + "\t multiputT: " + multiputTime)
//        writer.write("itN:" + num_of_updates + " * " + pack_size + "\tACC: " + acc + "\tSVMModelN: " + model.count() + "\tTime: " + itTimeSum+"\typT: " + ypTime +"\t{ zipFilterT: "+zipFilterTime+"\tget1T: " + WR + "\t get2T:" + WRSelectDataETime + "}\tcomputeT:" + computeTime + "\t multiputT: " + multiputTime +"\n")
//      }
//    }
//    writer.close()
//    //keep the effective part of the model
//    model = working_data.map { case (k, v) => (v._1, v._2) }.filter { case (k, v) => v > 0 }.cache()
//    working_data.unpersist()
//    //    println(" case (k, v) => (  v._2 + x._2._2 ) }.reduce((a, b) => a + b)))")
//
//
//  }
//
//  /** getting the number of support vectors of the trained model */
//  def getNumSupportVectors: Long = {
//    model.count()
//  }
//
//  /** make a prediction on a single data point */
//  def predict(data: LabeledPoint): Double = {
//
//    //    model.map { case (k, v) =>
//    //      val tempv = v
//    //      val tempk = k.label
//    //      val tempdot = kernel_func.evaluate(data.features, k.features)
//    //      val ans = tempv * tempk * tempdot
//    //      v * k.label * kernel_func.evaluate(data.features, k.features)
//    //    }
//    //      .reduce((a, b) => a + b)
//
//    s * model.map { case (k, v) => v * k.label * kernel_func.evaluate(data.features, k.features) }.reduce((a, b) => a + b)
//
//  }
//
//  /** Evaluate the accuracy given the test set as a local array */
//  def getAccuracy(data: Array[LabeledPoint]): Double = {
//    val N_c = data.map(x => predict(x) * x.label).count(x => x > 0)
//    val N = data.count(x => true)
//    N_c.toDouble / N
//
//  }
//
//  def getAccuracy(data: RDD[LabeledPoint]): Double = {
//    val N_c = data.map(x => predict(x) * x.label).filter(_ > 0).count()
//    val N = data.count()
//    N_c.toDouble / N
//
//  }
//
//  /** reset the regularization lambda */
//  def setLambda(new_lambda: Double) {
//    lambda = new_lambda
//  }
//
//}
//
//object ZFKmeansSVM_WR {
//  def main(args: Array[String]) {
//
//    Logger.getLogger("org").setLevel(Level.OFF)
//    Logger.getLogger("akka").setLevel(Level.OFF)
//    val logFile = "README.md" // Should be some file on your system
//    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local[30]").set("spark.driver.maxResultSize","2g")
//    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
//    val sc = new SparkContext(conf)
//    // /Users/zhangfan/Documents/data/zflittle /Users/zhangfan/Documents/data/zflittle 1 3 100 false 4 1 /Users/zhangfan/Documents/cancel 2 2
//    val dataPath = args(0)
//    val testPath = args(1)
//    val packN = args(2).toInt
//    val itN = args(3).toInt
//    val ratioL = args(4).split(",").map(_.toDouble)
//    val isSparse = args(5).toBoolean
//    val numFeature = args(6).toInt
//    val minPartN = args(7).toInt
//    val checkPointDir = args(8)
//
//    val kmeansK = args(9).toInt
//    val kmeansItN = args(10).toInt
//
//
//
//    val splitChar = ","
//
//    val data: RDD[LabeledPoint] = if (!isSparse) {
//      sc.textFile(dataPath, minPartN).map(line => {
//        val vs = line.split(splitChar).map(_.toDouble)
//        val features = vs.slice(0, vs.length - 1)
//        LabeledPoint(vs.last, Vectors.dense(features))
//      })
//    } else {
//      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
//    }
//
//    //kmeans
//    val zipTime = System.currentTimeMillis()
//    val kmeansData = data.map(lp => lp.features).cache()
//    val clusters = KMeans.train(kmeansData, kmeansK, kmeansItN)
//    // IndexedRDD[Long, Tuple2[LabeledPoint, Double]]
//    var indexedData :IndexedRDD[Long,(LabeledPoint, Double)]= IndexedRDD(data.zipWithUniqueId().map { case(k,v) => (v,(k,0D)) })
//    /** zipData [centerId,(lp,0.0b,Array(pointIndex))] */
//    val zipData: RDD[(Long, Tuple3[LabeledPoint, Double, Array[Long]])] = indexedData
//      .map { case (id, lpb) => Tuple2(clusters.predict(lpb._1.features).toLong, Tuple2(id, lpb._1)) }
//      .aggregateByKey(Tuple3(Array(0.0), Vectors.zeros(numFeature), new ArrayBuffer[Long]()))(
//        seqOp = (u, v) => {
//          //v (dataId,b)
//          u._1(0) += v._2.label
//          ZFBLAS.axpy(1.0, v._2.features, u._2)
//
//          u._3 += v._1
//          u
//        },
//        combOp = (u1, u2) => {
//          u1._1(0) += u2._1.last
//          ZFBLAS.axpy(1.0, u2._2, u1._2)
//          u1._3 ++= u2._3
//          u1
//        }
//      )
//      .map(t4 => {
//        val n = t4._2._3.size
//        val vec = t4._2._2
//        ZFBLAS.scal(1.0 / n, vec)
//        val lp = new LabeledPoint(t4._2._1.last / n, vec)
//        Tuple2(t4._1, Tuple3(lp, 0.0D, t4._2._3.toArray))
//      })
//    val cancel = zipData.collect()
//    println("zipTime:\t" + (System.currentTimeMillis() - zipTime))
//    println("zipN: " + cancel.length + "\t[,\t" + cancel.slice(0, 10).map(_._2._3.length).mkString(","))
//
//    val tempP = new ArrayBuffer[String]()
//    for (r <- ratioL) {
//      val ratio = r / 100.0
//      val splits = indexedData.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
//      val train = if (testPath.length > 3) IndexedRDD(indexedData.sample(withReplacement = false, ratio)).cache() else IndexedRDD(splits(0).sample(withReplacement = false, ratio)).cache()
//      val testData: RDD[LabeledPoint] = if (testPath.length > 3) {
//        if (!isSparse) {
//          sc.textFile(testPath).map(line => {
//            val vs = line.split(splitChar).map(_.toDouble)
//            val features = vs.slice(0, vs.length - 1)
//            LabeledPoint(vs.last, Vectors.dense(features))
//          })
//        } else {
//          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
//        }
//      } else splits(1).map(_._2._1)
//
//      println("partN:\t" + train.getNumPartitions)
//      val m = train.count()
//      //      var num_iter = 0
//      //      num_iter = (2 * m).toInt
//      val t1 = System.currentTimeMillis
//      val svm = new ZFKmeansSVM_WR(train, zipData.collect(), testData, 1.0 / m, "rbf", 0.1, checkPointDir)
//      svm.train(itN * packN, packN)
//      val t2 = System.currentTimeMillis
//      val runtime = t2 - t1
//
//      println("model count:\t" + svm.model.count())
//      println("Ratio\tDataN\titN\tpackN\tACC\tT")
//
//      val acc = svm.getAccuracy(testData)
//      var ss = ratio + "\t" + m + "\t" + (itN * packN) + "\t" + packN + "\t" + acc + "\t" + runtime + "\n"
//      System.out.println(ss)
//      tempP += ss
//      data.unpersist()
//      train.unpersist()
//
//    }
//    println("Ratio\tDataN\titN\tpackN\tACC\tT")
//    println(tempP.mkString("\n"))
//
//  }
//}
