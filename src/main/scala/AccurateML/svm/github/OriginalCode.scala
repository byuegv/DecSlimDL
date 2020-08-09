///*
// * Kernel SVM: the class for kernelized SVM on Spark
// * Using SGD
// * Usage example:
//    //data = some rdd of LabeledPoint
//    //setup amodel by regietering training data, specifying lambda,
//    //specifying kernel and kernel parameters
//    val model = new KernelSVM(data_train, 1.0, "rbf", 1.0)
//    //train the model by specifying # of iterations and packing size
//    model.train(1000,10)
// */
//import org.apache.spark.rdd._
//
//import Array._
//import org.apache.spark.mllib.regression.LabeledPoint
//import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
//import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
//import org.apache.spark.SparkContext
//import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
//import java.lang.System
//
//class KernelSVM(training_data:RDD[LabeledPoint], lambda_s: Double, kernel : String = "rbf",  gamma: Double = 1D, Budget : Int=100, checkpoint_dir: String = "../checkpoint") extends java.io.Serializable{
//  /** Initialization */
//  var lambda = lambda_s
//  //Currently only rbf kernel is implemented
//  var kernel_func = new RbfKernelFunc(gamma)
//  var model = training_data.map(x => (x, 0D))
//  var data = training_data
//  var s = 1D
//  //specify the checkpoint directory
//  data.sparkContext.setCheckpointDir(checkpoint_dir)
//
//  /** method train, based on P-Pack algorithm **/
//  def train(num_iter: Long, pack_size: Int = 1) {
//    //Free the current model from memory
//    model.unpersist()
//
//    var newRDD = data.zipWithUniqueId().map{case (k,v) => (v,(k, 0D))}
//    //Initialization
//    var working_data = IndexedRDD(newRDD)
//    var norm = 0D
//    var alpha = 0D
//    var t = 1
//    var i = 0
//    var j = 0
//    var num_of_updates = 0
//
//    var numb=0
//
//    val pair_idx = data.sparkContext.parallelize(range(0, pack_size).flatMap(x => (range(x, pack_size).map(y => (x,y)))))
//    val broad_kernel_func = data.sparkContext.broadcast(kernel_func)
//
//    // Training the model with pack updating
//
//    while (t <= num_iter) {
//      print(".")
//      var sample = working_data.takeSample(true, pack_size)
//
//      var broad_sample = data.sparkContext.broadcast(sample)
//
//      var yp = broad_sample.value.map(x => (working_data.map{case (k,v) => (v._1.label * v._2 * broad_kernel_func.value.evaluate(v._1.features, x._2._1.features))}.reduce((a, b) => a + b)))
//
//      var y = sample.map(x => x._2._1.label)
//
//      var local_set = Map[Long, (LabeledPoint, Double)]()
//
//      // Compute kernel inner product pairs
//      var inner_prod = pair_idx.map(x => (x, broad_kernel_func.value.evaluate(sample(x._1)._2._1.features, sample(x._2)._2._1.features))).collectAsMap()
//      // Compute sub gradients
//      for (i <- 0 until pack_size) {
//
//        t = t+1
//        s = (1 - 1D/(t))*s
//        for (j <- (i+1) until (pack_size)) {
//          yp(j) = (1 - 1D/(t))*yp(j)
//        }
//        if (y(i) * yp(i) < 1) {
//          norm = norm + (2*y(i)) / (lambda * t) * yp(i) + math.pow((y(i)/(lambda*t)), 2)*inner_prod((i,i))
//          alpha = sample(i)._2._2
//
//          if (alpha==0.0 ) numb = numb +1
//
//          local_set = local_set + (sample(i)._1 -> (sample(i)._2._1, alpha + (1/(lambda*t*s))))
//
//          for (j <- (i+1) to (pack_size-1)) {
//            yp(j) = yp(j) + y(j)/(lambda*t) * inner_prod((i,j))
//          }
//
//          if (norm > (1/lambda)) {
//            s = s * (1/math.sqrt(lambda*norm))
//            norm = (1/lambda)
//            for (j <- (i+1) to (pack_size-1)) {
//              yp(j) = yp(j) /math.sqrt(lambda*norm)
//            }
//          }
//
//        }
//
//      }
//      //batch update model
//      var to_forget = working_data
//      working_data = working_data.multiput(local_set).cache()
//      if (numb>Budget){
//        val minCV = working_data.filter(x=> x._2._2>0).reduce((a, b) => if (a._2._2 < b._2._2) a else b)
//        working_data = working_data.put(minCV._1, (minCV._2._1, 0.0)).cache()
//        numb=numb-1
//      }
//      to_forget.unpersist()
//
//
//
//
//      num_of_updates = num_of_updates + 1
//
//      //checkpoint
//      if (num_of_updates % 100 == 0 ) {
//        working_data.checkpoint()
//      }
//    }
//
//    //keep the effective part of the model
//    model = working_data.map{case (k, v) => (v._1, v._2)}.filter{case (k,v) => (v > 0)}.cache()
//    print("\nb = "+numb+"\n")
//    print("\nTamano model "+model.count()+"\n")
//    working_data.unpersist()
//
//  }
//
//  /** getting the number of support vectors of the trained model */
//  def getNumSupportVectors(): Long = {
//    model.count()
//  }
//
//  /** make a prediction on a single data point */
//  def predict (data: LabeledPoint): Double = {
//    s * (model.map{case (k,v) => v * k.label * kernel_func.evaluate(data.features, k.features)}.reduce((a, b) => a + b))
//
//  }
//
//  /** Evaluate the accuracy given the test set as a local array */
//  def getAccuracyAndAUC(data: Array[LabeledPoint],sc:SparkContext): (Double,Double,Double) = {
//    val t1 = System.currentTimeMillis
//    val N_caux = data.map(x => (predict(x),x.label))
//    val t2 = System.currentTimeMillis
//    val runtime = (t2 - t1)/1000
//    println("Testing time: "+runtime.toString)
//
//    val N_c = N_caux.count(x => x._1>0)
//    val N = data.count(x => true)
//
//    val tuples = sc.parallelize(N_caux)
//    val metrics = new BinaryClassificationMetrics(tuples)
//
//    metrics.roc()
//    (N_c.toDouble / N, metrics.areaUnderROC,runtime)
//
//  }
//
//  /** register a new training data */
//  def registerData(new_training_data:RDD[LabeledPoint]) {
//    data = new_training_data
//  }
//
//  /** reset the regularization lambda */
//  def setLambda(new_lambda: Double) {
//    lambda = new_lambda
//  }
//
//}