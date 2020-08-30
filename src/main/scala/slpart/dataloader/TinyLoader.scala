package slpart.dataloader
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.{ByteRecord, LabeledPointToSample, Sample}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.utils.{File, T}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StandardScaler
import slpart.dataloader.TinyImageLoader.rescaleLabel

import scala.collection.mutable.ArrayBuffer

object TinyLoader {
  private val hdfsPrefix: String = "hdfs:"
  /**
   * get the training data
   * @param path the directory where dataset in
   * @param isZeroScore weather to zeroScore training data
   * @return
   */
  def loadLabeledPoint(sc: SparkContext, path: String, isZeroScore: Boolean = true,class_num: Int = 100) = {
    assert(sc != null,"SparkContext can not be null")
    val trainPath = s"${path}/train"
    val valPath = s"${path}/val"
    val trained = sc.textFile(trainPath).map(line => {
      val splits = line.trim.split(",")
      val label = splits.head.toInt % class_num  // change class labe to 0 - class_num-1
      val features = splits.tail.map(_.toDouble)
      LabeledPoint(label.toDouble,Vectors.dense(features))
    })
    val validate = sc.textFile(valPath).map(line => {
      val splits = line.trim.split(",")
      val label = splits.head.toDouble
      val features = splits.tail.map(_.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    })
    if(isZeroScore){
      val scaler = new StandardScaler(true, true)
        .fit(trained.map(x => x.features))
      val trainScaled = trained.map(x => LabeledPoint(x.label,
        scaler.transform(x.features)))
      val validScaled = validate.map(x => LabeledPoint(x.label,
        scaler.transform(x.features)))
      (trainScaled,validScaled)
    }
    else{
      (trained,validate)
    }
  }

  def trainingSamplesAry(sc: SparkContext,path: String,zScore: Boolean = true,isAggregate: Boolean = false,category: Int = 100,
                         itqbitN: Int = 1,
                         itqitN: Int = 20, //压缩算法迭代次数
                         itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                         upBound: Int = 20, //每个压缩点包含原始点个数上限
                         splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                         isSparse: Boolean = false //输入数据是否为libsvm格式
                        ) = {
    if(isAggregate){
      val trainlp = loadLabeledPoint(sc,path,zScore,class_num = category)._1
      System.out.println("generate compressed training Samples  ...\n +" +
        s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
        s" splitN: ${splitN} isSparse: ${isSparse}")
      val tp = Aggregator.singleLayerAggregateAry(category,trainlp,
        itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
        splitN = splitN, isSparse = isSparse)
      tp.zipWithUniqueId().map(x => (x._2,x._1))
    }
    else{
      val trainsp = loadLabeledPoint(sc,path,zScore,class_num = category)._1.map(x => {
        val xs2 = x.features.toArray.map(_.toFloat)
        Sample[Float](Tensor(T(xs2.head,xs2.tail: _*)),Tensor(T(x.label.toFloat)))
      })
      val arySap = trainsp.zipWithIndex.map(x => (x._2.toLong,Array(x._1)))
      arySap
    }
  }
  def validateSamples(sc: SparkContext,path: String,zScore: Boolean = true,category: Int = 100) = {
    loadLabeledPoint(sc, path, zScore,class_num = category)._2.map(x => {
      val xs2 = x.features.toArray.map(_.toFloat)
      Sample[Float](Tensor(T(xs2.head,xs2.tail: _*)),Tensor(T(x.label.toFloat)))
    })
  }

}
