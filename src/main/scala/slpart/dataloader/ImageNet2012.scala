package com.intel.analytics.bigdl.slpart.dataloader

import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import slpart.dataloader.Aggregator

object ImageNet2012sl {

  def loadTrain(path: String,sc: SparkContext,zScore: Boolean = true,isAggregate: Boolean = false,category: Int = 10,
                itqbitN: Int = 1,
                itqitN: Int = 20, //压缩算法迭代次数
                itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                upBound: Int = 20, //每个压缩点包含原始点个数上限
                splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                isSparse: Boolean = false //输入数据是否为libsvm格式
    ) = {
    if(isAggregate){
      val lps = sc.textFile(path)
        .map(line => line.trim.split(" "))
        .map(sp => {
          val label = sp.head.trim.toDouble
          val features = sp.last.split(",").map(_.trim.toDouble)
          LabeledPoint(label,Vectors.dense(features))
        })
      val tp = Aggregator.singleLayerAggregateAry(category,lps,
        itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
        splitN = splitN, isSparse = isSparse)
      tp.zipWithUniqueId().map(x => (x._2,x._1))
    }
    else{
      val splits = sc.textFile(path)
        .map(line => line.trim.split(" "))
        .map(sp => {
          val label = sp.head.trim.toFloat
          val features = sp.last.split(",").map(_.trim.toFloat)
          Sample(Tensor(T(features.head,features.tail: _*)),Tensor(T(label)))
        })
      splits.zipWithUniqueId().map(x => (x._2,Array(x._1)))
    }
  }

  def loadValidation(path: String,sc: SparkContext) = {
    val splits = sc.textFile(path)
      .map(line => line.trim.split(" "))
      .map(sp => {
        val label = sp.head.trim.toFloat
        val features = sp.last.split(",").map(_.trim.toFloat)
        Sample(Tensor(T(features.head,features.tail: _*)),Tensor(T(label)))
      })
    splits
  }
}
