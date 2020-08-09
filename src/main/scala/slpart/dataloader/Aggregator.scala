package slpart.dataloader

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import com.intel.analytics.bigdl.numeric.NumericFloat
import slpart.sllsh.ZFHashLayer
import AccurateML.nonLinearRegression.ZFHash3
import com.intel.analytics.bigdl.dataset.{InheritedSampleMiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import slpart.datastruct.InheritSample


object Aggregator {
  private val scale: Double = 1000

  def singleLayerAggregateAry(category: Int,
                           data: RDD[LabeledPoint],
                           itqbitN: Int = 1,//设置的需要生成新属性的个数
                           itqitN: Int = 20, //压缩算法迭代次数
                           itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                           upBound: Int = 20, //每个压缩点包含原始点个数上限
                           splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                           isSparse: Boolean = false //输入数据是否为libsvm格式
                          ) = {
    val sc = data.sparkContext
    val hash = new ZFHashLayer(itqitN,itqratioN,upBound,isSparse, 1,sc, 0)
    val objectData = data.map(p => (p.label.toInt,p))
      .partitionBy(new CategoryPartitioner(category))
      .map(_._2)
      .mapPartitions(p => hash.zfHash(p.toIterable))
//    val maxOrigNum = objectData.map(_._2.length).max()
//    val minOrigNum = objectData.map(_._2.length).min()
//    System.out.println(s"comp - orig 1 - ${maxOrigNum},min: ${minOrigNum}")
    objectData.map(x => {
      val xs1 = x._1.features.toArray.map(_.toFloat)
      val sp1 = Sample[Float](Tensor(T(xs1.head,xs1.tail: _*)),Tensor(T(x._1.label.toFloat)))
      val sp2 = x._2.map(l => {
        val xs2 = l.features.toArray.map(_.toFloat)
        Sample[Float](Tensor(T(xs2.head,xs2.tail: _*)),Tensor(T(l.label.toFloat)))
      })
      Array(sp1) ++ sp2
    })
  }

  def multiLayerAggregateAry(category: Int,
                              data: RDD[LabeledPoint],
                              itqbitN: Int = 1,//设置的需要生成新属性的个数
                              itqitN: Int = 20, //压缩算法迭代次数
                              itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                              upBound: Int = 20, //每个压缩点包含原始点个数上限
                              splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                              isSparse: Boolean = false, //输入数据是否为libsvm格式
                              layerCount: Int = 2
                             ) = {
    val sc = data.sparkContext
    val hash = new ZFHashLayer(itqitN,itqratioN,upBound,isSparse, 2,sc, 0)
    val objectData = data.map(p => (p.label.toInt,p))
      .partitionBy(new CategoryPartitioner(category))
      .map(_._2)
      .mapPartitions(p => hash.zfHashMap(p))


    //    val maxOrigNum = objectData.map(_._2.length).max()
    //    val minOrigNum = objectData.map(_._2.length).min()
    //    System.out.println(s"comp - orig 1 - ${maxOrigNum},min: ${minOrigNum}")

    def lpToSample(lp: LabeledPoint): Sample[Float] = {
      val fea = lp.features.toArray.map(_.toFloat)
      val tlab = lp.label
      val lab = if(tlab < 1.0) 1.0 else if(tlab > category) category else tlab
      Sample[Float](Tensor(T(fea.head,fea.tail: _*)),Tensor(T(lab)))
    }

    objectData.map(zp =>{
      val lps = zp._1
      val lpidx = zp._2
      val coarsesp = lps(0).map(lpToSample).head
      val finesps = lps(1).map(lpToSample).toArray
      val originalsps = lpidx(1).toArray.map(idxs  => idxs.map(idx => lpToSample(lps(2)(idx))))
      InheritSample[Float](coarsesp,finesps,originalsps)
    })
  }

}
