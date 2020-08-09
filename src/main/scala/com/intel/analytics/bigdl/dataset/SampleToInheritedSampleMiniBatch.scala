package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.apache.commons.lang3.SerializationUtils
import java.util

import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import slpart.datastruct.InheritSample

import scala.collection.Iterator
import scala.reflect.ClassTag

class SampleToInheritedSampleMiniBatch[T: ClassTag](totalBatch: Int,miniBatch: Option[MiniBatch[T]] = None,
                                          partitionNum: Option[Int] = None)
                                         (implicit ev: TensorNumeric[T]) extends Transformer[InheritSample[T],InheritedSampleMiniBatch[T]]{
  private val batchPerPartition = Utils.getBatchSize(totalBatch,partitionNum)
  var inheritedMiniBatchBuffer = miniBatch.orNull
  private val batchSize = batchPerPartition
  private val aryBatch = new Array[InheritSample[T]](batchSize)

  override def apply(prev: Iterator[InheritSample[T]]): Iterator[InheritedSampleMiniBatch[T]] = {
    new Iterator[InheritedSampleMiniBatch[T]] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): InheritedSampleMiniBatch[T] = {
        if(prev.hasNext){
          var i = 0
          while(i < batchSize && prev.hasNext){
            val sample = prev.next()
            aryBatch(i) = sample
            i += 1
          }
          var cur = i
          while(i < batchSize){
            val idx = (RandomGenerator.RNG.uniform(0,1.0) * cur).toInt
            val sample = aryBatch(idx)
            aryBatch(i) = sample
            i += 1
          }
          new InheritedSampleMiniBatch[T](aryBatch)
        }
        else{
          null
        }
      }
    }
  }
}

object SampleToInheritedSampleMiniBatch{
  def apply[T: ClassTag](totalBatch: Int, miniBatch: Option[MiniBatch[T]] = None,
                         partitionNum: Option[Int] = None)
                        (implicit ev: TensorNumeric[T]): SampleToInheritedSampleMiniBatch[T] = new SampleToInheritedSampleMiniBatch[T](totalBatch, miniBatch,None)
}
