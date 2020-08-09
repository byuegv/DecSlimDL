package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import slpart.datastruct.InheritSample

class InheritedSampleMiniBatch[T: ClassTag](var samplesAry: Array[InheritSample[T]])
(implicit ev: TensorNumeric[T]) extends MiniBatch[T] {

  protected var coarseMiniBatch = (SampleToMiniBatch(size()).apply(samplesAry.map(_.getCoarseAggregatedSample()).toIterator)).next()

  def getAllOriginalSampleSize() = {
    samplesAry.map(_.originalSampleSize()).reduce(_ + _)
  }

  def getInheritSampleAry() = {
    samplesAry
  }

  def getInheritSampltAt(index: Int) = {
    require(index >=1 && index <= samplesAry.length,s"require index=${index} larger than 0 and no more than ${samplesAry.length}")
    samplesAry(index-1)
  }

  def getSeveralInhertSample(index: Array[Int]) = {
    require(index.length <= samplesAry.length)
    index.map(idx => samplesAry(idx))
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    require(offset > 0,"The offset should larger than 0")
    val nsamples = samplesAry.slice(offset-1,offset - 1 + length)
    new InheritedSampleMiniBatch[T](nsamples)
  }
  override def size(): Int = samplesAry.length

  override def getInput(): Activity = coarseMiniBatch.getInput()

  override def getTarget(): Activity = coarseMiniBatch.getTarget()

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = this

  def set(rsamplesAry: Array[InheritSample[T]]): this.type  = {
    samplesAry = rsamplesAry
    coarseMiniBatch = (SampleToMiniBatch(size()).apply(samplesAry.map(ih => ih.getCoarseAggregatedSample()).toIterator)).next()
    this
  }
}

object InheritedSampleMiniBatch{
  def apply[T: ClassTag](samplesAry: Array[InheritSample[T]])
           (implicit ev: TensorNumeric[T]): InheritedSampleMiniBatch[T] = new InheritedSampleMiniBatch[T](samplesAry)
}