package slpart.datastruct

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class InheritSample[T: ClassTag](coarseAgg: Sample[T],fineAgg: Array[Sample[T]],originalSamples: Array[Array[Sample[T]]])
                                (implicit ev: TensorNumeric[T]) extends Serializable{
  def size() = {
    (1,fineAgg.length,originalSamples.map(_.length))
  }
  // get original sample number
  def originalSampleSize() = {
    originalSamples.map(_.length).reduce(_ + _)
  }

  def getCoarseAggregatedSample(): Sample[T] = {
    coarseAgg
  }

  def getfineAggregatedSamples(): Array[Sample[T]] = {
    fineAgg
  }

  def getPartFineAggSamples(index: Int): Sample[T] = {
    require(index < fineAgg.length)
    fineAgg(index)
  }

  def getPartFineAggSamples(index: Array[Int]): Array[Sample[T]] = {
    require(index.length < fineAgg.length)
    index.map(idx => fineAgg(idx))
  }

  def getAllOriginalSamples(): Array[Sample[T]] = {
    originalSamples.reduce((lf,rg) => lf ++ rg)
  }

  def getPartOrigByIndex(index: Array[Int]): Array[Sample[T]] = {
    require(index.length <= fineAgg.length,"part samples should no more than all original samples")
    index.map(idx => originalSamples(idx)).reduce((lf,rg) => lf ++ rg)
  }
  def getPartOrigByIndex(index: Int): Array[Sample[T]] = {
    require(index <= fineAgg.length,"part samples should no more than all original samples")
    originalSamples(index)
  }
}

object InheritSample{
  def apply[T: ClassTag](coarseAgg: Sample[T],fineAgg: Array[Sample[T]],originalSamples: Array[Array[Sample[T]]])
                        (implicit ev: TensorNumeric[T]): InheritSample[T] = {
    new InheritSample[T](coarseAgg,fineAgg,originalSamples)
  }
}