package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.nn.{ EachClassNLLCriterion, LogSoftMax}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * This criterion combines LogSoftMax and ClassNLLCriterion in one single class.
  *
  * @param weights A tensor assigning weight to each of the classes
  */

class EachCrossEntropyCriterion[T: ClassTag](
                                          val weights: Tensor[T] = null,
                                          val sizeAverage: Boolean = false)
                                        (implicit ev: TensorNumeric[T]) extends Serializable {
  private val lsm = new LogSoftMax[T]()
  private val nll = new EachClassNLLCriterion[T](weights, sizeAverage)
  var output: Array[T] = null

  def forward(input: Tensor[T], target: Tensor[T]): Array[T] = {
    updateOutput(input, target)
  }

  def updateOutput(input: Tensor[T], target: Tensor[T]): Array[T] = {
    lsm.updateOutput(input)
    nll.updateOutput(lsm.output, target.asInstanceOf[Tensor[T]])
    output = nll.output
    output
  }
}

object EachCrossEntropyCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      weights: Tensor[T] = null, sizeAverage: Boolean = false)
                                                    (implicit ev: TensorNumeric[T]) : EachCrossEntropyCriterion[T] = {
    new EachCrossEntropyCriterion[T](weights, sizeAverage)
  }
}

