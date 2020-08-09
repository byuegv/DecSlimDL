package com.intel.analytics.bigdl.optim

import breeze.linalg.*
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.math._
import scala.reflect.ClassTag

class RectifiedAdam[@specialized(Float,Double) T: ClassTag](
      var learningRate: Double = 1e-3,
      var learningRateDecay: Double = 0.0,
      var beta1: Double = 0.9,
      var beta2: Double = 0.999,
      var Epsilon: Double = 1e-8,
      var weightDecay: Double = 0.0)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
  @transient
  private var buffer: Tensor[T] = null

  /**
    * An implementation of Rectified Adam https://arxiv.org/abs/1908.03265
    *
    * @param feval a function that takes a single input (X), the point of a evaluation, and
    *              returns f(X) and df/dX
    * @param parameter the initial point
    * @return the new x vector and the function list {fx}, evaluated before the update
    */
  override def optimize(feval: Tensor[T] => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    if(buffer == null) buffer = Tensor[T]()

    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon
    val wd = this.weightDecay

    val (fx,dfdx) = feval(parameter) // compute gradients

    var timestep = state.getOrElse[Int]("evalCounter",0)

    val (_m,_v,_denom) = if(state.get[Tensor[T]]("m").isDefined){
      (state.get[Tensor[T]]("m").get,state.get[Tensor[T]]("v").get,
      state.get[Tensor[T]]("denom").get.resizeAs(dfdx))
    }
    else{
      (Tensor[T]().resizeAs(dfdx).zero(),Tensor[T]().resizeAs(dfdx).zero(),
      Tensor[T]().resizeAs(dfdx).zero())
    }

    val clr = lr / (1 + timestep * lrd)
    timestep = timestep + 1
    val rhoSMA: Double = 2.0/(1 - beta2) - 1 // compute the maximum length of the approximated SMA
    /**
      * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
      * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
      */
    _m.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1-beta1),dfdx)
    // buffer = dfdx * dfdx
    buffer.resizeAs(dfdx).cmul(dfdx,dfdx)
    _v.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1-beta2),buffer)
    _denom.sqrt(_v)

    // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
    buffer.fill(ev.one)
    _denom.add(ev.fromType(eps),buffer)

    // efficiency improved upon by changing the order of computation, at expense of clarity
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val biasCorrection2 = 1 - pow(beta2, timestep)

    // compute bias-corrected moving average
//    val _m_t = _m / ev.fromType[Double]( biasCorrection1)
    // compute the length of the approximated SMA
    val rhot = rhoSMA - 2 * timestep * pow(beta2,timestep) / biasCorrection2

    // more conservative since it's an approximate value
    if(rhot > 4){
//      val _v_t = Tensor[T]().resizeAs(_v).copy(_v).div(ev.fromType[Double](biasCorrection2)).sqrt()
//      val tp1 = (betat-4)*(betat-2)*betaSMA
//      val tp2 = (betaSMA-4)*(betaSMA-2)*betat
//      val _r_t = Math.sqrt(tp1/tp2)

      val stepSize = clr * Math.sqrt(biasCorrection2*(rhot-4)/(rhoSMA-4)*(rhot-2)/rhot*rhoSMA/(rhoSMA-2)) / biasCorrection1

      if(wd != 0) {parameter.add(ev.fromType(-clr*wd),parameter)}

      parameter.addcdiv(ev.fromType[Double](-stepSize),_m,_denom)
    }
    else{
      val stepSize = clr / biasCorrection1

      if(wd != 0){ parameter.add(ev.fromType(-clr*wd),parameter)}

      parameter.add(ev.fromType[Double](-stepSize),_m)

    }
    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
    state("m") = _m // 1st moment variables
    state("v") = _v // 2nd moment variables
    state("denom") = _denom // 3nd moment variables
    (parameter,Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this.weightDecay = config.get[Double]("weightDecay").getOrElse(this.weightDecay)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("m")
    state.delete("v")
  }

  override def getLearningRate(): Double = this.learningRate
}
