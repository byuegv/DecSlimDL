package AccurateML.svm.github

/**
  * Created by zhangfan on 17/8/17.
  */

import AccurateML.blas.ZFBLAS
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/** Rbf Kernel, parametrized by gamma */
abstract class Kernels(gamma: Double) extends java.io.Serializable {
  def evaluate(x_1: Vector, x_2: Vector): Double = {
    0.0
  }
}

class RbfKernelFunc(gamma: Double) extends Kernels(gamma) {
  override def evaluate(x_1: Vector, x_2: Vector): Double = {
    math.exp(-1 * gamma * Vectors.sqdist(x_1, x_2))
    //    math.exp(-1 * gamma * math.pow(Vectors.sqdist(x_1, x_2), 2))
  }
}

class PolynomialKernelFunc(gamma: Double, coef0: Double = 0.0, degree: Double = 2.0) extends Kernels(gamma) {

  override def evaluate(x_1: Vector, x_2: Vector): Double = {
    math.pow(gamma * ZFBLAS.dot(x_1, x_2) + coef0, degree)
  }
}

class LinearKernelFunc(gamma: Double = 0.0) extends Kernels(gamma) {

  override def evaluate(x_1: Vector, x_2: Vector): Double = {
    ZFBLAS.dot(x_1, x_2)
  }
}

class TanhKernelFunc(gamma: Double = 0.0) extends Kernels(gamma) {

  override def evaluate(x_1: Vector, x_2: Vector): Double = {
    val xx = gamma * ZFBLAS.dot(x_1, x_2)
    val a = math.exp(xx)
    val b = math.exp(-xx)
    val ans = (a - b) / (a + b)
    ans
  }
}


//class ZFRbfKernelFunc(gamma: Double, lastIndexN: Int) extends java.io.Serializable {
//  var gamma: Double = gamma
//
//  def evaluate(x_1: Vector, x_2: Vector): Double = {
//    math.exp(-1 * gamma * math.pow(ZFCaculate.sqdist(x_1, x_2, lastIndexN), 2))
//  }
//}

