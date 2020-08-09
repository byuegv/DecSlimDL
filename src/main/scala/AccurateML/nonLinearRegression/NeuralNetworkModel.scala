package AccurateML.nonLinearRegression


import breeze.linalg.{DenseVector => BDV}


/**
  * @author osboxes
  */
class NeuralNetworkModel(inputDim: Int, hiddenUnits: Int) extends Serializable with NonlinearModel {
  var nodes: Int = hiddenUnits
  var n: Int = inputDim
  var dim: Int = (n + 2) * nodes


  def eval(w: BDV[Double], x: BDV[Double]): Double = {
    assert(x.size == n)
    assert(w.size == dim)
    var f: Double = 0.0
    for (i <- 0 to nodes - 1) {
      var arg: Double = 0.0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      f = f + w(i * (n + 2) + n + 1) / (1.0 + Math.exp(-arg))
    }
    return f
  }

  def grad(w: BDV[Double], x: BDV[Double]): BDV[Double] = {
    assert(x.size == n)
    assert(w.size == dim)

    var gper: BDV[Double] = BDV.zeros(dim) // (n+2)*nodes

    for (i <- 0 to nodes - 1) {
      var arg: Double = 0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      var sig: Double = 1.0 / (1.0 + Math.exp(-arg))

      gper(i * (n + 2) + n + 1) = sig
      gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
      for (j <- 0 to n - 1) {
        gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
      }
    }
    return gper;
  }

  def zfGrad(w: BDV[Double], x: BDV[Double], normBound: Double): Boolean = {
    assert(x.size == n)
    assert(w.size == dim)

    var sum = 0.0

    //    var gper: BDV[Double] = BDV.zeros(dim) // (n+2)*nodes

    for (i <- 0 to nodes - 1) {
      var arg: Double = 0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      var sig: Double = 1.0 / (1.0 + Math.exp(-arg))

      //      gper(i * (n + 2) + n + 1) = sig
      //      gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
      sum += sig
      sum += w(i * (n + 2) + n + 1) * sig * (1 - sig)
      if (sum > normBound)
        return true

      for (j <- 0 to n - 1) {
        //        gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
        sum += x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
        if (sum > normBound)
          return true
      }
    }
    //    return gper;
    return false
  }

  def zfGrad2(w: BDV[Double], x: BDV[Double]): Double = {
    assert(x.size == n)
    assert(w.size == dim)

    var sum = 0.0

    //    val gper: BDV[Double] = BDV.zeros(dim) // (n+2)*nodes

    for (i <- 0 to 0) {
      //nodes - 1
      var arg: Double = 0
      for (j <- 0 to n - 1) {
        arg = arg + x(j) * w(i * (n + 2) + j)
      }
      arg = arg + w(i * (n + 2) + n)
      val sig: Double = 1.0 / (1.0 + Math.exp(-arg))
      //      gper(i * (n + 2) + n + 1) = sig
      //      gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
      sum += math.pow(sig, 2)
      sum += math.pow(w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)

      for (j <- 0 to n - 1) {
        //        gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
        sum += math.pow(x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig), 2)
      }
    }
    //    return gper;
    return sum //> normBound
  }

  def getDim(): Int = {
    return dim
  }

  def getNodes(): Int = {
    return nodes
  }

  def setNodes(n: Int) = {
    nodes = n
    dim = (n + 2) * nodes

  }


  def gradnumer(w: BDV[Double], x: BDV[Double]): BDV[Double] = {
    var h: Double = 0.000001
    var g: BDV[Double] = BDV.zeros(this.dim)
    var xtemp: BDV[Double] = BDV.zeros(this.dim)
    xtemp = w.copy
    var f0 = eval(xtemp, x)

    for (i <- 0 until this.dim) {
      xtemp = w.copy
      xtemp(i) += h
      var f1 = eval(xtemp, x)
      g(i) = (f1 - f0) / h
    }
    return g
  }
}