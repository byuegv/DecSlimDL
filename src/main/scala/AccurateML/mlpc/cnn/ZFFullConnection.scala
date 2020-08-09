package AccurateML.mlpc.cnn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

//import org.apache.spark.util.random.XORShiftRandom
import scala.util.Random

/**
  * Created by zhangfan on 18/5/28.
  */
class ZFFullConnection {

}

object ZFFullConnection {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    //    val dataPath = "/Users/zhangfan/Documents/zf.txt"
    //    val numFeature = 6
    //    val hiddenN = 10
    //    val outN = 3
    //    val batchSize = 9
    //    val numIt = 100

    val dataPath = "/Users/zhangfan/Downloads/mnist/test.txt"
    val numFeature = 28 * 28
    val hiddenN = 80
    val outN = 10
    val batchSize = 10000
    val numIt = 100

    val data = MLUtils.loadLibSVMFile(sc, dataPath, numFeature, 1).map(lp => {
      val inc = new BDM[Double](numFeature, 1, lp.features.toArray)
      val c = new Array[Double](outN)
      c(lp.label.toInt) = 1
      val label = new BDM[Double](outN, 1, c)
      Tuple2(label, inc)
    })
    val dataN = data.count()
    val batchSamples = data.collect() //take(batchSize)

    val weights0 = new BDM[Double](hiddenN, numFeature, Array.fill(hiddenN * numFeature)(1.0))
    val weights1 = new BDM[Double](outN, hiddenN, Array.fill(hiddenN * outN)(1.0))
    val bais0 = new BDM[Double](hiddenN, 1, Array.fill(hiddenN)(1.0))
    val bais1 = new BDM[Double](outN, 1, Array.fill(outN)(1.0))
    initWeights(weights0, bais0)
    initWeights(weights1, bais1)


    for (it <- 1 to numIt) {
      val w0GradientSum = new BDM[Double](hiddenN, numFeature)
      val w1GradientSum = new BDM[Double](outN, hiddenN)
      val b0GradientSum = new BDM[Double](hiddenN, 1)
      val b1GradientSum = new BDM[Double](outN, 1)

      for (asample <- batchSamples) {
        val (labelVec, featureVec) = asample
        val out1 = forward(featureVec, weights0, bais0)
        val out2 = forwardout(out1, weights1, bais1)
        val delta2 = labelVec - out2
        val delta1 = getDelta(out1, weights1, delta2)
        val grad0 = delta1 * featureVec.t
        val gradb0 = delta1.copy
        val grad1 = delta2 * out1.t
        val gradb1 = delta2.copy

        w0GradientSum += grad0
        w1GradientSum += grad1
        b0GradientSum += gradb0
        b1GradientSum += gradb1
      }
      val batchN = batchSamples.size.toDouble
      weights0 += w0GradientSum / batchN
      weights1 += w1GradientSum / batchN
      bais0 += b0GradientSum / batchN
      bais1 += b1GradientSum / batchN

      if (it % 10 == 0) {
        val (rightN, loss) = data.map(t => {
          val (label, inc) = t
          val fout1 = forward(inc, weights0, bais0)
          val fout2 = forwardout(fout1, weights1, bais1)
          val fdelta2 = fout2 - label
          val loss = fdelta2.toArray.map(d => d * d).sum
          val pindex = fout2.toArray.zipWithIndex.maxBy(_._1)._2
          val index = label.toArray.zipWithIndex.maxBy(_._1)._2
          val right = if (index == pindex) 1 else 0
          Tuple2(right, loss)
        }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
        println(it + ", ACC," + rightN.toDouble / dataN + ",loss," + loss)
      }

    }


  }

  def forward(input: BDM[Double], weights: BDM[Double], bais: BDM[Double]): BDM[Double] = {
    val out: BDM[Double] = weights * input + bais
    ApplyInPlace(out, sigmoid)
    out
  }

  def forwardout(input: BDM[Double], weights: BDM[Double], bais: BDM[Double]): BDM[Double] = {
    val out: BDM[Double] = weights * input + bais
    val max = out.toArray.max
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        out(i, j) = math.exp(out(i, j) - max)
      }
    }
    val sum = out.toArray.sum
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        out(i, j) /= sum
      }
    }

    out
  }

  def sigmoid(x: Double): Double = {
    1.0 / (1 + math.exp(-x))
  }

  def initWeights(m: BDM[Double], b: BDM[Double]): Unit = {
    val numIn = m.cols
    val sqrtInt = math.sqrt(numIn)
    val random = new Random
    for (i <- 0 until m.rows) {
      b(i, 0) = (random.nextDouble() * 4.8 - 2.4) / sqrtInt
      for (j <- 0 until m.cols) {
        m(i, j) = (random.nextDouble() * 4.8 - 2.4) / sqrtInt
      }
    }
  }

  def getDelta(out: BDM[Double], nextWeights: BDM[Double], nextDelta: BDM[Double]): BDM[Double] = {
    val deltatemp: BDM[Double] = nextWeights.t * nextDelta
    val delta = new Array[Double](out.rows)
    for (i <- 0 until delta.length) {
      delta(i) = (1 - out(i, 0)) * out(i, 0) * deltatemp(i, 0)
    }
    new BDM[Double](deltatemp.rows, 1, delta)
  }

  object ApplyInPlace {
    def apply(out: BDM[Double], sigmoid: (Double) => Double) = {
      for (i <- 0 until out.rows) {
        for (j <- 0 until out.cols)
          out(i, j) = sigmoid(out(i, j))
      }
    }


    // TODO: use Breeze UFunc
    def apply(x: BDM[Double], y: BDM[Double], func: Double => Double): Unit = {
      var i = 0
      while (i < x.rows) {
        var j = 0
        while (j < x.cols) {
          y(i, j) = func(x(i, j))
          j += 1
        }
        i += 1
      }
    }

    // TODO: use Breeze UFunc
    def apply(
               x1: BDM[Double],
               x2: BDM[Double],
               y: BDM[Double],
               func: (Double, Double) => Double): Unit = {
      var i = 0
      while (i < x1.rows) {
        var j = 0
        while (j < x1.cols) {
          y(i, j) = func(x1(i, j), x2(i, j))
          j += 1
        }
        i += 1
      }
    }
  }


}
