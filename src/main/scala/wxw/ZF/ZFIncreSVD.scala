package cn.wangxw.ZF

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by zhangfan on 16/12/8.
  */
class IncreSVD(
                val rating: Array[LabeledPoint],
                val indexs: Array[Int],
                val nf: Int, //设置的需要生成新属性的个数
                val round: Int,
                val ratioN: Int, //10 min is 1
                val initValue: Double = 0.1,
                val lrate: Double = 0.001, //0.001
                val k: Double = 0.015 //0.015
              ) extends Serializable {
  val n = indexs.size
  //原始数据集实例个数
  val m = rating.last.features.size
  //原始数据集属性个数
  val movieFeas = new DenseMatrix[Double](nf, m, Array.fill(nf * m)(initValue))
  val userFeas = new DenseMatrix[Double](nf, n, Array.fill(nf * n)(initValue))
  val zfInf = 30 //3

  // 求均方差：
  def zfmse(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    val diff: DenseMatrix[Double] = a - b
    var mse = 0.0
    for (i <- 0 until diff.rows) {
      for (j <- 0 until diff.cols) {
        mse += diff(i, j) * diff(i, j)
      }
    }
    mse = sqrt(mse / (diff.rows.toDouble * diff.cols.toDouble))
    mse

  }

  def calcFeaaturesSparse(): Unit = {

    for (f <- 0 until nf) {
      for (r <- 0 until round) {
        for (ku <- 0 until n) {
          var cnt = 0
          rating(indexs(ku)).features.foreachActive((i, value) => {
            if (cnt % ratioN == 0) {
              val km = i
              val p = predictRating(km, ku)

              val err = rating(indexs(ku)).features.apply(km) - p
              val cf = userFeas(f, ku)
              val mf = movieFeas(f, km)


              userFeas(f, ku) += lrate * (err * mf - k * cf)
              if (userFeas(f, ku).equals(Double.NaN) || userFeas(f, ku).equals(Double.PositiveInfinity) || userFeas(f, ku).equals(Double.NegativeInfinity)) {
              //  System.err.println("Double.NaN")
              }
              movieFeas(f, km) += lrate * (err * cf - k * mf)
            }
            cnt += 1
          })
        }
      }

    }
  }

  /**
    * @param isSparse
    */
  def calcFeaatures(isSparse: Boolean = false): Unit = {
    if (isSparse) {
      calcFeaaturesSparse()
    } else {
      val sliceM: Int = math.round((m / ratioN.toFloat))
      for (f <- 0 until nf) {
        for (r <- 0 until round) {
          for (ku <- 0 until n) {
            var kmN = 0
            while (kmN < sliceM) {
              val km = (f + kmN * sliceM) % m
              //  val km = (itIndex+kmN)%m
              val p = predictRating(km, ku)

              val err = rating(indexs(ku)).features.apply(km) - p
              val cf = userFeas(f, ku)
              val mf = movieFeas(f, km)

              userFeas(f, ku) += lrate * (err * mf - k * cf)
              if (userFeas(f, ku).equals(Double.NaN) || userFeas(f, ku).equals(Double.PositiveInfinity) || userFeas(f, ku).equals(Double.NegativeInfinity)) {
                System.err.println("Double.NaN")
              }
              movieFeas(f, km) += lrate * (err * cf - k * mf)

              kmN += 1
            }
          }
        }
      }
    }


  }

  // 截断为 ±zfInf 范围内：
  def zfScaler(a: Double): Double = {
    var sum = a
    if (math.abs(sum) > zfInf) {
      //      println("zfInf",+sum)
      sum = if (sum > 0) zfInf else -zfInf
    }
    sum

  }

  def predictRating(mid: Int, uid: Int): Double = {
    var p = 0.0
    for (f <- 0 until nf) {
      p += userFeas(f, uid) * movieFeas(f, mid)
    }
    zfScaler(p)
  }

  def predictRating(mid: Int, uid: Int, fi: Int, acache: Double, bTrailing: Boolean): Double = {
    //    var sum: Double = if (acache > 0) acache else 1.0
    var sum = acache
    sum += movieFeas(fi, mid) * userFeas(fi, uid)
    sum = zfScaler(sum)
    if (bTrailing) {
      sum += (nf - fi - 1) * (initValue * initValue)
      sum = zfScaler(sum)
    }
    sum
  }

}


