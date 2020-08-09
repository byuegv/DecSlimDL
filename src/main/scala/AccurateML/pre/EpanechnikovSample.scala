package org.apache.spark.mllib.zfpre

import java.util.Random

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{BLAS, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 17/5/15.
  */
class EpanechnikovSample {

}

object EpanechnikovSample {

  def scalerVec(data: RDD[Vector]): RDD[Vector] = {
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data)
    val data2 = data.map(x => scaler2.transform(x))
    return data2
  }

  /**
    *
    * @param data : 已经做过0均值,1方差
    * @param sN    : 用于初始化的sample的个数
    * @param a     : 指数
    * @param ratio :
    * @param isSparse
    * @param sortP : 根据求出的数据实例被选中的概率p,抽取ratio比例的时候,如果sortP设置为false,则使用rand根据p随机抽取数据;如果true,按照p排序从大到小抽取
    * @return
    */
  def epaneSample(data: RDD[Vector], sN: Int, a: Int, ratio: Double, isSparse: Boolean, sortP: Boolean): RDD[Vector] = {
    val samples: Array[Vector] = data.take(sN)
    val d = samples.last.size
    val ab = math.pow(5.0, 0.5) * math.pow(samples.size, -1.0 / (d + 4))
    val comm = math.pow((3 / 4.0), d) / (math.pow(ab, d))

//    val si_mean: Vector = {
//      val (count, sum) = data.treeAggregate((Array(0.0), Vectors.zeros(d)))(
//        seqOp = (u, v) => {
//          u._1(0) += 1
//          BLAS.axpy(1.0, v, u._2)
//          u
//        },
//        combOp = (u1, u2) => {
//          u2._1(0) += u1._1(0)
//          BLAS.axpy(1.0, u1._2, u2._2)
//          u2
//        })
//      BLAS.scal(1.0 / count(0), sum)
//      sum
//    }
//    data.treeAggregate(Array.fill(d)(0.0))(
//      seqOp = (u, v) => {
//        v.foreachActive { (i, value) => {
//          u(i) += math.pow(value - si_mean(i), 2)
//        }
//        }
//        u
//      },
//      combOp = (u1, u2) => {
//        for (i <- 0 until u2.size) u2(i) += u1(i)
//        u2
//      }
//    )
//    val si = data.map(vec => {
//      BLAS.axpy(-1, si_mean, vec)
//      vec.toArray.map(math.pow(_, 2.0))
//    }).


    //distribute sample
    val fx: RDD[Double] = data.map(vec => {
      var ksum = 0.0
      for (sample <- samples) {

        var dot = 1.0
        if (!isSparse) {
          val diff: Vector = sample.copy
          BLAS.axpy(-1, vec, diff)
          var vi = 0
          while (vi < diff.size && dot != 0) {
            val xB = diff(vi) / ab
            if (math.abs(xB) >= 1) {
              dot = 0
            } else {
              dot *= (1 - math.pow(xB, 2.0))
            }
            vi += 1
          }
        } else {
          (sample, vec) match {
            case (v1: SparseVector, v2: SparseVector) =>
              val v1Values = v1.values
              val v1Indices = v1.indices
              val v2Values = v2.values
              val v2Indices = v2.indices
              val nnzv1 = v1Indices.length
              val nnzv2 = v2Indices.length

              var kv1 = 0
              var kv2 = 0
              while ((kv1 < nnzv1 || kv2 < nnzv2) && dot != 0) {
                var score = 0.0

                if (kv2 >= nnzv2 || (kv1 < nnzv1 && v1Indices(kv1) < v2Indices(kv2))) {
                  score = v1Values(kv1)
                  kv1 += 1
                } else if (kv1 >= nnzv1 || (kv2 < nnzv2 && v2Indices(kv2) < v1Indices(kv1))) {
                  score = v2Values(kv2)
                  kv2 += 1
                } else {
                  score = v1Values(kv1) - v2Values(kv2)
                  kv1 += 1
                  kv2 += 1
                }

                val xB = score / ab
                if (math.abs(xB) >= 1) {
                  dot = 0
                } else {
                  dot *= (1 - math.pow(xB, 2.0))
                }
              }
          }
        } //end-sparse dot
        ksum += dot
      }

      val ans = ksum * comm / samples.size
      math.pow(ans, a)
    }).persist(StorageLevel.MEMORY_AND_DISK)


    val selectData: RDD[Vector] = if (!sortP) {
      val k = fx.sum()
      val ratioB = ratio * data.count()
      val p: RDD[Double] = fx.map(x => ratioB * x / k)
      val rand = new Random()
      data.zip(p).filter(t => {
        t._2 >= rand.nextDouble()
      }).map(t => t._1)
    } else {
      data.zip(fx).mapPartitions(pit => {
        val probs = new ArrayBuffer[(Vector, Double)]()
        while (pit.hasNext) {
          probs += pit.next()
        }
        val partN = probs.size
        probs.sortBy(-_._2).slice(0, (ratio * partN).toInt).map(_._1).toIterator
      })
    }

    selectData

  }

  def randSample(data: RDD[Vector], ratio: Double, dataPath: String): Unit = {
    val selectData = data.sample(false, ratio)
    selectData.map(_.toArray.mkString(",")).repartition(1).saveAsTextFile(dataPath + ".rand")
    println("rand N", +selectData.count())

  }


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("gmm")
    val sc = new SparkContext(conf)



    //    KMeans.train(parsedData, numClusters, numIterations)


    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Downloads/r.data.scaler"
    val test100: Array[Double] = args(5).split(",").map(_.toDouble)

    val sortP = args(6).toBoolean

    val isSparse = args(7).toBoolean
    val minPartN = args(8).toInt
    val sN = args(9).toInt //random select substract of data to get sN kernel


    val a = 1 //math.pow(p,a)

    val data = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
        .map(point => point.features)
    } else {
      sc.textFile(dataPath, minPartN).map(line => Vectors.dense(line.split(",").map(_.toDouble)))
    }
    val allDataN = data.count()
    println("allDataN, " + allDataN)

    for (r <- test100) {
      val ratio = r / 100.0
      val fN = data.first().size
      //      val sparseData = data.map(vec => Vectors.sparse(fN, Array.range(0, fN), vec.toArray))
      //      epaneSample(sparseData, sN, ratio, dataPath + "." + ratio, true)
      //      epaneSample(data, sN, ratio, dataPath + "." + ratio, false)
      //      randSample(data, ratio, dataPath + "." + ratio)


      var time = System.currentTimeMillis()
      val selectData = epaneSample(data, sN, a, ratio, isSparse, sortP).persist(StorageLevel.MEMORY_AND_DISK)
      //      selectData.map(vec => vec.toArray.mkString(",")).repartition(1).saveAsTextFile(dataPath + ".epane."+ratio+"."+ sN)
      val selectN = selectData.count()
      time = System.currentTimeMillis() - time
      println("expectN, " + (allDataN * ratio).toInt + "\tepaneN," + selectN + "\tselectT, " + time) //trigger selectData

      time = System.currentTimeMillis()
      val train = selectData
      val clusters = org.apache.spark.mllib.clustering.KMeans.train(train, k, itN)
      time = System.currentTimeMillis() - time
      val WSSSE = clusters.computeCost(data)
      println("ratio, " + ratio + "\t WSSSE, " + WSSSE + "\ttrainN, " + train.count() + "\ttrainT, " + time) //Within Set Sum of Squared Errors


      println()


    }


  }
}