package cn.wangxw.aggpoint

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.SparkContext

class IncreSVD(val rating: Tensor[Float], //ArrayBuffer[LabeledPoint]
               val nf: Int, //设置的需要生成新属性的个数
               val round: Int,
               val initValue: Float = 0.1f,   // 0.1
               val lrate: Float = 0.001f, //0.001
               val k: Float = 0.015f //0.015
              ) extends Serializable {

  val n = rating.size(1)  // 用户数，即图片数
  val m = rating.size(2)  // 电影数，即每张图片的像素数
  val predictCache = Tensor[Float](n, m).fill(0f)
  val movieFeature = Tensor[Float](nf, m).fill(initValue)
  val userFeature = Tensor[Float](nf, n).fill(initValue)

  val zfInf = 255 // 原值为30，改为255后效果更好（Loss更低）

  def calcFeatures(): Unit = {
    if(nf >= m){
      //println(s"nf>=m, nf:${nf}, m:${m}, rating:${rating}, userFeature:${userFeature}")
      userFeature.set(rating).t();
      return;
    }
    for (f <- 1 to nf) {
      //println(s"IncreSVD: nf:${f}/${nf}, ${round} epoch")
      for (r <- 0 until round) {
        for (ku <- 1 to n) {
          // 根据用户 ku 对所有电影评分训练 f 属性
          for (km <- 1 to m) {
            val thisRating = rating.valueAt(ku, km)

            //val p = predictRating(km, ku)  // 无cache计算
            val p = predictRating(km, ku, f)


            val err = thisRating - p
            val cf = userFeature.valueAt(f, ku)
            val mf = movieFeature.valueAt(f, km)

            // lrate决定了这次learning的大小，err决定了learning的方向
            userFeature.setValue(f, ku, cf + lrate * (err * mf - k * cf));
            //if (userFeas(f, ku).equals(Float.NaN) || userFeas(f, ku).equals(Float.PositiveInfinity) || userFeas(f, ku).equals(Float.NegativeInfinity)) {
            //  System.err.println("line 62 Float.NaN")
            //}
            // }
            movieFeature.setValue(f, km, mf + lrate * (err * cf - k * mf))
          }
        }
      }

      // 此时 f 属性已经训练完成，更新cache：
      // 该更新过程不能放到 predictRating 内部，
      // 因为在 属性f 的所有 rating 全部训练完成前，userFeature和movieFeature是一直变化的，
      // 如果在预测 (u,m,f) 后就直接更新 cache(u,m,f)，则在下一次预测 (u,m,f+1) 前，
      // 对 m 从 m+1 开始对 userFeature(f,u) 的更新，和 n 从 n+1 开始对 movieFeature(f,m) 的更新
      // 无法反映到 cache 中
      for(i <- 1 to n){
        for(j <- 1 to m){
          val tmp = predictCache.valueAt(i, j) + movieFeature.valueAt(f, j) * userFeature.valueAt(f, i)
          predictCache.setValue(i, j, tmp)
        }
      }
    }
  }

  def calcFeaturesZF(sampleRatioN: Int): Unit = {
    if(nf >= m){
      //println(s"nf>=m, nf:${nf}, m:${m}, rating:${rating}, userFeature:${userFeature}")
      userFeature.set(rating).t();
      return;
    }
    val sliceM: Int = math.round((m / sampleRatioN).toFloat) // 最终选取多少movie来计算 //    val itIndex = sliceM*hashIt
    for (f <- 1 to nf) {
      //println(s"IncreSVD: nf:${f}/${nf}, ${round} epoch")
      for (r <- 0 until round) {
        for (ku <- 1 to n) {
          // 根据用户 ku 对所有电影评分训练 f 属性
          var kmN = 0
          while (kmN < sliceM) {
            //val km = (f-1 + kmN * sliceM) % m + 1   // 原版ZF代码，但m整除ratioN时，会导致kmN在大于ratioN后：km周期性重复
            val km = (f-1 + kmN * sampleRatioN) % m + 1

            val p = predictRating(km, ku, f)

            val thisRating = rating.valueAt(ku, km)
            val err = thisRating - p

            val cf = userFeature.valueAt(f, ku)
            val mf = movieFeature.valueAt(f, km)


            userFeature.setValue(f, ku, cf + lrate * (err * mf - k * cf));
            //if (userFeas(f, ku).equals(Double.NaN) || userFeas(f, ku).equals(Double.PositiveInfinity) || userFeas(f, ku).equals(Double.NegativeInfinity)) {
            //  System.err.println(s"line 95 Double.NaN ${userFeas(f, ku)}")
            //}
            movieFeature.setValue(f, km, mf + lrate * (err * cf - k * mf))

            kmN += 1
          }
        }
      }

      // 此时 f 属性已经训练完成，更新cache：
      // 该更新过程不能放到 predictRating 内部，
      // 因为在 属性f 的所有 rating 全部训练完成前，userFeature和movieFeature是一直变化的，
      // 如果在预测 (u,m,f) 后就直接更新 cache(u,m,f)，则在下一次预测 (u,m,f+1) 前，
      // 对 m 从 m+1 开始对 userFeature(f,u) 的更新，和 n 从 n+1 开始对 movieFeature(f,m) 的更新
      // 无法反映到 cache 中
      for(i <- 1 to n){
        for(j <- 1 to m){
          val tmp = predictCache.valueAt(i, j) + movieFeature.valueAt(f, j) * userFeature.valueAt(f, i)
          predictCache.setValue(i, j, tmp)
        }
      }
    }
  }

  // 裁切：
  def zfScaler(a: Float): Float = {
    var sum = a
    if (math.abs(sum) > zfInf) {
      //      println("zfInf",+sum)
      sum = if (sum > 0) zfInf else -zfInf
    }
    sum
  }

  /**
   * Predicts the rating for a user/movie pair using
   * all features that have been trained.
   *
   * 使用所有的 user/movie 对
   *
   * @param  mid The movie to predict the rating for.
   * @param  uid The user to predict the rating for.
   * @return The predicted rating for (uid, mid).
   */
  def predictRating(mid: Int, uid: Int): Float = {
    var p = 0.0f
    for (f <- 1 to nf) {
      p += userFeature.valueAt(f, uid) * movieFeature.valueAt(f, mid)
    }
    zfScaler(p)
  }

  /**
   * For use during training.
   *
   * 预测一个 对应于 fi
   *
   * @param  mid    Movie id.
   * @param  uid    User id.
   * @param  bTrailing
   * @return The predicted rating for use during training.
   */
    // 相较于没有cache的预测，速度加快了约25%
  def predictRating(mid: Int, uid: Int, fi: Int, bTrailing: Boolean = true): Float = {
    var sum = predictCache.valueAt(uid, mid) + movieFeature.valueAt(fi, mid) * userFeature.valueAt(fi, uid)
    sum = zfScaler(sum)
    if (bTrailing) {  // TODO: bTrailing 有什么用...
      sum += (nf - fi) * (initValue * initValue)
      sum = zfScaler(sum)
    }
    sum
  }
}

object IncreSVD {
  def calcDistri(rating: Tensor[Float],
                 nf: Int, //设置的需要生成新属性的个数
                 svdRound: Int,
                 sc: SparkContext,
                 initValue: Float = 0.1f,   // 0.1
                 lrate: Float = 0.001f, //0.001
                 k: Float = 0.015f,
                 sampleRatioN: Int = 10  // 使用ZF实现时，对每个user来说，每sampleRatioN个movie的rating中选1个用于训练
                ): Tensor[Float] = {

    val n = rating.size(1)
    val m = rating.size(2)

        val svd = new IncreSVD(rating, nf, svdRound, initValue, lrate, k)
        svd.calcFeaturesZF(sampleRatioN)
        svd.userFeature.t()
  }


  def calcDistri(rating: Tensor[Float],
                 nf: Array[Int], //设置的需要生成新属性的个数
                 svdRound: Int,
                 sc: SparkContext,
                 initValue: Float,   // 0.1
                 lrate: Float, //0.001
                 k: Float
                ): Tensor[Float] = {

    val n = rating.size(1)
    val m = rating.size(2)

        // mnist 数据集
        var curImgSize = m

        // 若cpu核数较多，可以减小 sliceN._1
        val sliceSeg = new ArrayBuffer[(Int, Int, Int)]()
        // 依次加入每轮降维的参数，TODO 实验调优：
        sliceSeg += ((nf(0),nf(1), svdRound))  // 第一次降维：每28px降维至2px，迭代10次，即784px->56px，并行度28
        sliceSeg += ((nf(2), nf(3), svdRound))  // 56px -> 56/7=8px，并行度8
        sliceSeg += ((nf(4), nf(5), svdRound))  // 8px -> 5px，并行度1
        // TODO: 最后一次降维设置为 (7,5,120) 时，validate的Loss会更低一点，设为(8,5,120)时Loss会高

        val newRating = rating + 0  // 避免引用传递
        var pixelArr = new ArrayBuffer[Tensor[Float]]()

        for(sliceN <- 0 until sliceSeg.length){
          require(curImgSize % sliceSeg(sliceN)._1 == 0, s"像素无法被平均分割：slice${sliceN}")

          pixelArr.clear()  // 这里要clear，否则会残留上一次循环中降维后的结果
          // 所有图片切成28px的段：
          for(i <- 0 until curImgSize/sliceSeg(sliceN)._1){  // 循环28次
            //println(s"i:${i}, curImgSize:${curImgSize}, sliceN:${sliceN}, sliceSeg(sliceN)._1:${sliceSeg(sliceN)._1}, newRating.narrow:${newRating.narrow(2, i*sliceSeg(sliceN)._1 + 1, sliceSeg(sliceN)._1)}")
            pixelArr += newRating.narrow(2, i*sliceSeg(sliceN)._1 + 1, sliceSeg(sliceN)._1)
          }
          // 现在 pixelArr 中每个元素为所有图片的28像素tensor，称之为每个图片的像素段tensor（seg)
          val pixelRDD = sc.makeRDD(pixelArr.toArray)
          pixelArr.clear()
          // 并行降维：
          pixelArr ++= pixelRDD
            .map((segTensor) => {
              val svd = new IncreSVD(segTensor, sliceSeg(sliceN)._2, sliceSeg(sliceN)._3, initValue, lrate, k)
              svd.calcFeatures()
              svd.userFeature.t()
            })
            .collect()
          pixelRDD.unpersist()
          newRating.fill(0)  // 也可以不写这句，反正下面都会覆盖掉原值
          for(i <- 0 until curImgSize/sliceSeg(sliceN)._1){
            // 在 newRating 的第2维上(0,2)，即把第1、2列（的1~n行）更新为pixelArr(0)：
            newRating.narrow(2,i*sliceSeg(sliceN)._2 + 1,sliceSeg(sliceN)._2).update(T(T(1,n)), pixelArr(i));
          }

          curImgSize = (curImgSize/sliceSeg(sliceN)._1) * sliceSeg(sliceN)._2 // 第sliceN轮降维后的单张图片像素数
        }
        return newRating.narrow(2, 1, sliceSeg(sliceSeg.length-1)._2)

  }

}
