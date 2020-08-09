package cn.wangxw.aggpoint

import java.awt.image.BufferedImage
import java.io.{File, ObjectOutputStream, PrintWriter}
import java.nio.ByteBuffer

import breeze.linalg.{DenseMatrix, cov, svd}
import breeze.linalg.svd.SVD
import cn.wangxw.ZF.{IncreSVD, ZFBLAS, ZFHashLayer}
import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample}
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToSample, BytesToBGRImg, BytesToGreyImg, GreyImgNormalizer, GreyImgToSample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.T
import javax.imageio.ImageIO
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.clustering.{BisectingKMeans, GaussianMixture, KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
// import com.github.fommil.netlib.NativeSystemLAPACK

object AggPoint extends Serializable {

  @inline def byte2Float(byte: Byte): Float = {
    // 虽然我不知道下面和 byte.toFloat 有啥区别，但bigdl库的 BytesToGreyImg 是这么写的：
    byte & 0xff
  }

  /**
   * 将 Spark Mllib 的 Vector 转为 Sample 中的 feature tensor；
      Sample 的 feature tensor 应该是 3*height*width (3*frameLength) 的，
      即对于 feature.storage().array() 而言（由 BGR 转为 RGB 排列）：
        arr(offset + j) = content(j * 3 + 2)
        arr(offset + j + frameLength) = content(j * 3 + 1)
        arr(offset + j + frameLength * 2) = content(j * 3)
        content 是 LabeledBGRImg's Array[Float] / Array[Byte] 的顺序

   * @param prev
   * @param model
   * @param toRGB
   * @return
   */
  def toFeature(prev: Vector, model: String = "alexnet", toRGB: Boolean = true): Tensor[Float] = {
    model match {
      case "alexnet" => {
        val channel = 3
        val height = 32
        val width = 32
        val frameLength = height * width
        require(channel * frameLength == prev.size, "size not equal")
        val feature = Tensor[Float](channel, height, width)
        if(toRGB){
          for(i <- 0 until frameLength){
            feature.storage().array()(i) = prev(i*3+2).toFloat
            feature.storage().array()(i+frameLength) = prev(i*3+1).toFloat
            feature.storage().array()(i+frameLength*2) = prev(i*3).toFloat
          }
        }
        else {
          for(i <- 0 until frameLength){
            feature.storage().array()(i) = prev(i*3).toFloat
            feature.storage().array()(i+frameLength) = prev(i*3+1).toFloat
            feature.storage().array()(i+frameLength*2) = prev(i*3+2).toFloat
          }
        }
        feature
      }
      case "lenet" => {
        val featureBuffer = Tensor[Float]()
        val featureSize = new Array[Int](2)
        featureSize(0) = 28
        featureSize(1) = 28
        featureBuffer.set(Storage(prev.toArray.map(_.toFloat)), sizes = featureSize)
        featureBuffer
      }
      case _ => {
        throw new Exception("model name unknown")
      }
    }
  }

  /**
   * 由 1 维的 tensor 转为 Sample feature 的 3 维 tensor
   * @param prev
   * @param model
   * @param toRGB
   * @return
   */
  def toFeature(prev: Tensor[Float], model: String, toRGB: Boolean): Tensor[Float] = {
    model match {
      case "alexnet" => {
        val channel = 3
        val height = 32
        val width = 32
        val frameLength = height * width
        require(channel * frameLength == prev.nElement(), "size not equal")
        val feature = Tensor[Float](channel, height, width)
        if(toRGB){
          for(i <- 0 until frameLength){
            feature.storage().array()(i) = prev.valueAt(i*3+2 + 1)
            feature.storage().array()(i+frameLength) = prev.valueAt(i*3+1 + 1)
            feature.storage().array()(i+frameLength*2) = prev.valueAt(i*3 + 1)
          }
        }
        else {
          for(i <- 0 until frameLength){
            feature.storage().array()(i) = prev.valueAt(i*3 + 1)
            feature.storage().array()(i+frameLength) = prev.valueAt(i*3+1 + 1)
            feature.storage().array()(i+frameLength*2) = prev.valueAt(i*3+2 + 1)
          }
        }
        feature
      }
      case "lenet" => {
        // lenet 无需像alexnet一样适用三维feature
        val height = 28
        val width = 28
        val frameLength = height * width
        require(frameLength == prev.nElement(), "size not equal")
        val feature = Tensor[Float](height, width)
        for(i <- 0 until frameLength){
          feature.setValue(i / width + 1,i % width + 1,prev.valueAt(i+1))
        }
        feature
      }
      case _ => {
        throw new Exception("model name unknown")
      }
    }
  }

  /**
   * 将 ByteRecord 类型的图像数组 转换到一个大的 Tensor 中
   * @param prev
   * @return
   */
  def byteRecords2Tensor(prev: Array[ByteRecord], byteBufferOffset: Int): Tensor[Float] = {

    val imgNum = prev.length
    val imgSize = prev(0).data.size - byteBufferOffset // 每张图片的像素点数量
    val allImgTensor = Tensor[Float](imgNum, imgSize);

    var cnt = 1;
    prev.foreach((br) => {
      // 将 ByteRecord 的 data 数组转为 tensor：
      val t = Tensor[Float](imgSize)
      var i = 0;

      br.data.slice(byteBufferOffset, br.data.length).foreach((byte) => {
        i += 1
        t.setValue(i, byte2Float(byte))
      })
      require(i == imgSize, "图片像素点数量错误");

      allImgTensor.update(cnt, t) // 将 allImgTensor 的第 cnt 行替换为实际的 img tensor
      cnt += 1
    })
    allImgTensor
  }

  /**
   * 将 ByteRecord 类型的图像数组 转换到一个大的 Tensor 中
   * @param prev
   * @return
   */
  def byteRecords2Tensor(
                          prev: Array[ByteRecord],
                          byteBufferOffset: Int,
                          means: (Double, Double, Double),
                          std: (Double, Double, Double),
                          normalize: Double = 255
                        ): Tensor[Float] = {

    val imgNum = prev.length
    val imgSize = prev(0).data.size - byteBufferOffset // 每张图片的像素点数量
    val allImgTensor = Tensor[Float](imgNum, imgSize);

    if(byteBufferOffset == 8){
      require(imgSize%3==0, "imgSize % 3 != 0")
    }

    var cnt = 1;
    prev.foreach((br) => {
      // 将 ByteRecord 的 data 数组转为 tensor：
      val t = Tensor[Float](imgSize)
      var i = 0;

      br.data.slice(byteBufferOffset, br.data.length).foreach((byte) => {
        val d = byte2Float(byte).toDouble   // 保持 Double 精度，避免误差太大
        // 因为对于单通道的mnist数据集，means和stds是元素值相同的三元素元组，
        // 因此仍适用以下逻辑
        if(i % 3 == 0){
          t.setValue(
            i+1,
            ((d / normalize - means._3) / std._3).toFloat
          )
        }
        else if(i % 3 == 1){
          t.setValue(
            i+1,
            ((d / normalize - means._2) / std._2).toFloat
          )
        }
        else if(i % 3 == 2){
          t.setValue(
            i+1,
            ((d / normalize - means._1) / std._1).toFloat
          )
        }

        i += 1
      })
      require(i == imgSize, "图片像素点数量错误");

      allImgTensor.update(cnt, t) // 将 allImgTensor 的第 cnt 行替换为实际的 img tensor
      cnt += 1
    })
    allImgTensor
  }

  /**
   * 把单张图片的 tensor 转为 byte 数组；
   * 要注意该 tensor 应该是未归一化的，否则转为 byte 后的值都是 0；
   * @param prev
   * @param byteBufferOffset
   * @param model
   * @return
   */
  def tensor2bytes(prev: Tensor[Float], byteBufferOffset: Int, model: String = "lenet"): Array[Byte] = {
    val imgSize = prev.size(prev.nDimension())
    model match {
      case "lenet" => {
        val aggByteRecord = new Array[Byte](imgSize + byteBufferOffset)
        for (i <- 1 to prev.nElement()) {
          aggByteRecord(i - 1 + byteBufferOffset) = prev.valueAt(i).toByte
        }
        aggByteRecord
      }
      case "alexnet" => {
        val aggByteRecord = new Array[Byte](imgSize + byteBufferOffset)
        ByteBuffer.wrap(aggByteRecord).putInt(32).putInt(32)
        for (i <- 1 to prev.nElement()) {
          aggByteRecord(i - 1 + byteBufferOffset) = prev.valueAt(i).toByte
        }
        aggByteRecord
      }
    }

  }

  /**
   * 递归分桶，被 lsh 方法调用
   * @param imgArr
   * @param ans
   * @param thisPixel
   * @param vIndex
   */
  def splitU(imgArr: Array[(Tensor[Float], Int)],
             ans: mutable.HashMap[Int, StringBuilder],
             thisPixel: Int,
             vIndex: Array[Int]
            ): Unit = {
    if (thisPixel > imgArr(0)._1.size(2)) {
      return;
    }
    var pixelArr = ArrayBuffer.empty[(Int, Float)] // Int是图片ID，Float是该图片在第thisPixel上的值
    for (i <- vIndex) {
      pixelArr += Tuple2(i, imgArr(i)._1.valueAt(1, thisPixel))
    }
    pixelArr = pixelArr.sortWith((img1, img2) => {
      img1._2 < img2._2 // 根据图片在第thisPixel上的值排序
    })
    val halfSize = pixelArr.length / 2
    for (i <- 0 until halfSize) {
      val imgId = imgArr(pixelArr(i)._1)._2
      if (ans.get(imgId).isDefined) {
        ans(imgId) ++= "0"
      }
      else {
        ans.update(imgId, new mutable.StringBuilder("0"))
      }
    }
    for (i <- halfSize until pixelArr.length) {
      val imgId = imgArr(pixelArr(i)._1)._2
      if (ans.get(imgId).isDefined) {
        ans(imgId) ++= "1"
      }
      else {
        ans.update(imgId, new mutable.StringBuilder("1"))
      }
    }
    val index0 = pixelArr.slice(0, halfSize).map(_._1).toArray
    val index1 = pixelArr.slice(halfSize, pixelArr.length).map(_._1).toArray
    splitU(imgArr, ans, thisPixel + 1, index0)
    splitU(imgArr, ans, thisPixel + 1, index1)
  }

  /**
   * 参考 ZFHashLayer$hashOnce，对 imgArr 中图片进行分桶，依次根据图片的各个像素值，
   * 注意：imgId的处理有点复杂
   * @param imgArr   数组，元素为元组，元组的 _1 为图片的 1*n 的 tensor，_2 为图片 ID
   * @param mode   1：ZF版实现，2：我自己的实现
   * @return
   */
  def lsh(imgArr: Array[(Tensor[Float], Int)], mode: Int = 1): Array[Array[Int]] = {
    val nPixel = imgArr(0)._1.size(2)

    if (mode == 1) {
      val ans = new mutable.HashMap[Int, StringBuilder]()
      splitU(imgArr, ans, 1, Array.range(0, imgArr.length))
      val bucketMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
      ans.foreach((tup) => {
        val imgId = tup._1
        val imgHash = tup._2.toString()
        //println(s"imgId:${imgId} imgHash:${imgHash}")

        if (bucketMap.get(imgHash).isDefined) {
          bucketMap(imgHash) += imgId
        }
        else {
          bucketMap.update(imgHash, ArrayBuffer(imgId))
        }
      })
      bucketMap.map((tup) => {
        tup._2.toArray
      }).toArray

    }
    else {
      // 之前的实现，直接对图片排序
      // 复杂度：O( (nlogn)+() )
      var newImgBuckets = ArrayBuffer.empty[ArrayBuffer[(Tensor[Float], Int)]]
      val tmpImgBuckets = new ArrayBuffer[ArrayBuffer[(Tensor[Float], Int)]]()

      newImgBuckets += ArrayBuffer(imgArr: _*)

      val sliceNum = 2; // 对每个属性排序后分几桶

      //println("开始分桶...")

      for (time <- 1 to nPixel) {
        //println(s"第 ${time}/${nPixel} 次分桶...")
        // 第 time 次排序，取出上次分好的各个桶，各自继续排序分桶：
        newImgBuckets.foreach((oldBucket) => {
          if (oldBucket.length >= sliceNum) {
            // 对每一桶按照第 time 个像素排序：
            val tmpBucket = oldBucket.sortWith((t1, t2) => {
              // 因为虽然一张图片（1 个tensor）是一行，但该 tensor 仍是二维，只不过第一维size为1
              t1._1.valueAt(1, time) < t2._1.valueAt(1, time)
            })

            // 排序之后分桶：
            val num = math.floor(tmpBucket.length / sliceNum).toInt; // 除了最后一桶，每桶数量
            for (buc <- 0 until sliceNum) {
              if (buc < sliceNum - 1) {
                tmpImgBuckets += tmpBucket.slice(buc * num, (buc + 1) * num)
              } else {
                // 最后一桶分完所有剩余（因为可能存在除不尽的情况）：
                tmpImgBuckets += tmpBucket.slice(buc * num, tmpBucket.length)
              }
            }
          }
          else if (oldBucket.length > 0) {
            // 此时该桶（arr）已经不够分为 sliceNum 桶了，直接仍归为一桶：
            tmpImgBuckets += oldBucket
          }
        })
        newImgBuckets.clear()
        newImgBuckets ++= tmpImgBuckets // 不能直接赋值，会导致引用传递
        tmpImgBuckets.clear()
      }

      newImgBuckets.map((arr) => {
        arr.map(tup => {
          tup._2
        }).toArray
      }).toArray
    }
  }

  /**
   * 直接用 Spark Mllib 的 SVD 进行降维，对小的矩阵（tensor）超慢，
   * 若输入的 thisImgTensor 是一维的，将会根据 imgWidth 和 imgHei 把该tensor分解为
   * 为二维图片；若输入的是二维的大tensor（正常适用情况），将不再分割。
   * @param thisImgTensor 图像tensor，一维的
   * @param imgWidth
   * @param imgHei
   * @param nf
   * @param sc
   * @return
   */
  def svdDistri(thisImgTensor: Tensor[Float], imgWidth: Int, imgHei: Int, nf: Int, sc: SparkContext)
  : (Tensor[Float], Tensor[Float], Tensor[Float]) = {

    val vectorArr = ArrayBuffer[Vector]()
    if (thisImgTensor.nDimension() == 1) {
      // 该图像为一维 tensor：
      for (i <- 0 until imgHei) {
        vectorArr += Vectors.dense(
          thisImgTensor.narrow(1, i * imgWidth + 1, imgWidth).toArray().map(_.toDouble)
        )
      }
    }
    else if (thisImgTensor.size(1) == 1) {
      // 该图像为 1*n 的一维 tensor：
      val newThisImgTensor = (thisImgTensor - 0).resize(thisImgTensor.size(2))

      for (i <- 0 until imgHei) {
        vectorArr += Vectors.dense(
          newThisImgTensor.narrow(1, i * imgWidth + 1, imgWidth).toArray().map(_.toDouble)
        )
      }
    }
    else if (thisImgTensor.nDimension() == 2) {
      // 该图像为二维
      for (i <- 0 until imgHei) {
        vectorArr += Vectors.dense(
          (thisImgTensor.narrow(1, i + 1, 1) - 0).resize(thisImgTensor.size(2)).toArray().map(_.toDouble)
        )
      }
    }
    else {
      throw new Exception("SVD: thisImgTensor is of unknown type.")
    }

    val allImgVector = sc.parallelize(vectorArr)
    val mat: RowMatrix = new RowMatrix(allImgVector)
    // Compute the top 5 singular values and corresponding singular vectors.
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(nf, computeU = true)
    val U: RowMatrix = svd.U // The U factor is a RowMatrix.
    val s: Vector = svd.s // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V // The V factor is a local dense matrix.

    // 对单张图像（28*28）时：
    // nf=5时，第72张图片的U的第二维是4，而不是5，导致数组越界报错
    // nf=6时，第72张图片的U的第二维是4，而不是5，导致数组越界报错
    // nf=7时，第8张图片的U的第二维是6，而不是7，导致数组越界报错
    // nf=8时，第8张图片的U的第二维是6，而不是8，导致数组越界报错
    // 因此，设 k 为 nf 和 第二维size 的较小值

    val collect = U.rows.collect()
    //println(s"idCnt: ${idCnt}")
    //println(s"U row vector size: ${collect(0).size}")
    //println(s"V col: ${V.numCols}")
    //println(s"U:${collect.foreach(println)}")
    //println(s"s:${s}")
    //println(s"V:${V}")

    val k = math.min(nf, V.numCols)
    if (k != nf) {
      println(s"WARN: nf(${nf}) is not equal to V.numCols(${V.numCols})")
    }

    val uTensor = Tensor[Float](imgHei, k)
    for (i <- 1 to collect.length) {
      val arr = collect(i - 1).toArray
      for (j <- 1 to k) {
        // nf=5时arr(j-1)会报错，数组越界4
        uTensor.setValue(i, j, arr(j - 1).toFloat)
      }
    }
    val sTensor = Tensor[Float](k, k)
    for (i <- 1 to k) {
      sTensor.setValue(i, i, s(i - 1).toFloat)
    }
    val vTensor = Tensor[Float](k, imgWidth)
    for (i <- 1 to k) {
      for (j <- 1 to imgWidth) {
        vTensor.setValue(i, j, V(j - 1, i - 1).toFloat)
      }
    }
    require(uTensor.size(2) == sTensor.size(1))
    require(sTensor.size(2) == vTensor.size(1))
    (uTensor, sTensor, vTensor)
  }

  def rgb2gray(r: AnyVal, g: AnyVal, b: AnyVal): Float = {
    (0.3 * r.asInstanceOf[Float] + 0.59 * g.asInstanceOf[Float] + 0.11 * b.asInstanceOf[Float]).toFloat
  }

  /**
   * RGB图像需要转为灰度图像，算法为 Gray=R*0.3+G*0.59+B*0.11，
   * 参考：https://baike.baidu.com/item/%E7%81%B0%E5%BA%A6%E5%9B%BE%E5%83%8F
   * @param imgBytes
   * @param imgHeight
   * @param imgWidth
   * @param channel
   * @param toHeight
   * @param toWidth
   * @param toGreyDegree
   * @return
   */
  def aHash(
             imgBytes: Array[Byte],
             imgHeight: Int = 28,
             imgWidth: Int = 28,
             channel: Int = 1,
             toHeight: Int = 8,
             toWidth: Int = 8,
             toGreyDegree: Int = 64): Long = {

    // 转为 OpenCVMat，格式，并resize图像：
    val matImg = OpenCVMat.fromPixelsBytes(imgBytes, imgHeight, imgWidth, channel)
    val (bytes, _, _, _) = OpenCVMat.toBytePixels(
      Resize.transform(matImg, matImg, toWidth, toHeight),
      new Array[Byte](toWidth * toHeight)
    )
    require(bytes.length == toWidth*toHeight*channel, "OpenCVMat 降维后数组长度错误")

    // 转为64阶灰度，并计算平均值
    var byteSum = 0f;
    val fArr = ArrayBuffer.empty[Float]
    channel match {
      case 1 => {
        // 通道为1，是mnist数据集
        bytes.foreach(b => {
          val f = AggPoint.byte2Float(b) / 255f * toGreyDegree.toFloat // 转为64阶（灰度）
          byteSum += f
          fArr += f
        })
      }
      case 3 => {
        for (i <- 0 until toHeight; j <- 0 until toWidth) {
          val idx = (i * toWidth + j) * 3 // 从idx开始的3个byte分别为B、G、R
          var gray = rgb2gray(
            AggPoint.byte2Float(bytes(idx + 2)),
            AggPoint.byte2Float(bytes(idx + 1)),
            AggPoint.byte2Float(bytes(idx))
          )
          gray = gray / 255f * toGreyDegree.toFloat
          byteSum += gray
          fArr += gray
        }
      }
    }

    val resLen = fArr.length
    require(resLen == toWidth * toHeight, s"fArr.lenght != ${toWidth}*${toHeight}")
    byteSum = byteSum / resLen.toFloat

    // 计算哈希指纹（64位长整型）：
    var hashVal = 0L
    for (i <- 0 until resLen - 1) {
      if (fArr(i) >= byteSum) {
        hashVal = hashVal | (1 << (resLen - 1 - i))
      }
    }
    hashVal
  }

  /**
   *
   * @param imgTensor 1*imgSize 的单张图像 tensor
   * @param imgHeight
   * @param imgWidth
   * @param channel
   * @param toHeight
   * @param toWidth
   * @param toGreyDegree
   * @return
   */
  def pHash(
             imgTensor: Tensor[Float],
             imgHeight: Int = 28,
             imgWidth: Int = 28,
             channel: Int = 1,
             toHeight: Int = 8,
             toWidth: Int = 8,
             toGreyDegree: Int = 256): Long = {
    // 因为 DCT 可以直接对 8*8 或者 32*32 的图像大小进行计算，所以无需 resize 了


    // 转为灰度
    var intArr = channel match {
      case 1 => {
        (imgTensor - 0).resize(imgHeight * imgWidth).toArray().map(_.toInt)
      }
      case 3 => {
        val arr = new Array[Int](imgHeight * imgWidth)
        for (i <- 0 until imgHeight; j <- 0 until imgWidth) {
          val idx = (i * imgWidth + j) * 3 + 1 // 从idx开始的3个byte分别为B、G、R
          var gray = rgb2gray(
            imgTensor.valueAt(1,idx + 2),
            imgTensor.valueAt(1,idx + 1),
            imgTensor.valueAt(1,idx)
          )
          gray = gray / 255f * toGreyDegree.toFloat
          arr(idx / 3) = gray.toInt
        }
        arr
      }
    }

    val dct = new DCT
    intArr = dct.DCT(intArr, imgWidth)

    // 保留左上角8*8
    val remain = new Array[Int](toHeight * toWidth)
    var sum = 0f
    for (i <- 0 until toHeight) {
      for (j <- 0 until toWidth) {
        val idx = i * toWidth + j
        remain(idx) = intArr(i * imgWidth + j)
        sum += remain(idx).toFloat
      }
    }
    sum = sum / (toHeight * toWidth).toFloat

    // 计算 hash 指纹：
    var hashVal = 0L
    for (i <- 0 until remain.length) {
      if (remain(i).toFloat >= sum) {
        hashVal = hashVal | (1 << (remain.length - 1 - i))
      }
    }
    hashVal
  }

  def dHash(
             imgBytes: Array[Byte],
             imgHeight: Int = 28,
             imgWidth: Int = 28,
             channel: Int = 1,
             toHeight: Int = 8,
             toWidth: Int = 8,
             toGreyDegree: Int = 256): Long = {

    // 转为 OpenCVMat，格式，并resize图像：
    val matImg = OpenCVMat.fromPixelsBytes(imgBytes, imgHeight, imgWidth, channel)
    val (bytes, _, _, _) = OpenCVMat.toBytePixels(
      Resize.transform(matImg, matImg, (toWidth + 1), toHeight),
      new Array[Byte]((toWidth + 1) * toHeight)
    )
    require(bytes.length == (toWidth+1)*toHeight*channel, "OpenCVMat 数组长度错误")

    // 转为灰度：
    val imgArr = channel match {
      case 1 => {
        bytes.map(AggPoint.byte2Float)
      }
      case 3 => {
        val arr = new Array[Float](toHeight * (toWidth+1))
        for (i <- 0 until toHeight; j <- 0 until (toWidth+1)) {
          val idx = (i * toWidth + j) * 3 // 从idx开始的3个byte分别为B、G、R
          var gray = rgb2gray(
            AggPoint.byte2Float(bytes(idx + 2)),
            AggPoint.byte2Float(bytes(idx + 1)),
            AggPoint.byte2Float(bytes(idx))
          )
          gray = gray / 255f * toGreyDegree.toFloat
          arr(idx / 3) = gray
        }
        arr
      }
    }
    // 计算哈希指纹（64位长整型）：

    var hashVal = 0L

    for (i <- 0 until toHeight; j <- 1 to toWidth) {
      val idx = i * (toWidth + 1) + j
      if (imgArr(idx - 1) > imgArr(idx)) {
        val cnt = idx - (i + 1)
        hashVal = hashVal | (1 << (64 - 1 - cnt))
      }
    }
    hashVal
  }

  /**
   * 计算两个 Long 之间的汉明距离，即 a 和 b 的不相同的位数
   * @param a
   * @param b
   * @return 0最近，1最远
   */
  def hammingDistance(a: Long, b: Long): Float = {
    val c = a ^ b;
    var diffCnt = 0f
    for (i <- 0 until 64) {
      if (((c >> i) & 1L) == 1) {
        // 第 i 位为 1，说明对应位的 a 和 b 异或为 1，说明不同，则距离 +1
        diffCnt += 1
      }
    }
    diffCnt / 64
  }

  /**
   * 巴氏系数，可以配合直方图使用；参考：https://github.com/nivance/image-similarity
   * @param a
   * @param b
   * @return
   */
  def bhattacharyya(a: Tensor[Float], b: Tensor[Float]): Float = {
    val aTensor = if (a.nDimension() > 1) (a - 0).resize(a.size(2)) else (a - 0)
    val bTensor = if (b.nDimension() > 1) (b - 0).resize(b.size(2)) else (b - 0)
    require(aTensor.nElement() == bTensor.nElement())
    var ans = 0f;
    for (i <- 1 to aTensor.size(1)) {
      ans += math.sqrt((aTensor.valueAt(i) * bTensor.valueAt(i)).toDouble).toFloat
    }
    ans
  }

  /**
   * ZF版余弦相似度（应该是吧）
   * @param v1
   * @param v2
   * @return [-1,1]，越大越相似
   */
  def zfCosine(v1: Vector, v2: Vector): Double = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and Dim(v2)" +
      s"=${v2.size}.")
    //    var squaredDistance = 0.0
    var numerator = 0.0
    var denominatorA = 0.0
    var denominatorB = 0.0
    var ans = 0.0

    (v1, v2) match {
      case (v1: SparseVector, v2: SparseVector) =>
        val v1Values = v1.values
        val v1Indices = v1.indices
        val v2Values = v2.values
        val v2Indices = v2.indices
        val nnzv1 = v1Indices.length
        val nnzv2 = v2Indices.length

        var kv1 = 0
        var kv2 = 0
        while (kv1 < nnzv1 || kv2 < nnzv2) {
          //          var score = 0.0
          var adot = 0.0

          if (kv2 >= nnzv2 || (kv1 < nnzv1 && v1Indices(kv1) < v2Indices(kv2))) {
            //            score = v1Values(kv1)
            denominatorA += math.pow(v1Values(kv1), 2)
            kv1 += 1
          } else if (kv1 >= nnzv1 || (kv2 < nnzv2 && v2Indices(kv2) < v1Indices(kv1))) {
            //            score = v2Values(kv2)
            denominatorB += math.pow(v2Values(kv2), 2)
            kv2 += 1
          } else {
            //            score = v1Values(kv1) - v2Values(kv2)
            denominatorA += math.pow(v1Values(kv1), 2)
            denominatorB += math.pow(v2Values(kv2), 2)
            adot = v1Values(kv1) * v2Values(kv2)
            kv1 += 1
            kv2 += 1
          }
          numerator += adot
          //          squaredDistance += score * score
        }
        ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))

      case (v1: SparseVector, v2: DenseVector) =>
        ans = zfCosine(v1, v2)

      case (v1: DenseVector, v2: SparseVector) =>
        ans = zfCosine(v2, v1)

      case (DenseVector(vv1), DenseVector(vv2)) =>
        var kv = 0
        val sz = vv1.length
        while (kv < sz) {
          //          val score = vv1(kv) - vv2(kv)
          //          squaredDistance += score * score
          denominatorA += math.pow(vv1(kv), 2)
          denominatorB += math.pow(vv2(kv), 2)
          numerator += vv1(kv) * vv2(kv)
          kv += 1
        }
        ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))
      case _ =>
        throw new IllegalArgumentException("Do not support vector type " + v1.getClass +
          " and " + v2.getClass)
    }
    ans
  }

  /**
   * Returns the squared distance between DenseVector and SparseVector.
   */
  def zfCosine(v1: SparseVector, v2: DenseVector): Double = {
    var kv1 = 0
    var kv2 = 0
    val indices = v1.indices
    //    var squaredDistance = 0.0
    var numerator = 0.0
    var denominatorA = 0.0
    var denominatorB = 0.0
    var ans = 0.0

    val nnzv1 = indices.length
    val nnzv2 = v2.size
    var iv1 = if (nnzv1 > 0) indices(kv1) else -1

    while (kv2 < nnzv2) {
      //      var score = 0.0
      var adot = 0.0
      if (kv2 != iv1) {
        //        score = v2(kv2)
        denominatorB += v2(kv2) * v2(kv2)
      } else {
        //        score = v1.values(kv1) - v2(kv2)
        denominatorA += v1.values(kv1) * v1.values(kv1)
        denominatorB += v2(kv2) * v2(kv2)
        adot = v1.values(kv1) * v2(kv2)
        if (kv1 < nnzv1 - 1) {
          kv1 += 1
          iv1 = indices(kv1)
        }
      }
      //      squaredDistance += score * score
      numerator += adot
      kv2 += 1
    }
    //    squaredDistance
    ans = numerator / (math.pow(denominatorA, 0.5) * math.pow(denominatorB, 0.5))
    ans
  }

  /**
   * 递归的删除 dir 目录及其内所有文件
   * @param dir
   * @return
   */
  def deleteDir(dir: File): Boolean = {
    if (dir.isDirectory) {
      val children = dir.list();
      //递归删除目录中的子目录下
      for (i <- 0 until children.length) {
        val success = deleteDir(new File(dir, children(i)));
        if (!success) {
          return false;
        }
      }
    }
    // 目录此时为空，可以删除
    return dir.delete();
  }

  /**
   * 保存一张图片
   * @param model
   * @param byteArr
   * @param label
   * @param path
   */
  def saveImg(model: String, byteArr: Array[Byte], label: Float, path: String, _imgHeight: Int = 0, _imgWidth: Int =
  0): Unit = {
    //println(s"第 ${bucketCnt} 桶：原始图像 ${samplesBGR.length} 张，压缩点 ${aggPointBGR.length} 张")
    val byteBufferOffset = 8
    var imgHeight = _imgHeight
    var imgWidth = _imgWidth
    model match {
      case "lenet" => {
        if (imgHeight == 0) {
          imgHeight = 28
        }
        if (imgWidth == 0) {
          imgWidth = 28
        }

        val newByteArr = new Array[Byte](byteArr.length * 3 + byteBufferOffset)
        ByteBuffer.wrap(newByteArr).putInt(imgWidth).putInt(imgHeight)
        for (i <- 0 until byteArr.length) {
          newByteArr(byteBufferOffset + i * 3) = byteArr(i)
          newByteArr(byteBufferOffset + i * 3 + 1) = byteArr(i)
          newByteArr(byteBufferOffset + i * 3 + 2) = byteArr(i)
        }
        val br = ByteRecord(newByteArr, label)
        val sampleBGR = new BytesToBGRImg(normalize = 1f, imgWidth, imgHeight).apply(
          Iterator(br)
        ).toArray
        sampleBGR(0).save(path, 1)
      }
      case "alexnet" => {
        if (imgHeight == 0) {
          imgHeight = 32
        }
        if (imgWidth == 0) {
          imgWidth = 32
        }
        val prefixValue = ByteBuffer.wrap(byteArr).getInt()
        if (prefixValue != 32) {
          // 说明bytes的前8个字节不是32、32，则根据参数中的宽、高赋值
          val newByteArr = new Array[Byte](byteArr.length + 8)
          ByteBuffer.wrap(newByteArr).putInt(imgWidth).putInt(imgHeight)
          byteArr.copyToArray(newByteArr, 8)

          val aggPointBGR = new BytesToBGRImg(normalize = 1f, imgWidth, imgHeight).apply(
            Iterator(ByteRecord(newByteArr, label))
          ).toArray
          aggPointBGR(0).save(path, 1)
        }
        else {
          val aggPointBGR = new BytesToBGRImg(normalize = 1f, imgWidth, imgHeight).apply(
            Iterator(ByteRecord(byteArr, label))
          ).toArray
          aggPointBGR(0).save(path, 1)
        }
      }
    }
  }

  def saveImg(model: String, br: ByteRecord, path: String): Unit = {
    //println(s"第 ${bucketCnt} 桶：原始图像 ${samplesBGR.length} 张，压缩点 ${aggPointBGR.length} 张")
    val byteBufferOffset = 8
    model match {
      case "lenet" => {
        saveImg(model, br.data, br.label, path)
      }
      case "alexnet" => {
        //val aggPointBGR = new BytesToBGRImg(normalize = 1f, 32, 32).apply(
        //  Iterator(br)
        //).toArray
        //aggPointBGR(0).save(path, 1)
        saveImg(model, br.data, br.label, path)
      }
    }
  }

  /**
   * 将输入的 ByteRecord 数组，转换为最终的带压缩点的数据集合；
   * 该方法已弃用；
   * @param model   “lenet”或“alexnet”
   * @param imgs
   * @param byteBufferOffset
   * @param sc
   * @param svdComputeNum
   * @param svdNumPerBucket
   * @param svdNfStr
   * @param svdInitValue
   * @param svdLrate
   * @param svdK
   * @return
   */
  @deprecated
  def run(
           model: String,
           imgs: Array[ByteRecord],
           byteBufferOffset: Int = 0,
           sc: SparkContext,
           svdComputeNum: Int = 120,
           svdNumPerBucket: Int = 2,
           svdNfStr: Option[String] = Some("28,2,8,1,7,5"),
           svdInitValue: Double = 0.1f, // 0.1
           svdLrate: Double = 0.001f, //0.001
           svdK: Double = 0.015f,
           svdMode: Int = 2, // 1：ZF版抽样迭代；2：我的分布式迭代
           svdRatioN: Int = 10, // svdMode为1时，需要指定此参数
           lshMode: Int = 1 // 1：ZF版递归实现；2：我自己的循环实现
         ): Array[(Sample[Float], Seq[Sample[Float]])] = {

    // MiniBatch 通过 .getInput() 可获得 32*28*28 的 tensor，即 32 张 28*28 分辨率的图片数据
    // 因此压缩点的压缩应该在最后两维上进行
    // TODO：若进行对照测试时，应保证图像预处理方式一致
    val trainMean = 0.13066047740239506
    val trainStd = 0.3081078

    val testMean = 0.13251460696903547
    val testStd = 0.31048024

    val imgNum = imgs.length
    val imgSize = imgs(0).data.size - byteBufferOffset // 每张图片的像素点数量
    println(s"Current model:${model}, imgNum:${imgNum}, imgSize:${imgSize}")

    val allImgTensor = byteRecords2Tensor(imgs, byteBufferOffset);


    println("allImgTensor 加载完成，开始降维计算...")
    // 将 svdNfStr 字符串分割为降维各个阶段的参数：
    //print("NF:")
    val nfArr = svdNfStr.get.split(',').map((s) => {
      val thisNf = s.trim.toInt
      //print(" " + thisNf)
      thisNf
    }) // 将字符串以逗号分割为各个阶段的nf值
    //println("")
    val nPixel = nfArr(nfArr.length - 1); // 压缩到 nPixel 个像素

    /**
     * 开始 IncreSVD 降维
     * 输入：allImgTensor: Tensor[Float]
     * 输出：aggImtTensor: Tensor[Float]
     */
    if (svdMode == 2) {
      //println(s"分布式降维过程：原图片尺寸 ${imgSize} px：" +
      //  s"\n -> 每 ${nfArr(0)}px 变为 ${nfArr(1)}px，降至 ${imgSize / nfArr(0) * nfArr(1)} px" +
      //  s"\n -> 每 ${nfArr(2)}px 变为 ${nfArr(3)}px，降至 ${imgSize / nfArr(0) * nfArr(1) / nfArr(2) * nfArr(3)} px" +
      //  s"\n -> 每 ${nfArr(4)}px 变为 ${nfArr(5)}px，降至 ${
      //    imgSize / nfArr(0) * nfArr(1) / nfArr(2) * nfArr(3) / nfArr(4) * nfArr(5)
      //  } px");
    }
    else {
      //println("ZF版IncreSVD实现，抽样式迭代...")
    }
    val svdRound = svdComputeNum; // 迭代120次
    //println(s"每次降维迭代 ${svdRound} 次，最终降至 ${nfArr(5)}px，开始降维迭代计算（可能需要较长时间，请稍等）...")

    //return ArrayBuffer.empty[(Sample[Float], Seq[Sample[Float]])].toArray

    val svdStart = System.nanoTime()
    val aggImgTensor = svdMode match {
      case 1 => {
        IncreSVD.calcDistri(
          allImgTensor,
          nfArr.last,
          svdRound,
          sc,
          svdInitValue.toFloat,
          svdLrate.toFloat,
          svdK.toFloat,
          svdRatioN
        )
      }
      case 2 => {
        IncreSVD.calcDistri(
          allImgTensor,
          nfArr,
          svdRound,
          sc,
          svdInitValue.toFloat,
          svdLrate.toFloat,
          svdK.toFloat
        )
      }
    }
    //println(s"svd 降维计算花费：${(System.nanoTime() - svdStart) / 1e9}s")

    println("降维计算完成，开始LSH...")
    // 此时，aggImgTensor 的每一行为一张压缩过的图片数据

    require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")

    // Int相当于图片ID，以下排序均为带ID排序，便于最终根据ID与原图片数据imgs一一对应：
    var aggImgArr = new ArrayBuffer[(Tensor[Float], Float, Int)]()

    for (i <- 1 to imgNum) {
      aggImgArr += Tuple3(aggImgTensor.narrow(1, i, 1), imgs(i - 1).label, i) // 将每行数据（一张图片）放入数组，便于之后LSH的排序分堆
    }

    /*
    小数据输出测试
    var buCnt = 0
    lsh(aggImgArr.slice(0,12).map((tup) => {
      println(s"${tup._3}:\t${tup._1}")
      Tuple2(tup._1, tup._3)
    }).toArray).foreach(arr=>{
      buCnt += 1
      println(s"第 ${buCnt} 桶:\t${arr.mkString("\t")}")
    })
     */

    //var bucketCnt1 = 0
    val bucketArr = sc.parallelize(aggImgArr).groupBy((tup) => {
      // 根据label将RDD分组
      tup._2
    })
      .map((tupleIter) => {
        // 对于label相同的图片（tupleIter._2）进行分桶：

        val tmpStartTime = System.nanoTime()
        val imgArr = tupleIter._2.map((tup) => {
          Tuple2(tup._1, tup._3)
        }).toArray
        val lshResult = lsh(imgArr, 1)

        println(s"label:${tupleIter._1}, lsh cost:${(System.nanoTime() - tmpStartTime) / 1e9}s")
        lshResult
      })
      .flatMap((buckets) => {
        // 将label相同的多个桶组成的数组newImgArr展平：
        buckets
      })
      .map((bucket) => {
        // 对每个桶求压缩点：
        // 每个桶为一个 ArrayBuffer[ Tensor, Int ]，Tensor为压缩后的图片数据，Int为原图片ID
        var aggFeature = Tensor[Float](imgSize).fill(0)
        var aggLabel = 0f;
        val originPoints = new ArrayBuffer[(ByteRecord, Int)]() // Int 为原图ID，从0开始

        //bucketCnt1 += 1
        bucket.foreach((imgId) => {
          // 这里所有的 img 应该计算出一个压缩点：
          val originPoint = imgs(imgId - 1) // id是按照tensor从下标1开始的，所以换为数组时需要-1
          aggLabel += originPoint.label
          aggFeature += allImgTensor.narrow(1, imgId, 1) // TODO 会不会超过Float范围
          originPoints += Tuple2(originPoint, imgId - 1)
        })
        // 计算压缩点（求平均值）：
        aggFeature /= bucket.length
        aggLabel /= bucket.length
        //require(aggLabel == tupleIter._1, "计算出的平均 label 与实际 label 不同")
        (aggFeature, aggLabel, originPoints.toArray)
      })
      .collect()


    println(s"共有 ${bucketArr.length} 个桶")
    var bucketCnt = 0

    bucketArr.map((tup) => {
      // TODO: 不知道为什么，这个 map 里的操作放在executor上就回报错：not Serializable: scala.collection.Iterator
      // TODO: 可能是因为 Transformer ？


      // 在 driver 上对压缩点和原始图像进行预处理：
      val aggFeature = tup._1
      val aggLabel = tup._2
      val originPoints = tup._3


      bucketCnt += 1
      //println(s"${bucketCnt} : ${originPoints(0).label} : ")
      //for(j <- 0 until 4){
      //  for(i <- 7 until 18){
      //    print(s"${originPoints(j).data(i) & 0xff}   ")
      //  }
      //  println("")
      //}

      model match {
        case "lenet" => {
          val bucketDir = "D:\\课件（研）\\bigdl\\datasets\\lenetResult"

          // 手动Transformer和自动的结果还是不太一样，弃用手动的：
          //val aggPointSample1 = Sample(
          //  (aggFeature.resize(28, 28) / 255f - trainMean.toFloat) / trainStd.toFloat,
          //  aggLabel
          //)

          val aggByteRecord = new Array[Byte](imgSize + byteBufferOffset)
          for (i <- 1 to aggFeature.nElement()) {
            aggByteRecord(i - 1 + byteBufferOffset) = aggFeature.valueAt(i).toByte
          }
          val aggPointBGR = new GreyImgNormalizer(trainMean, trainStd).apply( // 类型LabeledGreyImage不变，(value-mean)/std
            new BytesToGreyImg(28, 28).apply( // Array[Byte] 转换为 LabeledGreyImage:Array[Float],norm到0~1
              Iterator(ByteRecord(aggByteRecord, aggLabel))
            )
          ).toArray
          val aggPointSample = new GreyImgToSample().apply(aggPointBGR.toIterator).next()


          // 保存图片：
          //saveImg(model, aggByteRecord, aggLabel, s"${bucketDir}\\${bucketCnt}_agg_${aggLabel.toInt}.jpg")

          // 将压缩点对应的原始图片数据转换为Sample格式：
          val samples = new GreyImgToSample().apply(
            new GreyImgNormalizer(trainMean, trainStd).apply( // 类型LabeledGreyImage不变，(value-mean)/std
              new BytesToGreyImg(28, 28).apply( // Array[Byte] 转换为 LabeledGreyImage:Array[Float],norm到0~1
                originPoints.map(_._1).toIterator
              )
            )
          ).toArray // TODO: BytesToBGRImg.apply 有bug，BytesToGreyImg 就没bug？

          //originPoints.foreach(br => {
          //  saveImg(model, br._1, s"${bucketDir}\\${bucketCnt}_origin_${br._1.label.toInt}_${br._2}.jpg")
          //})

          // 将压缩点和对应的图片合为元组：
          Tuple2(aggPointSample, samples.toSeq)
        }
        case "alexnet" => {
          val bucketDir = "D:\\课件（研）\\bigdl\\datasets\\alexnetResult"

          val aggByteRecord = new Array[Byte](imgSize + byteBufferOffset)
          ByteBuffer.wrap(aggByteRecord).putInt(32).putInt(32)
          for (i <- 1 to aggFeature.nElement()) {
            aggByteRecord(i - 1 + byteBufferOffset) = aggFeature.valueAt(i).toByte
          }

          // 原本 BytesToBGRImg 会 resize 到 227，因为改模型了，不用resize了：
          val aggPointBGR = new BytesToBGRImg(normalize = 1f, 32, 32).apply(
            Iterator(ByteRecord(aggByteRecord, aggLabel))
          ).toArray
          val aggPointSample = new BGRImgToSample(toRGB = false).apply(aggPointBGR.toIterator).next()

          // 保存图片：
          //println(s"第 ${bucketCnt} 桶：原始图像 ${samplesBGR.length} 张，压缩点 ${aggPointBGR.length} 张")
          //saveImg(model, aggByteRecord, aggLabel, s"${bucketDir}\\${bucketCnt}_agg_${aggPointBGR(0).label().toInt}
          // .jpg")


          /*
            貌似发现了 BigDL 库的一个 Bug：\dataset\image\BGRImgToSample.scala 中，
            class BytesToBGRImg 的 apply 方法，传入 Iterator[ByteRecord] 后返回的 Iterator[LabeledBGRImage]
            是相同的！！！
            通过看源码发现，迭代器中所有图片貌似都放到了同一段内存 buffer 中，导致之后的数据覆盖了之前的数据！
            所以只能一张图片一张图的转换。
           */
          // 将压缩点对应的原始图片数据转换为Sample格式：
          val samples = originPoints.map((br) => {
            val sampleBGR = new BytesToBGRImg(normalize = 1f, 32, 32).apply(
              Iterator(br._1)
            ).toArray
            //saveImg(model, br._1, s"${bucketDir}\\${bucketCnt}_origin_${br._1.label.toInt}_${br._2}.jpg")
            val sample = new BGRImgToSample(toRGB = false).apply(sampleBGR.toIterator).toArray
            sample(0)
          })


          // 将压缩点和对应的图片合为元组：
          Tuple2(aggPointSample, samples.toSeq)
        }
      }
    })

    //ArrayBuffer.empty[(Sample[Float], Seq[Sample[Float]])].toArray
  }
}


/*

//val bucketsAccum = new BucketAccum
//sc.register(bucketsAccum)
/**
 * 手写了个累加器，但仍然无法直接从executor累加 ArrayBuffer[(Sample[Float], Seq[Sample[Float]])]，
 * 会报 Task not Serializable 相关错误，
 * 目前该累加器未使用。
 */
class BucketAccum
  extends AccumulatorV2[(Sample[Float], Seq[Sample[Float]]), ArrayBuffer[(Sample[Float], Seq[Sample[Float]])]] {

  val list = new ArrayBuffer[(Sample[Float], Seq[Sample[Float]])]()

  // 当前累加器是否为初始化状态：
  override def isZero: Boolean = {
    list.isEmpty // 列表为空时即为初始化状态
  }

  // 复制累加器对象：
  override def copy(): AccumulatorV2[(Sample[Float], Seq[Sample[Float]]), ArrayBuffer[(Sample[Float],
    Seq[Sample[Float]])]] = {
    new BucketAccum() // 为简便，直接返回一个新实例
  }

  // 重置累加器：
  override def reset(): Unit = {
    list.clear()
  }

  // 向累加器中增加数据：
  override def add(v: (Sample[Float], Seq[Sample[Float]])): Unit = {
    list += v
  }

  // 合并：
  override def merge(other: AccumulatorV2[(Sample[Float], Seq[Sample[Float]]), ArrayBuffer[(Sample[Float],
    Seq[Sample[Float]])]]): Unit = {
    other.value.foreach(bucket => {
      list += bucket
    })
  }

  // 获取累加器结果
  override def value: ArrayBuffer[(Sample[Float], Seq[Sample[Float]])] = list
}
 */

/**
 *
 * @param model “lenet”或“alexnet”
 * @param imgs 所有图像的ByteReocrd数组
 * @param byteBufferOffset 当数据集为cifar-10时，ByteRecord的offset（因为前8个字节为2个Int，表示图像宽高）
 * @param sc
 */
class AggPoint(
                model: String,
                imgs: Array[ByteRecord],
                byteBufferOffset: Int = 0,
                @transient sc: SparkContext, // sc 无法被序列化
                trainMeans: (Double, Double, Double),
                trainStds: (Double, Double, Double)
              ) extends Serializable {

  import org.apache.log4j.Logger
  import com.intel.analytics.bigdl.utils.LoggerFilter
  @transient val logger: Logger = Logger.getLogger(getClass)
  LoggerFilter.redirectSparkInfoLogs()

  //val trainMeans = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  //val trainStds = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  //val testMeans = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  //val testStds = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  val imgNum = imgs.length
  val imgSize = imgs(0).data.size - byteBufferOffset // 每张图片的像素点数量
  //val allImgTensor = AggPoint.byteRecords2Tensor(imgs, byteBufferOffset);
  val allImgTensorNormal = AggPoint.byteRecords2Tensor(imgs, byteBufferOffset, trainMeans, trainStds, 255)
  // TODO: imgs 和 allImgTensor 可以用 broadcast 优化一下

  println(s"Current model:${model}, imgNum:${imgNum}, imgSize:${imgSize}")


  /**
   * 根据 IncreSVD 降维，我的分布式实现
   * @param svdComputeNum 迭代次数
   * @param svdNumPerBucket 每个桶分为几个子桶
   * @param svdNfStr 分布式多层降维时，每层的属性个数
   * @param svdInitValue IncreSVD的初始值
   * @param svdLrate IncreSVD的学习率
   * @param svdK IncreSVD的K
   * @return
   */
  @deprecated
  def reduceDimByIncreSVD(
                           svdComputeNum: Int,
                           svdNumPerBucket: Int,
                           svdNfStr: Option[String],
                           svdInitValue: Double, // 0.1
                           svdLrate: Double, //0.001
                           svdK: Double
                         ): Tensor[Float] = {

    println(s"开始 IncreSVD 降维")

    // 将 svdNfStr 字符串分割为降维各个阶段的参数：
    //print("NF:")
    val nfArr = svdNfStr.get.split(',').map((s) => {
      val thisNf = s.trim.toInt
      //print(" " + thisNf)
      thisNf
    }) // 将字符串以逗号分割为各个阶段的nf值
    //println("")
    val nPixel = nfArr(nfArr.length - 1); // 压缩到 nPixel 个像素

      println(s"分布式降维过程：原图片尺寸 ${imgSize} px：" +
        s"\n -> 每 ${nfArr(0)}px 变为 ${nfArr(1)}px，降至 ${imgSize / nfArr(0) * nfArr(1)} px" +
        s"\n -> 每 ${nfArr(2)}px 变为 ${nfArr(3)}px，降至 ${imgSize / nfArr(0) * nfArr(1) / nfArr(2) * nfArr(3)} px" +
        s"\n -> 每 ${nfArr(4)}px 变为 ${nfArr(5)}px，降至 ${
          imgSize / nfArr(0) * nfArr(1) / nfArr(2) * nfArr(3) / nfArr(4) * nfArr(5)
        } px");

    val svdRound = svdComputeNum; // 迭代120次
    //println(s"每次降维迭代 ${svdRound} 次，最终降至 ${nfArr(5)}px，开始降维迭代计算（可能需要较长时间，请稍等）...")

    //return ArrayBuffer.empty[(Sample[Float], Seq[Sample[Float]])].toArray

    val svdStart = System.nanoTime()
    val aggImgTensor = IncreSVD.calcDistri(
      allImgTensorNormal,
      nfArr,
      svdRound,
      sc,
      svdInitValue.toFloat,
      svdLrate.toFloat,
      svdK.toFloat
    )
    //println(s"svd 降维计算花费：${(System.nanoTime() - svdStart) / 1e9}s")
    aggImgTensor
  }


  // 我自己复现的 ZF 的 IncreSVD：
  @deprecated
  def reduceDimByIncreSVD(
                           svdComputeNum: Int = 120,
                           svdNumPerBucket: Int = 2,
                           svdNf: Int = 5,
                           svdInitValue: Double = 0.1f, // 0.1
                           svdLrate: Double = 0.001f, //0.001
                           svdK: Double = 0.015f,
                           svdRatioN: Int = 5
                         ): Tensor[Float] = {

    println(s"开始 IncreSVD 降维(ZF)，svdRatioN:${svdRatioN}")

    //println(s"每次降维迭代 ${svdRound} 次，最终降至 ${nfArr(5)}px，开始降维迭代计算（可能需要较长时间，请稍等）...")

    //return ArrayBuffer.empty[(Sample[Float], Seq[Sample[Float]])].toArray

    val svdStart = System.nanoTime()
    val aggImgTensor = IncreSVD.calcDistri(
      allImgTensorNormal,
      svdNf,
      svdComputeNum,
      sc,
      svdInitValue.toFloat,
      svdLrate.toFloat,
      svdK.toFloat,
      svdRatioN
    )
    //println(s"svd 降维计算花费：${(System.nanoTime() - svdStart) / 1e9}s")
    aggImgTensor
  }

  /**
   * 什么都不做，直接取图像前 nf 个像素作为特征像素
   * @param nf
   * @return
   */
  def reduceDimByNull(nf: Int): Tensor[Float] = {
    allImgTensorNormal.narrow(2, 1, nf) - 0
  }


  def reduceDimByIncreSVD(
                           upBound: Int,
                           svdComputeNum: Int,
                           svdRatioN: Int
                         ):Array[(Sample[Float], Seq[Sample[Float]])] = {

    import cn.wangxw.ZF.{IncreSVD, ZFBLAS, ZFHashLayer}

    // 直接用ZFMllib的IncreSVD+LSH的代码（经过少量改写）

    val bufferOffset = byteBufferOffset;
    val normalize = 255;

    val imgNum = imgs.length
    val imgSize = imgs(0).data.size - bufferOffset // 每张图片的像素点数量

    logger.info("TIMER: Compute agg point by [ ZF: IncreSVD + LSH ]")
    //require((imgs.last.data.length - bufferOffset) % 3 == 0, "Array[Byte] length % 3 != 0 ")
    val labeledPoints = imgs.map((br) => {
      // 将 ByteRecord 的 data 数组转为 tensor：
      val t = ArrayBuffer.empty[Double]

      var i = 0
      br.data.slice(bufferOffset, br.data.length).foreach((byte) => {
        val f = AggPoint.byte2Float(byte).toDouble
        if (i % 3 == 0) {
          // ByteRecord 是 BGR 顺序，means 和 stds 是 RGB 顺序
          t += (f / normalize - trainMeans._3) / trainStds._3
        }
        else if (i % 3 == 1) {
          t += (f / normalize - trainMeans._2) / trainStds._2
        }
        else if (i % 3 == 2) {
          t += (f / normalize - trainMeans._1) / trainStds._1
        }

        i += 1
      })
      //println(s"label:${br.label}, size:${t.length}")
      //t.foreach(d => {
      //  print(s"${d}\t")
      //})
      //println("")

      new LabeledPoint(br.label, Vectors.dense(t.toArray))
    })

    val zfHash = new ZFHashLayer(svdComputeNum, svdRatioN, upBound, false, sc = sc)

    val svdStart = System.nanoTime()
    val indexs = Array.range(0, labeledPoints.size)
    val tempItqBitN = math.log(indexs.size / 10 / upBound) / math.log(2) // 需要分桶的次数，也是降维后目标属性个数
    val zfsvd = new IncreSVD(labeledPoints, indexs, math.round(tempItqBitN).toInt,
      svdComputeNum,
      svdRatioN)
    zfsvd.calcFeaatures(false)
    logger.info(s"IncreSVD cost: ${(System.nanoTime() - svdStart) / 1e9} seconds")

    val lshStart = System.nanoTime()
    import breeze.linalg.{DenseMatrix => BDM}
    val v: BDM[Double] = zfsvd.userFeas.t // 降维后的矩阵
    val numFeature = labeledPoints.last.features.size
    val trainSet = labeledPoints.zipWithIndex
      .groupBy(_._1.label.toInt).par // 根据label先分组
      .map(tup => {
        val label = tup._1
        val arr = tup._2 // LabeledPoint, ID(index from 0)
        val newV: BDM[Double] = BDM.zeros[Double](arr.length, v.cols)
        val sameLabelLps = new Array[LabeledPoint](arr.length)
        var cnt = 0
        arr.foreach(tup => {
          val id = tup._2 // 全局id，从 0 开始
          // 从所有图片的矩阵 v 中把 label 相同的提取到 newV
          for (i <- 0 until v.cols) {
            newV(cnt, i) = v(id, i)
          }
          sameLabelLps(cnt) = labeledPoints(id)
          cnt += 1
        })

        val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
        val u = BDM.zeros[Int](newV.rows, newV.cols)
        val splitStart = System.nanoTime()
        zfHash.splitU(newV, u, 0, Array.range(0, newV.rows))
        for (i <- 0 until u.rows) {
          val key = u(i, ::).inner.toArray.mkString("")
          val mapKey = key
          val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
          aset += indexs(i)
          aMap.update(mapKey, aset)
        }
        logger.info(s"split cost: ${(System.nanoTime() - splitStart) / 1e9} seconds")
        val buckets = aMap.map(_._2.toArray).toArray
        buckets.map(is => {
          val zipFeature1 = Vectors.zeros(numFeature)
          var label1 = 0.0
          val points = is.map(i => sameLabelLps(i))
          points.foreach(p => {
            ZFBLAS.axpy(1.0, p.features, zipFeature1)
            label1 += p.label
          })
          ZFBLAS.scal(1.0 / is.size, zipFeature1)
          label1 = label1 / is.size
          (new LabeledPoint(label1, zipFeature1), points)
        })
          .map(tup => {
            // 归一化之后是不能转为 ByteRecord 再转为 BGRImg 的，因为浮点数经过 .toByte 会仅保留整数部分，
            // 导致几乎所有的浮点值都变为 0 了

            //bucketCnt += 1
            // 压缩点：
            val aggPointSample = Sample(
              AggPoint.toFeature(tup._1.features, model, true),
              tup._1.label.toFloat
            )

            // 原本的 new BytesToBGRImg(normalize = 1f, 32, 32)，new BGRImgToSample(toRGB = false)

            // 保存图片：
            //AggPoint.saveImg("alexnet", aggBytes, tup._1.label.toFloat,
            //  s"${bucketDir}\\${bucketCnt}_agg_${aggPointBGR(0).label().toInt}.jpg")

            // 将压缩点对应的原始图片数据转换为Sample格式：
            //var imgCnt = 0
            val samples = tup._2.map((lp) => {
              //imgCnt += 1
              Sample(
                AggPoint.toFeature(lp.features, model, true),
                lp.label.toFloat
              )
            })

            //println(s"第 ${bucketCnt} 桶：原始图像 ${samples.length} 张，压缩点 ${aggPointBGR.length} 张")

            // 将压缩点和对应的图片合为元组：
            Tuple2(aggPointSample, samples.toSeq)
          }).toArray
      }).flatten.toArray

    logger.info(s"LSH cost: ${(System.nanoTime() - lshStart) / 1e9} seconds")

    // 调试检查：
    //trainSet.foreach(tup => {
    //  println(s"agg: ${tup._1.label().valueAt(1)} , ${tup._1.feature().toString.slice(0,50)}")
    //  tup._2.foreach(sam => {
    //    if(sam.label() != tup._1.label()){
    //      logger.error("label 不相等！！！")
    //    }
    //  })
    //  print(s"有 ${tup._2.length} 个：")
    //  println(s"dim:${tup._2.last.feature().nDimension()} size:${tup._2.last.feature().nElement()}")
    //  tup._2.foreach(sam => {
    //    println(s"   ${sam.feature().toString.slice(0,100)}")
    //  })
    //})

    trainSet
  }

  /**
   * PCA 降维；可用LSH-single-bisecting分桶；
   * @param nf 降维后目标属性数
   * @return
   */
  def reduceDimByPCA(nf: Int): Tensor[Float] = {

    logger.info(s"开始 PCA 降维，目标属性个数：${nf}")

    // 转为 spark mllib 的格式（Vector，LabeledPoint），
    // 然后用库里的 PCA 降维，
    // 再转回 BigDL 的格式

    val allImgArr = ArrayBuffer[(Tensor[Float], Float)]() // feature,id
    for (i <- 1 to allImgTensorNormal.size(1)) {
      allImgArr += Tuple2(allImgTensorNormal.narrow(1, i, 1), i)
    }
    val allImgVector = sc.parallelize(allImgArr.map(t => {
      new LabeledPoint(t._2.toDouble,
        Vectors.dense(
          // 要用 toDouble 显式地将 Float 转为 Double，不能直接用 .asInstanceOf[Array[Double]]：
          t._1.resize(imgSize).toArray().map(_.toDouble)
        )
      )
    }))

    // Compute the top 5 principal components.
    logger.info("start to train PCA")
    val pca = new PCA(nf).fit(allImgVector.map(_.features))
    logger.info("start to compute PCA")
    // Project vectors to the linear space spanned by the top 5 principal
    // components, keeping the label
    val projected = allImgVector.map(p => p.copy(features = pca.transform(p.features))).collect()
    logger.info("start to transform to tensor")
    val ansTensor = Tensor[Float](imgs.length, nf)
    for (i <- 1 to projected.length) {
      val arr = projected(i - 1).features.toArray
      for (j <- 1 to nf) {
        ansTensor.setValue(projected(i - 1).label.toInt, j, arr(j - 1).toFloat)
      }
    }
    ansTensor
  }

  /**
   * 类似于 IncreSVD 的方法，把所有图像组成一个大的 tensor，进行降维，
   * 取 SVD 降维后的 U（左奇异值矩阵）
   * @param nf
   * @return
   */
  def reduceDimBySVD(nf: Int): Tensor[Float] = {

    val (uTensor, sTensor, vTensor) = AggPoint.svdDistri(
      allImgTensorNormal,
      imgSize,
      allImgTensorNormal.size(1),
      nf,
      sc
    )

    // TODO: 不确定 SVD 降维之后的各个图片是否还按照原顺序

    uTensor
  }

  /**
   * 直接用的 Breeze 库的 SVD 方法
   * @param nf
   * @return
   */
  def reduceDimBySVDSingle(nf: Int): Tensor[Float] = {
    val allImgArr = ArrayBuffer[(Tensor[Float], Int)]() // feature,id(从1开始)
    for (i <- 1 to allImgTensorNormal.size(1)) {
      allImgArr += Tuple2(allImgTensorNormal.narrow(1, i, 1), i)
    }

    val (imgWidth, imgHei, channel) = model match {
      case "lenet" => {
        (28, 28, 1)
      }
      case "alexnet" => {
        (32, 32, 3)
      }
    }

    require(imgSize / channel == imgWidth * imgHei, "图像尺寸错误")

    var idCnt = 0

    // 改为分布式之后好像更慢了点，所以改回本地的并行处理了
    val imgSvArr = allImgArr.par
      .map(tup => {
      val thisImgTensor = tup._1
      val x = DenseMatrix.fill[Double](imgHei, imgWidth)(0) // 建该图像矩阵，填充0

      thisImgTensor.resize(imgSize)

      for (i <- 0 until imgHei) {
        for (j <- 0 until imgWidth) {
          x(i, j) = thisImgTensor.valueAt(i * imgWidth + j + 1)
        }
      }
      //val covmat = cov(x)  // 将x转为某种格式？这是从spark还是哪复制的，能直接用，但不用这行代码
      // 参考：https://github.com/scalanlp/breeze/tree/a478712806fd534f37016f421ef02d1762741db0/math/src/main/scala
      // /breeze/linalg
      val SVD(u, s, v) = svd(x)

      //println(s"id:${idCnt}")
      //println(s"u(${u.rows},${u.cols}):\n${u}\ns:${s}\nv(${v.rows},${v.cols}):${v}")

      // 将DenseVector等Spark Mllib的格式转为 BigDL 的格式：
      //val uTensor = Tensor[Float](imgHei, nf)
      //val sTensor = Tensor[Float](nf, nf)
      //val vTensor = Tensor[Float](nf, imgWidth)
      //for(i <- 0 until imgHei){
      //  for(j <- 0 until nf){
      //    uTensor.setValue(i+1, j+1, u(i,j).toFloat)
      //  }
      //}
      //for(i <- 0 until nf){
      //  sTensor.setValue(i+1, i+1, s(i).toFloat)
      //}
      //for(i <- 0 until nf){
      //  for(j <- 0 until imgWidth){
      //    vTensor.setValue(i+1, j+1, v(i,j).toFloat)
      //  }
      //}


      // 由降维后的三个tensor还原图像：
      //var ansTensor = uTensor * sTensor * vTensor
      //
      //var minV = 0f
      //var maxV = 0f
      //for(i <- 1 to ansTensor.size(1)){
      //  for(j <- 1 to ansTensor.size(2)){
      //    minV = math.min(ansTensor.valueAt(i,j), minV)
      //    maxV = math.max(ansTensor.valueAt(i,j), maxV)
      //  }
      //}
      //
      //ansTensor = (ansTensor - minV) / (maxV - minV) * 255f
      //
      //AggPoint.saveImg(
      //  model,
      //  imgs(idCnt),
      //  s"D:\\课件（研）\\bigdl\\datasets\\lenet_svd\\${idCnt}_${imgs(idCnt).label}_origin.jpg"
      //)
      //AggPoint.saveImg(
      //  model,
      //  AggPoint.tensor2bytes(ansTensor.resize(imgSize), byteBufferOffset, model),
      //  imgs(idCnt).label,
      //  s"D:\\课件（研）\\bigdl\\datasets\\lenet_svd\\${idCnt}_${imgs(idCnt).label}_svd.jpg"
      //)

      idCnt += 1

      // TODO: 直接将奇异值作为分桶的特征值是否合适？
      Tuple2(s.toArray, tup._2)
    }).toArray

    val svTensor = Tensor[Float](imgs.length, nf)
    for (i <- imgSvArr.indices) {
      for (j <- 1 to nf) {
        // imgSvArr(i)._2 为图像ID，将 (imgSvArr(i)._1) 数组的每个值赋到 tensor 中：
        svTensor.setValue(imgSvArr(i)._2, j, (imgSvArr(i)._1) (j - 1).toFloat)
      }
    }

    svTensor
  }

  /**
   * 计算所有图像的 pHash 指纹串；可用汉明距离衡量相似度；
   * @param hashFpStrLen
   * @return imgNum*1 的长整型 tensor，每个长整型为 64 位，表示该图像的指纹
   */
  def computeAHash(hashFpStrLen: Int = 64): Tensor[Float] = {
    var imgCnt = 1
    val ansTensor = Tensor[Long](imgNum, 1) // n*1的tensor
    model match {
      case "lenet" => {
        imgs.foreach(br => {
          // 注意：br 的 label 比实际的 label 值大 1；

          ansTensor.setValue(
            imgCnt,
            1,
            AggPoint.aHash(br.data, 28, 28, 1)
          )
          //AggPoint.saveImg("lenet", bytes, br.label, s"D:\\课件（研）\\bigdl\\datasets\\test\\${imgCnt}_${br.label}
          // .jpg", 8, 8)
          imgCnt += 1
        })
      }
      case "alexnet" => {
        imgs.foreach(br => {
          // 注意：br 的 label 比实际的 label 值大 1；

          ansTensor.setValue(
            imgCnt,
            1,
            AggPoint.aHash(br.data.slice(byteBufferOffset, br.data.length),
              32,
              32,
              3
            )
          )
          //AggPoint.saveImg("lenet", bytes, br.label, s"D:\\课件（研）\\bigdl\\datasets\\test\\${imgCnt}_${br.label}
          // .jpg", 8, 8)
          imgCnt += 1
        })
      }
    }
    long2floatTensor(ansTensor)
  }

  def computePHash(): Tensor[Float] = {
    val ansTensor = Tensor[Long](imgNum, 1)
    model match {
      case "lenet" => {
        for (i <- 1 to imgNum) {
          ansTensor.setValue(
            i,
            1,
            AggPoint.pHash(
              allImgTensorNormal.narrow(1, i, 1),
              28, 28, 1, 8, 8
            )
          )
        }
      }
      case "alexnet" => {
        for (i <- 1 to imgNum) {
          ansTensor.setValue(
            i,
            1,
            AggPoint.pHash(
              allImgTensorNormal.narrow(1, i, 1),
              32, 32, 3, 8, 8
            )
          )
        }
      }
    }
    long2floatTensor(ansTensor)
  }

  def computeDHash(): Tensor[Float] = {
    var imgCnt = 1
    val ansTensor = Tensor[Long](imgNum, 1) // n*1的tensor
    model match {
      case "lenet" => {
        imgs.foreach(br => {
          // 注意：br 的 label 比实际的 label 值大 1；

          ansTensor.setValue(
            imgCnt,
            1,
            AggPoint.dHash(br.data, 28, 28, 1)
          )
          //AggPoint.saveImg("lenet", bytes, br.label, s"D:\\课件（研）\\bigdl\\datasets\\test\\${imgCnt}_${br.label}
          // .jpg", 8, 8)
          imgCnt += 1
        })
      }
      case "alexnet" => {
        imgs.foreach(br => {
          // 注意：br 的 label 比实际的 label 值大 1；

          ansTensor.setValue(
            imgCnt,
            1,
            AggPoint.aHash(br.data.slice(byteBufferOffset, br.data.length),
              32,
              32,
              3
            )
          )
          //AggPoint.saveImg("lenet", bytes, br.label, s"D:\\课件（研）\\bigdl\\datasets\\test\\${imgCnt}_${br.label}
          // .jpg", 8, 8)
          imgCnt += 1
        })
      }
    }

    long2floatTensor(ansTensor)
  }

  private def long2floatTensor(ansTensor: Tensor[Long], hashFpStrLen: Int = 64): Tensor[Float] = {
    // 将Long转为64个Int：
    val t = Tensor[Float](imgNum, hashFpStrLen)
    for (i <- 1 to imgNum) {
      for (j <- 1 to hashFpStrLen) {
        t.setValue(
          i,
          j,
          ((ansTensor.valueAt(i, 1) >> (hashFpStrLen - j)) & 1).toFloat
        )
      }
    }
    t
  }

  /**
   * 计算直方图；可用欧氏距离衡量相似度
   * @param segNum 对于RGB图像，通道值0~255划分为 segNum 个区间；
   *               如，为6时，0~255/6为第一区间，该区间的值映射为0，255/6~255/6*2映射为1；最大映射为5；
   *               那么，RGB组合起来的索引值为 B*6*6+G*6+R*1，最大为 215；也可以理解为用6进制对BGR编码；
   *               参考：https://segmentfault.com/a/1190000018849195；
   *               另一种（未实现的）编码：https://blog.csdn.net/u010186001/article/details/52800250
   * @return
   */
  def computeHistogram(segNum: Int = 6): Tensor[Float] = {
    model match {
      case "lenet" => {
        val ansTensor = Tensor[Float](imgNum, 256).fill(0f)
        for (i <- 1 to imgNum; j <- 1 to imgSize) {
          val color = allImgTensorNormal.valueAt(i, j).toInt + 1
          //println(s"i:${i}, color:${color}")
          ansTensor.setValue(i, color,
            ansTensor.valueAt(i, color) + 1f
          )
        }
        ansTensor / imgSize // 归一化，归至 0~1
      }
      case "alexnet" => {
        val idxNum = segNum * segNum * segNum
        val ansTensor = Tensor[Float](imgNum, idxNum).fill(0f)
        for (i <- 1 to imgNum; j <- 0 until (imgSize/3)) {
          val idx = j * 3 + 1
          val color = (
            math.min(segNum-1, math.floor(allImgTensorNormal.valueAt(i, idx) * segNum / 255)) * 6 * 6 +
            math.min(segNum-1,  math.floor(allImgTensorNormal.valueAt(i, idx + 1) * segNum / 255)) * 6 +
            math.min(segNum-1,  math.floor(allImgTensorNormal.valueAt(i, idx + 2) * segNum / 255))
            ).toInt + 1
          //println(s"i:${i}, color:${color}")
          ansTensor.setValue(i, color,
            ansTensor.valueAt(i, color) + 1f
          )
        }
        ansTensor / (imgSize/3) // 归一化，归至 0~1
      }
    }
  }

  /**
   * K-means 分桶；参考：https://spark.apache.org/docs/2.3.0/mllib-clustering.html
   * @param aggImgTensor
   * @param upBound
   * @param numIterations 最多迭代次数
   * @return
   */
  def clusterByKMeans(aggImgTensor: Tensor[Float], upBound: Int = 5, numIterations: Int = 20)
  : Array[(Tensor[Float], Float, Array[Int])] = {

    require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")


    val bucketNum = math.round(aggImgTensor.size(1).toFloat / 10.toFloat / upBound.toFloat).toInt


    // Load and parse the data
    //val data = sc.textFile("data/mllib/kmeans_data.txt")
    //val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    val vectorArr = new Array[linalg.Vector](imgNum)
    for (i <- 1 to imgNum) {
      vectorArr(i - 1) = Vectors.dense(
        (aggImgTensor.narrow(1, i, 1) - 0)
          .resize(aggImgTensor.size(2))
          .toArray()
          .map(f => f.toDouble)
      )
    }

    vectorArr.zipWithIndex.groupBy(tup => {
      imgs(tup._2).label
    }).par.map(sameLabelGroup => {
      // 对label相同的图像们进行k-means：

      val label = sameLabelGroup._1
      val parsedData = sc.parallelize(sameLabelGroup._2.map(_._1)).cache()  // 把 Vector 取出来
      val clusters = KMeans.train(parsedData, bucketNum, numIterations)
      var imgCnt = 0;
      clusters.predict(parsedData)
        .collect()
        .zipWithIndex // (Int1, Int2)  第 Int2 张图像在第 Int1 个桶中
        .map(tup => (tup._2, tup._1))   // 把两个 Int 的位置换一下，现在为 (imgId, bucketId)
        .groupBy(_._2).par // 各个桶并行执行，但好像本地没有太大提升
        .map(tup => {
          // (桶ID，Array[图片ID，桶ID])
          val bucketId = tup._1
          //logger.info(s"start K-Means predict for bucket ${bucketId}")
          val thisBucketImgArr = tup._2
          require(bucketId == thisBucketImgArr.last._2, "图片的分桶ID产生异常")
          //thisBucketImgArr.foreach(img => {
          //  val idInSameLabelGroup = img._1  // 从 1 开始
          //  // 换算为实际的全局ID：
          //  val imgId = sameLabelGroup._2(idInSameLabelGroup-1)._2
          //
          //  if(bucketId == clusters.predict(vectorArr(imgId-1))){
          //    println(s"ok, img_${imgId}'s bucket is ${bucketId}")
          //  }
          //  else{
          //    println(s"-------------img_${imgId}, bucketId:${bucketId}, predict:${clusters.predict(vectorArr
          //    (imgId))}")
          //  }
          //})
          computeBucket(thisBucketImgArr.map(tup => {
            // tup: [Int, Int]: [图片在相同label的group中的id,bucketId]
            val idInSameLabelGroup = tup._1 // 从 0 开始
            // 换算为实际的全局ID：
            val globalId = sameLabelGroup._2(idInSameLabelGroup)._2
            require(globalId >= 0 && globalId < imgNum, "ID 错误")
            globalId + 1
          }))
        }).toArray
    }).flatMap(arr => arr).toArray

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    //val WSSSE = clusters.computeCost(parsedData)
    //println(s"Within Set Sum of Squared Errors = $WSSSE")
  }

  // TODO: 不知道为什么，很慢：
  def clusterByBisectingKMeans(aggImgTensor: Tensor[Float], upBound: Int = 5, numIterations: Int = 20,
                               minBucketSize: Int = 4)
  : Array[(Tensor[Float], Float, Array[Int])] = {

    require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")
    val bucketNum = math.round(aggImgTensor.size(1).toFloat / 10.toFloat / upBound.toFloat).toInt

    // Load and parse the data
    //val data = sc.textFile("data/mllib/kmeans_data.txt")
    //val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    val vectorArr = new Array[linalg.Vector](imgNum)
    for (i <- 1 to imgNum) {
      vectorArr(i - 1) = Vectors.dense(
        (aggImgTensor.narrow(1, i, 1) - 0)
          .resize(aggImgTensor.size(2))
          .toArray()
          .map(f => f.toDouble)
      )
    }
    // todo: 先对label相同的分组


    var imgCnt = 0;
    vectorArr.map(vec => {
      imgCnt += 1
      (vec, imgCnt)
    }).groupBy(tup => {
      imgs(tup._2 - 1).label
    }).par.map(sameLabelGroup => {
      // 对label相同的图像们进行k-means：

      val label = sameLabelGroup._1
      val parsedData = sc.parallelize(sameLabelGroup._2.map(_._1)).cache()

      // Clustering the data into 6 clusters by BisectingKMeans.
      val bkm = new BisectingKMeans()
        .setK(bucketNum)
        .setMaxIterations(numIterations)
        .setMinDivisibleClusterSize(minBucketSize) // 每个桶最少有多少图像
      val clusters = bkm.run(parsedData)

      var imgCnt = 0;
      clusters.predict(parsedData)
        .collect()
        .map(bucketId => {
          imgCnt += 1
          Tuple2(imgCnt, bucketId)
        })
        .groupBy(_._2).par // 各个桶并行执行，但好像本地没有太大提升
        .map(tup => {
          // (桶ID，Array[图片ID，桶ID])
          val bucketId = tup._1
          val thisBucketImgArr = tup._2
          require(bucketId == thisBucketImgArr.last._2, "图片的分桶ID产生异常")
          //thisBucketImgArr.foreach(img => {
          //  val idInSameLabelGroup = img._1  // 从 1 开始
          //  // 换算为实际的全局ID：
          //  val imgId = sameLabelGroup._2(idInSameLabelGroup-1)._2
          //
          //  if(bucketId == clusters.predict(vectorArr(imgId-1))){
          //    println(s"ok, img_${imgId}'s bucket is ${bucketId}")
          //  }
          //  else{
          //    println(s"-------------img_${imgId}, bucketId:${bucketId}, predict:${clusters.predict(vectorArr
          //    (imgId))}")
          //  }
          //})
          computeBucket(thisBucketImgArr.map(tup => {
            // tup: [Int, Int]: [图片在相同label的sameLabelGroup中的索引,bucketId]
            val idInSameLabelGroup = tup._1 // 从 1 开始
            // 换算为实际的全局ID：
            sameLabelGroup._2(idInSameLabelGroup - 1)._2
          }))
        }).toArray
    }).flatMap(arr => arr).toArray

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    //val WSSSE = clusters.computeCost(parsedData)
    //println(s"Within Set Sum of Squared Errors = $WSSSE")
  }

  def clusterByGaussianMixture(aggImgTensor: Tensor[Float], upBound: Int = 5, numIterations: Int = 20)
  : Array[(Tensor[Float], Float, Array[Int])] = {

    require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")
    val bucketNum = math.round(aggImgTensor.size(1).toFloat / 10.toFloat / upBound.toFloat).toInt

    // Load and parse the data
    //val data = sc.textFile("data/mllib/kmeans_data.txt")
    //val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    val vectorArr = new Array[linalg.Vector](imgNum)
    for (i <- 1 to imgNum) {
      vectorArr(i - 1) = Vectors.dense(
        (aggImgTensor.narrow(1, i, 1) - 0)
          .resize(aggImgTensor.size(2))
          .toArray()
          .map(f => f.toDouble)
      )
    }
    // todo: 先对label相同的分组


    var imgCnt = 0;
    vectorArr.map(vec => {
      imgCnt += 1
      (vec, imgCnt)
    }).groupBy(tup => {
      imgs(tup._2 - 1).label
    }).par.map(sameLabelGroup => {
      // 对label相同的图像们进行k-means：

      val label = sameLabelGroup._1
      val parsedData = sc.parallelize(sameLabelGroup._2.map(_._1)).cache()

      val clusters = new GaussianMixture().setK(bucketNum).run(parsedData)

      var imgCnt = 0;
      clusters.predict(parsedData)
        .collect()
        .map(bucketId => {
          imgCnt += 1
          Tuple2(imgCnt, bucketId)
        })
        .groupBy(_._2).par // 各个桶并行执行，但好像本地没有太大提升
        .map(tup => {
          // (桶ID，Array[图片ID，桶ID])
          val bucketId = tup._1
          val thisBucketImgArr = tup._2
          require(bucketId == thisBucketImgArr.last._2, "图片的分桶ID产生异常")
          //thisBucketImgArr.foreach(img => {
          //  val idInSameLabelGroup = img._1  // 从 1 开始
          //  // 换算为实际的全局ID：
          //  val imgId = sameLabelGroup._2(idInSameLabelGroup-1)._2
          //
          //  if(bucketId == clusters.predict(vectorArr(imgId-1))){
          //    println(s"ok, img_${imgId}'s bucket is ${bucketId}")
          //  }
          //  else{
          //    println(s"-------------img_${imgId}, bucketId:${bucketId}, predict:${clusters.predict(vectorArr
          //    (imgId))}")
          //  }
          //})
          computeBucket(thisBucketImgArr.map(tup => {
            // tup: [Int, Int]: [图片在相同label的sameLabelGroup中的索引,bucketId]
            val idInSameLabelGroup = tup._1 // 从 1 开始
            // 换算为实际的全局ID：
            sameLabelGroup._2(idInSameLabelGroup - 1)._2
          }))
        }).toArray
    }).flatMap(arr => arr).toArray

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    //val WSSSE = clusters.computeCost(parsedData)
    //println(s"Within Set Sum of Squared Errors = $WSSSE")
  }

  /**
   * 根据LSH分桶（聚类）：依次根据 aggImgTensor 的每一行的每一个值进行排序、二分桶，
   * 最终（label相同的图像）分得 2^aggImgTensor.size(2)^ 桶；
   * 可以看作是对汉明距离的哈希函数；
   * @param aggImgTensor 压缩后的图片 tensor，每行为一张图片
   * @param lshMode 1为ZF版实现，2为我自己的实现
   * @return Array[压缩点的 feature tensor, label, Array[(原始点的 ByteRecord, ID（从0开始，与imgs对应）)]]
   **/
  def clusterByLshBisecting(aggImgTensor: Tensor[Float], lshMode: Int = 1)
  : Array[(Tensor[Float], Float, Array[Int])] = {
    logger.info("降维计算完成，开始LSH...")
    // 此时，aggImgTensor 的每一行为一张压缩过的图片数据

    require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")

    // Int相当于图片ID，以下排序均为带ID排序，便于最终根据ID与原图片数据imgs一一对应：
    var aggImgArr = new ArrayBuffer[(Tensor[Float], Float, Int)]()

    for (i <- 1 to imgNum) {
      aggImgArr += Tuple3(aggImgTensor.narrow(1, i, 1), imgs(i - 1).label, i) // 将每行数据（一张图片）放入数组，便于之后LSH的排序分堆
    }

    /*
    小数据输出测试
    var buCnt = 0
    lsh(aggImgArr.slice(0,12).map((tup) => {
      println(s"${tup._3}:\t${tup._1}")
      Tuple2(tup._1, tup._3)
    }).toArray).foreach(arr=>{
      buCnt += 1
      println(s"第 ${buCnt} 桶:\t${arr.mkString("\t")}")
    })
     */

    logger.info("start lsh")
    //var bucketCnt1 = 0
    val bucketArr = aggImgArr.groupBy((tup) => {
      // 根据label将RDD分组
      tup._2
    }).par
      .map((tupleIter) => {
        // 对于label相同的图片（tupleIter._2）进行分桶：


        logger.info(s"start LSH split")
        val imgArr = tupleIter._2.map((tup) => {
          Tuple2(tup._1, tup._3)
        }).toArray
        val splitTime = System.nanoTime()
        val lshResult = AggPoint.lsh(imgArr, 1)
        logger.info(s"split cost: ${(System.nanoTime() - splitTime) / 1e9} seconds")


        //println(s"label:${tupleIter._1}, lsh cost:${(System.nanoTime() - tmpStartTime) / 1e9}s")
        lshResult
      })
      .flatMap((buckets) => {
        // 将label相同的多个桶组成的数组newImgArr展平：
        buckets
      })
      .map((bucket) => {
        // IMPORTANT: 这个map中一些东西无法被序列化，会报错 Task not Serializable，
        // 通过 SerializationDebugger 的输出：
        // Serialization stack:
        //	- object not serializable (class: org.apache.spark.SparkContext, value: org.apache.spark.SparkContext@1b7332a7)
        //	- field (class: cn.wangxw.aggpoint.AggPoint, name: sc, type: class org.apache.spark.SparkContext)
        // 可以发现，是 sc 无法被序列化。推断，这个 map 里用到的一些东西无法被序列化时，
        // Scala 会试图序列化整个 class AggPoint，但 sc:SparkContext 无法被序列化，
        // 导致最终整个类无法被序列化，因而报错；
        // 所以在该 class 继承 Serializable 时，通过 @transient 告诉 Scala 无需序列化 sc，
        // 才能实现整个类的序列化。
        // 参考：https://www.jianshu.com/p/080f18900f62
        // https://stackoverflow.com/questions/22592811/task-not-serializable-java-io-notserializableexception-when
        // -calling-function-ou

        computeBucket(bucket)
      })
    bucketArr.toArray
  }



  //def clusterByLshEucli(aggImgTensor: Tensor[Float])
  //: Array[(Tensor[Float], Float, Array[(ByteRecord, Int)])] = {
  //  require(aggImgTensor.size(1) == imgNum, "压缩后图片数量与原数量不符")
  //  val hashNum = 5
  //  val hashTensor = Tensor[Int](imgNum, hashNum)
  //  for(i <- 1 to hashNum){
  //    // 随机生成一个哈希函数：H(V) = |V·R + b| / a，R是一个随机向量，a是桶宽，b是在[0,a]之间均匀分布的随机变量
  //
  //  }
  //
  //
  //}

  /**
   * 对该桶计算压缩点相关数据
   * @param bucket 元素为图像ID，下标从 1 开始
   * @return 用于生成压缩点的格式
   */
  private def computeBucket(bucket: Array[Int]): (Tensor[Float], Float, Array[Int]) = {
    // 对每个桶求压缩点：
    // 每个桶为一个 ArrayBuffer[ Int ]，Int为原图片ID，从 1 开始
    var aggFeature = Tensor[Float](imgSize).fill(0)
    var aggLabel = 0f;
    val originPoints = new ArrayBuffer[Int]() // Int 为原图ID，从0开始

    //bucketCnt1 += 1
    val testLabel = imgs(bucket.last - 1).label
    bucket.foreach((imgId) => {
      // 这里所有的 img 应该计算出一个压缩点：
      require(imgId >= 1 && imgId <= imgNum, "imgId 越界")
      val thisLabel = imgs(imgId - 1).label
      require(testLabel == thisLabel, s"thisLabel(${thisLabel}) != testLabel(${testLabel})")
      aggLabel += thisLabel  // id是按照tensor从下标1开始的
      aggFeature += allImgTensorNormal.narrow(1, imgId, 1)
      originPoints += imgId
    })
    // 计算压缩点（求平均值）：
    aggFeature /= bucket.length
    aggLabel /= bucket.length
    require(aggLabel == testLabel, s"aggLabel(${aggLabel}) != testLabel(${testLabel})")
    (aggFeature, aggLabel, originPoints.toArray)
  }


  /**
   * 根据每桶计算压缩点、Transformer
   * @param bucketArr 数组，元素为聚类后的压缩点、原始点等
   * @param bucketDir 不为空字符串时，将图片保存到该目录下
   * @return 数组，元素为元组：压缩点、原始点
   */
  def computeAgg(
                  bucketArr: Array[(Tensor[Float], Float, Array[Int])],
                  bucketDir: String = "",
                  normalize: Boolean = true)
  : Array[(Sample[Float], Seq[Sample[Float]])] = {
    println(s"共有 ${bucketArr.length} 个桶")
    var bucketCnt = 0

    bucketArr.map((tup) => {
      // TODO: 不知道为什么，这个 map 里的操作放在executor上就回报错：not Serializable: scala.collection.Iterator
      // TODO: 可能是因为 Transformer ？

      // 在 driver 上对压缩点和原始图像进行预处理：
      val aggFeature = tup._1  // dim=1
      val aggLabel = tup._2
      val originPoints = tup._3

      bucketCnt += 1

      model match {
        case "lenet" => {

          // 手动Transformer和自动的结果还是不太一样，弃用手动的：
          //val aggPointSample1 = Sample(
          //  (aggFeature.resize(28, 28) / 255f - trainMean.toFloat) / trainStd.toFloat,
          //  aggLabel
          //)

          val aggPointSample = Sample(AggPoint.toFeature(aggFeature, model, true), aggLabel)

          if (!bucketDir.equals("")) {
            // 保存图片：
            //AggPoint.saveImg(model, aggBytes, aggLabel, s"${bucketDir}\\${bucketCnt}_agg_${aggLabel.toInt - 1}.jpg")
          }

          // 将压缩点对应的原始图片数据转换为Sample格式：
          //val samples = new GreyImgToSample().apply(
          //  new GreyImgNormalizer(trainMean, trainStd).apply( // 类型LabeledGreyImage不变，(value-mean)/std
          //    new BytesToGreyImg(28, 28).apply( // Array[Byte] 转换为 LabeledGreyImage:Array[Float],norm到0~1
          //      originPoints.map(_._1).toIterator
          //    )
          //  )
          //).toArray // TODO: BytesToBGRImg.apply 有bug，BytesToGreyImg 就没bug？
          val samples = originPoints.map(imgId => {
            require(imgId >= 1 && imgId <= imgNum, s"imgId out of range(1 ~ ${imgNum})")
            val t = (allImgTensorNormal.narrow(1, imgId, 1) - 0).resize(imgSize)
            Sample(
              AggPoint.toFeature(t, model, true),
              aggLabel
            )
          })

          if (!bucketDir.equals("")) {
            //originPoints.foreach(br => {
            //  AggPoint.saveImg(model, br._1, s"${bucketDir}\\${bucketCnt}_origin_${br._1.label.toInt - 1}_${br._2}.jpg")
            //})
          }

          // 将压缩点和对应的图片合为元组：
          Tuple2(aggPointSample, samples.toSeq)
        }
        case "alexnet" => {

          //println(s"aggFeature size:${aggFeature.nDimension()}")

          val aggPointSample = Sample(AggPoint.toFeature(aggFeature, model, true), aggLabel)

          // 保存图片：
          //AggPoint.saveImg(model, aggBytes, aggLabel, s"${bucketDir}\\${bucketCnt}_agg_${aggPointBGR(0).label().toInt}.jpg")

          // 将压缩点对应的原始图片数据转换为Sample格式：
          //val samples = originPoints.map((br) => {
          //  println(s"index:${br._2}")
          //  Sample(
          //    AggPoint.toFeature(
          //      (allImgTensorNormal.narrow(1, br._2+1, 1)-0).resize(imgSize),
          //      model,
          //      true
          //    ),
          //    br._1.label
          //  )
          //})
          val samples = originPoints.map(imgId => {
            require(imgId >= 1 && imgId <= imgNum, s"imgId out of range(1 ~ ${imgNum})")
            val t = (allImgTensorNormal.narrow(1, imgId, 1) - 0).resize(imgSize)
            Sample(
              AggPoint.toFeature(t, model, true),
              aggLabel
            )
          })

          //println(s"第 ${bucketCnt} 桶：原始图像 ${samples.length} 张，压缩点 ${aggPointBGR.length} 张")

          // 将压缩点和对应的图片合为元组：
          Tuple2(aggPointSample, samples.toSeq)
        }
      }
    })
  }
}
