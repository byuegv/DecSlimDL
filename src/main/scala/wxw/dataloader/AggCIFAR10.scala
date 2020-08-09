package wxw.dataloader

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import cn.wangxw.aggpoint.AggPoint
import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample}
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToSample, BytesToBGRImg}
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.mutable.ArrayBuffer

object AggCIFAR10 {
  private val hdfsPrefix: String = "hdfs:"

  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  /**
   * load cifar data.
   * read cifar from hdfs if data folder starts with "hdfs:", otherwise form local file.
   *
   * @param featureFile
   * @param result
   */
  private def load(featureFile: String, result: ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1 // 一张图片字节量
    val channelOffset = rowNum * colNum // 每个频道字节量
    val bufferOffset = 8 //
    val featureBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }

    val featureArray = featureBuffer.array() // 所有的字节转为数组
    val featureCount = featureArray.length / imageOffset // 图片数量
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte](rowNum * colNum * 3 + bufferOffset)
      val byteBuffer = ByteBuffer.wrap(img)
      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i * imageOffset).toFloat
      var y = 0
      val start = i * imageOffset + 1
      /*
      y:0 x:0 img(0+2+8)=fa(1+0),img(8+2)=fa(1+0)
            0 img(0+1+8)=fa(1+32*32),img(8+1)=fa(1+32*32)
            0 img(0+0+8)=fa(1+2*32*32),img(8+0)=fa(1+2*32*32)
            1 img(3+1+8)=fa(1+1+0+32*32),img(8+4)=fa(2+32*32)
            2 img(
       */
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          // 将 feature（原始图像）的 [[R][G][B]] 字节排列为 [[B][G][R]]
          img((x + y * colNum) * 3 + 2 + bufferOffset) = featureArray(start + x + y * colNum) // R值
          img((x + y * colNum) * 3 + 1 + bufferOffset) = featureArray(start + x + y * colNum + channelOffset) // G值
          img((x + y * colNum) * 3 + bufferOffset) = featureArray(start + x + y * colNum + 2 * channelOffset) // B值
          x += 1
        }
        y += 1
      }
      result.append(ByteRecord(img, label + 1.0f))
      i += 1
    }
  }
  private def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = dataFile + "/test_batch.bin"
    load(testFile, result)
    result.toArray
  }
  private def loadTrain(dataFile: String, useRatioN: Int = 1): Array[ByteRecord] = {
    val allFiles = Array(
      dataFile + "/data_batch_1.bin",
      dataFile + "/data_batch_2.bin",
      dataFile + "/data_batch_3.bin",
      dataFile + "/data_batch_4.bin",
      dataFile + "/data_batch_5.bin")

    val results = new ArrayBuffer[ByteRecord]()
    allFiles.foreach(load(_, results))
    val result = results.toArray.slice(0, results.length/useRatioN)

    // 根据本地所有数据集进行训练（60000 张图片的 img-label 的 ByteRecord）：
    println(s"Loaded complete local data(${result.length})!")
    result
  }

  def trainCifarData(@transient sc: SparkContext,
                     labelSelection: Set[Int],
                      aggMethod: String = "ISVDLSH",
                      path: String = ".",
                      modelName: String = "alexnet",
                      upBound: Int = 10,
                      svdComputeNum: Int = 20,
                      svdRatioN: Int = 100,
                      nf: Int = 10
                    ) = {
    val imgs = loadTrain(path,useRatioN = 1).filter(x => labelSelection.contains(scala.math.round(x.label)))
    System.out.println(s"The number of ByteRecord of selected points: ${imgs.length}")
    val agg  = new AggPoint(modelName,imgs, 8,sc,trainMean,trainStd)
    val trainSet = aggMethod match {
      case "HKMEANS" => {
        System.out.println("TIMER: Compute agg point by [ ZF: AHash + KMeans ]")
        val aggImgTensor = agg.computeAHash()
        val buckArr = agg.clusterByKMeans(aggImgTensor,upBound,20)
        agg.computeAgg(buckArr,normalize = true)
      }
      case "SVDKMEANS" => {
        System.out.println("TIMER: Compute agg point by [ ZF: SVD Single + KMeans ]")
        val aggImgTensor = agg.reduceDimBySVDSingle(nf)
        val buckArr = agg.clusterByKMeans(aggImgTensor,upBound,20)
        agg.computeAgg(buckArr,normalize = true)
      }
      case _ => agg.reduceDimByIncreSVD(upBound,svdComputeNum,svdRatioN)
    }
    val res = trainSet.map(x => {
      Array(x._1) ++ x._2.toArray[Sample[Float]]
    })
    sc.parallelize(res)
  }

  def valCifarData(@transient sc: SparkContext,path: String = ".") = {
    val byteRecords = loadTest(path)
    System.out.println("zero Score validate labeledPoint")
    val trans = BytesToBGRImg() -> BGRImgNormalizer(testMean,testStd) -> BGRImgToSample()
    val res = trans.apply(byteRecords.toIterator).toArray
    sc.parallelize(res)
  }
}
