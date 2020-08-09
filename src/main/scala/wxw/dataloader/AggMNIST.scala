package wxw.dataloader

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import cn.wangxw.aggpoint.AggPoint
import com.intel.analytics.bigdl.dataset.{ByteRecord, Sample}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToSample}
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat

object AggMNIST {
  private val hdfsPrefix: String = "hdfs:"
  private val trainMean = 0.13066047740239506
  private val trainStd = 0.3081078

  private val testMean = 0.13251460696903547
  private val testStd = 0.31048024
  /**
   * load mnist data.
   * read mnist from hdfs if data folder starts with "hdfs:", otherwise form local file.
   *
   * @param featureFile
   * @param labelFile
   * @return
   */
  private def load(
            featureFile: String,
            labelFile: String,
            useRatioN: Int = 1
          )
  : Array[ByteRecord] = {

    val featureBuffer = if (featureFile.startsWith("hdfs")) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith("hdfs")) {
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }

    // 验证数据的起始值 magic number，（http://yann.lecun.com/exdb/mnist/）：
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelMagicNumber = labelBuffer.getInt()
    require(labelMagicNumber == 2049)

    val labelCount = labelBuffer.getInt() / useRatioN // 60000
    val featureCount = featureBuffer.getInt() / useRatioN // 由于虚拟机内存问题，本地测试时取子集60000/useRatioN
    require(labelCount == featureCount)
    val rowNum = featureBuffer.getInt() // 28
    val colNum = featureBuffer.getInt() // 28
    val result = new Array[ByteRecord](featureCount) // result[60000]
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte](rowNum * colNum)
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          // 获取 28*28 个字节像素值：
          img(y * colNum + x) = featureBuffer.get() // 填入一维数组的 [y,x] 处
          x += 1
        }
        y += 1
      } // 将图片字节像素和对应的标签存入 result：
      // ByteRecord(scala.Array[scala.Byte], scala.Float)
      result(i) = ByteRecord(img, labelBuffer.get().toFloat+1.0f)
      i += 1
    }

    // 根据本地所有数据集进行训练（60000 张图片的 img-label 的 ByteRecord）：
    println(s"Loaded complete local data(${result.length})!")
    result
  }

  /**
   * load mnist data.
   * read mnist from hdfs if data folder starts with "hdfs:", otherwise form local file.
   * @param featureFile
   * @param labelFile
   * @return
   */
  private def loadVal(featureFile: String, labelFile: String): Array[ByteRecord] = {

    val featureBuffer = if (featureFile.startsWith(hdfsPrefix)) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith(hdfsPrefix)) {
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f) // change 0-base label to 1-base label
      i += 1
    }
    result
  }

  def trainMnistData(@transient sc: SparkContext,
                     labelSelection: Set[Int],
                     aggMethod: String = "ISVDLSH",
                    path: String = ".",
                    modelName: String = "lenet",
                    upBound: Int = 10,
                    svdComputeNum: Int = 20,
                    svdRatioN: Int = 100,
                    nf: Int = 10
                    ) = {
    val trainImagePath = path + "/train-images-idx3-ubyte"
    val trainLabelPath = path + "/train-labels-idx1-ubyte"

    val imgs = loadVal(trainImagePath,trainLabelPath).filter(x => labelSelection.contains(scala.math.round(x.label)))
    System.out.println(s"The number of ByteRecord of selected points: ${imgs.length}")
    val tMeans = Tuple3(trainMean,trainMean,trainMean)
    val tStds = Tuple3(trainStd,trainStd,trainStd)
    val agg  = new AggPoint(modelName,imgs, 0,sc,tMeans,tStds)
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
      Array(x._1) ++ x._2.toArray.toArray[Sample[Float]]
    })
    sc.parallelize(res)
  }

  def valMnistData(@transient sc: SparkContext,path: String = ".") = {
    val trainImagePath = path + "/t10k-images-idx3-ubyte"
    val trainLabelPath = path + "/t10k-labels-idx1-ubyte"
    val byteRecords = loadVal(trainImagePath,trainLabelPath)
    System.out.println("zero Score validate labeledPoint")
    val trans = BytesToGreyImg(28,28) -> GreyImgNormalizer(testMean,testStd) -> GreyImgToSample()
    val res = trans.apply(byteRecords.toIterator).toArray
    sc.parallelize(res)
  }

}
