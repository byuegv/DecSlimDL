package slpart.dataloader
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.{ByteRecord, LabeledPointToSample}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.mutable.ArrayBuffer

object Cifar10Loader {
  private val hdfsPrefix: String = "hdfs:"

  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  /**
    * load cifar data.
    * read cifar from hdfs if data folder starts with "hdfs:", otherwise form local file.
    * @param featureFile
    * @param result
    */
  def load(featureFile: String, result : ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = if (featureFile.startsWith(hdfsPrefix)) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }

    val featureArray = featureBuffer.array()
    val featureCount = featureArray.length / (rowNum * colNum * 3 + 1)

    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum * 3 + bufferOffset))
      val byteBuffer = ByteBuffer.wrap(img)
      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i * imageOffset).toFloat
      var y = 0
      val start = i * imageOffset + 1
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img((x + y * colNum) * 3 + 2 + bufferOffset) =
            featureArray(start + x + y * colNum)
          img((x + y * colNum) * 3 + 1 + bufferOffset) =
            featureArray(start + x + y * colNum + channelOffset)
          img((x + y * colNum) * 3 + bufferOffset) =
            featureArray(start + x + y * colNum + 2 * channelOffset)
          x += 1
        }
        y += 1
      }
      result.append(ByteRecord(img, label + 1.0f))
      i += 1
    }
  }

  def loadTrain(dataFile: String): Array[ByteRecord] = {
    val allFiles = Array(
      dataFile + "/data_batch_1.bin",
      dataFile + "/data_batch_2.bin",
      dataFile + "/data_batch_3.bin",
      dataFile + "/data_batch_4.bin",
      dataFile + "/data_batch_5.bin"
    )

    val result = new ArrayBuffer[ByteRecord]()
    allFiles.foreach(load(_, result))
    result.toArray
  }

  def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = dataFile + "/test_batch.bin"
    load(testFile, result)
    result.toArray
  }

  /**
    * get the training data
    * @param path the directory where dataset in
    * @param isZeroScore weather to zeroScore training data
    * @return
    */
  def trainingLabeledPoint(path: String,isZeroScore: Boolean = true) = {
    val byteRecords = loadTrain(path)
    if(isZeroScore){
      System.out.println("zero Score training labeledPoint")
      val trans = BytesToBGRImg() -> BGRImgNormalizer(trainMean,trainStd) -> BGRImgToLabeledPoint()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans =  BytesToBGRImg() -> BGRImgToLabeledPoint()
      trans.apply(byteRecords.toIterator).toArray
    }
  }

  /**
    * get the validate data
    * @param path the directory where dataset in
    * @param isZeroScore weather to zeroScore validate data
    * @return
    */
  def validationSamples(path: String,isZeroScore: Boolean = true) = {
    val byteRecords = loadTest(path)
    if(isZeroScore){
      System.out.println("zero Score validate labeledPoint")
      val trans = BytesToBGRImg() -> BGRImgNormalizer(testMean,testStd) -> BGRImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans = BytesToBGRImg() -> BGRImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
  }
  def trainingSamples(sc: SparkContext,path: String,zScore: Boolean = true) = {
    val byteRecords = loadTrain(path)
    if(zScore){
      System.out.println("zero Score training labeledPoint")
      val trans = BytesToBGRImg() -> BGRImgNormalizer(trainMean,trainStd) -> BGRImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans =  BytesToBGRImg() -> BGRImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
  }

  def trainingSamplesAry(sc: SparkContext,path: String,zScore: Boolean = true,isAggregate: Boolean = false,category: Int = 10,
                         itqbitN: Int = 1,
                         itqitN: Int = 20, //压缩算法迭代次数
                         itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                         upBound: Int = 20, //每个压缩点包含原始点个数上限
                         splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                         isSparse: Boolean = false //输入数据是否为libsvm格式
                        ) = {
    if(isAggregate){
      val trainlp = trainingLabeledPoint(path,isZeroScore = zScore)
      System.out.println("generate compressed training Samples  ...\n +" +
        s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
        s" splitN: ${splitN} isSparse: ${isSparse}")
      val tp = Aggregator.singleLayerAggregateAry(category,sc.parallelize(trainlp),
        itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
        splitN = splitN, isSparse = isSparse)
      tp.zipWithUniqueId().map(x => (x._2,x._1))
    }
    else{
      val trainsp = trainingSamples(sc,path,zScore)
      val arySap = trainsp.zipWithIndex.map(x => (x._2.toLong,Array(x._1)))
      val tp = sc.parallelize(arySap)
      tp
    }
  }
  def validateSamples(sc: SparkContext,path: String,zScore: Boolean = true) = {
    val validatesp = validationSamples(path,isZeroScore = zScore)
    sc.parallelize(validatesp)
  }

}
