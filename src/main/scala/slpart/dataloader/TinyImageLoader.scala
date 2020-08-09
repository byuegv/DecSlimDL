package slpart.dataloader

import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint


object TinyImageLoader {
  def loadLabeledPoint(dirPath: String,sc: SparkContext = null) = {
    assert(sc != null,"SparkContext can not be null")
    val parsed = sc.textFile(dirPath).map(line => {
      val splits = line.trim.split(",")
      val label = rescaleLabel(splits.head).toDouble
      val features = splits.tail.map(_.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    })
    parsed
  }

  def loadSamples(dirPath: String,sc: SparkContext = null) = {
    assert(sc != null,"SparkContext can not be null")
    val parsed = sc.textFile(dirPath).map(line => {
      val splits = line.trim.split(",")
      val label = rescaleLabel(splits.head).toFloat
      val features = splits.tail.map(_.toFloat)
      Sample(Tensor(T(features.head,features.tail: _*)),Tensor(T(label)))
    })
    parsed
  }

  def rescaleLabel(label: String) = {
    val tlab = label.toInt
    val clnum = 10
    val mod = tlab % clnum
    if(mod == 0) clnum else mod
  }

  def xrescaleLabel(label: String) = {
    val tlab = label.toInt
    tlab
  }

  def loadTrainLabeledPoint(classNum: Int = 10,dirPath: String,sc: SparkContext = null)= {
    assert(sc != null,"SparkContext can not be null")
    val files = s"${dirPath}/xtrain_set.csv"
    val parsed = sc.textFile(files).map(line => {
      val splits = line.trim.split(",")
      val label = xrescaleLabel(splits.head).toDouble
      val features = splits.tail.map(_.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    })
    parsed
  }
  def loadValidationLabeledPoint(classNum: Int = 10,dirPath: String,sc: SparkContext = null)= {
    assert(sc != null,"SparkContext can not be null")
    val files = s"${dirPath}/xtest_set.csv"
    val parsed = sc.textFile(files).map(line => {
      val splits = line.trim.split(",")
      val label = xrescaleLabel(splits.head).toDouble
      val features = splits.tail.map(_.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    })
    parsed
  }
}
