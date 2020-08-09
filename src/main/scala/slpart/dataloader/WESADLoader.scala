package slpart.dataloader
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object WESADLoader {
  def loadAsLabeledPoint(dirPath: String,labOrConstr: Boolean = true,sc: SparkContext = null) = {
    assert(sc != null,"SparkContext can not be null")
    val parsed = sc.textFile(dirPath + "/*").map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map{line =>{
        val items = line.split(' ')
        // val label = items.head.split(";").last.toDouble
        val lorc = items.head.split(";")
        val label = if(labOrConstr) lorc.head.toDouble + 1.0 // change 0-based to 1-based
        else lorc.last.toDouble + 1.0 // change 0-based to 1-based

        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
        val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip

        // check if indices are one-based and in ascending order
        var previous = -1
        var i = 0
        val indicesLength = indices.length
        while (i < indicesLength) {
          val current = indices(i)
          require(current > previous, s"indices should be one-based and in ascending order;"
            + s""" found current=$current, previous=$previous; line="$line"""")
          previous = current
          i += 1
        }
        (label, indices, values)
      }}
    // get the max index of features
    val d = parsed.map { case (label, indices, values) =>
      indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1
    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }
}
