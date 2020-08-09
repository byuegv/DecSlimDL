package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.ann._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasStepSize, HasTol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, PredictorParams}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.optimization.{ZFGradientDescent, ZFUpdater}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.JavaConverters._


private[classification] trait ZFMultilayerPerceptronParams extends PredictorParams
  with HasSeed with HasMaxIter with HasTol with HasStepSize {

  @Since("1.5.0")
  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Sizes of layers from input layer to output layer. " +
      "E.g., Array(780, 100, 10) means 780 inputs, " +
      "one hidden layer with 100 neurons and output layer of 10 neurons.",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length > 1)


  //  @Since("1.5.0")
  //  final def getLayers: Array[Int] = $(layers)


  @Since("1.5.0")
  final val blockSize: IntParam = new IntParam(this, "blockSize",
    "Block size for stacking input data in matrices. Data is stacked within partitions." +
      " If block size is more than remaining data in a partition then " +
      "it is adjusted to the size of this data. Recommended size is between 10 and 1000",
    ParamValidators.gt(0))


  @Since("1.5.0")
  final def getBlockSize: Int = $(blockSize)


  @Since("2.0.0")
  final val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${ZFMLPC.supportedSolvers.mkString(", ")}. (Default l-bfgs)",
    ParamValidators.inArray[String](ZFMLPC.supportedSolvers))


  @Since("2.0.0")
  final def getSolver: String = $(solver)


  @Since("2.0.0")
  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of the model")


  @Since("2.0.0")
  final def getInitialWeights: Vector = $(initialWeights)

  setDefault(maxIter -> 100, tol -> 1e-4, blockSize -> 128,
    solver -> ZFMLPC.LBFGS, stepSize -> 0.03)
}


object ZFLabelConverter {


  def encodeLabeledPoint(labeledPoint: LabeledPoint, labelCount: Int): (Vector, Vector) = {
    val output = Array.fill(labelCount)(0.0)
    output(labeledPoint.label.toInt) = 1.0
    (labeledPoint.features, Vectors.dense(output))
  }


  def decodeLabel(output: Vector): Double = {
    output.argmax.toDouble
  }
}


@Since("1.5.0")
@Experimental
class ZFMLPC @Since("1.5.0")(testset: Dataset[_], zflayers: Array[Int],
                             @Since("1.5.0") override val uid: String = Identifiable.randomUID("mlpc"))
  extends
    ZFMultilayerPerceptronParams with DefaultParamsWritable {


  private var _seed = "org.apache.spark.ml.ann.FeedForwardTrainer".hashCode.toLong
  private var _weights: Vector = null
  private val dataStacker = new ZFDataStacker(getBlockSize, zflayers(0), zflayers.last)
  val topology = ZFFeedForwardTopology.multiLayerPerceptron(zflayers, softmaxOnTop = true)
  val ziptopology = ZFFeedForwardTopology.multiLayerPerceptron(zflayers, softmaxOnTop = true)
  private var _gradient: ZFANNGradient = new ZFANNGradient(ziptopology, topology, dataStacker)
  private var _updater: ZFUpdater = new ZFANNUpdater()


  private var optimizer: ZFGradientDescent = new ZFGradientDescent(_gradient, _updater, zflayers, testset) //LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(100)


  def getWeights: Vector = _weights


  def setWeights(value: Vector): this.type = {
    _weights = value
    this
  }


  def SGDOptimizer: ZFGradientDescent = {
    val sgd = new ZFGradientDescent(_gradient, _updater, zflayers, testset)
    optimizer = sgd
    sgd
  }


  @Since("1.5.0")
  def setLayers(value: Array[Int]): this.type = set(layers, value)


  @Since("1.5.0")
  def setBlockSize(value: Int): this.type = {
    dataStacker.stackSize = value
    set(blockSize, value)

  }


  @Since("2.0.0")
  def setSolver(value: String): this.type = set(solver, value)


  @Since("1.5.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)


  @Since("1.5.0")
  def setTol(value: Double): this.type = set(tol, value)


  @Since("1.5.0")
  def setSeed(value: Long): this.type = set(seed, value)


  @Since("2.0.0")
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)


  @Since("2.0.0")
  def setStepSize(value: Double): this.type = set(stepSize, value)

  @Since("1.5.0")
  override def copy(extra: ParamMap): ZFMLPC = defaultCopy(extra)

  protected def extractLabeledPoints(dataset: Dataset[_]): RDD[LabeledPoint] = {
    dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
  }

  def train(dataset: Dataset[_]): ZFMultilayerPerceptronClassificationModel = {
    val myLayers = zflayers
    val labels = myLayers.last
    val lpData = extractLabeledPoints(dataset)
    val data = lpData.map(lp => ZFLabelConverter.encodeLabeledPoint(lp, labels))


    if ($(solver) == ZFMLPC.GD) {
      val sgd = new ZFGradientDescent(_gradient, _updater, zflayers, testset)
      optimizer = sgd
      sgd
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by ZFMLPC.")
    }

    if (isDefined(initialWeights)) {
      setWeights($(initialWeights))
    } else {
      setSeed($(seed))
    }
    val w = if (getWeights == null) {
      topology.model(_seed).weights
    } else {
      getWeights
    }

    val oldw: OldVector = OldVectors.fromML(w)
    val (newWeights, _, accTime) = optimizer.runMiniBatchSGD(data, dataStacker, oldw)

    val model = new ZFMultilayerPerceptronClassificationModel(uid, myLayers, newWeights.asML)
    model.setOtherTime(accTime)
    model
  }
}

@Since("2.0.0")
object ZFMLPC
  extends DefaultParamsReadable[ZFMLPC] {


  private[classification] val LBFGS = "l-bfgs"


  private[classification] val GD = "gd"


  private[classification] val supportedSolvers = Array(LBFGS, GD)

  @Since("2.0.0")
  override def load(path: String): ZFMLPC = super.load(path)
}


@Since("1.5.0")
@Experimental
class ZFMultilayerPerceptronClassificationModel(
                                                 @Since("1.5.0") override val uid: String,
                                                 @Since("1.5.0") val layers: Array[Int],
                                                 @Since("2.0.0") val weights: Vector
                                               )
  extends PredictionModel[Vector, ZFMultilayerPerceptronClassificationModel]
    with Serializable with MLWritable {

  var zfOtherTime = 0L

  def setOtherTime(otherT: Long): Unit = {
    zfOtherTime = otherT
  }

  @Since("1.6.0")
  override val numFeatures: Int = layers.head

  private val mlpModel = FeedForwardTopology
    .multiLayerPerceptron(layers, softmaxOnTop = true)
    .model(weights)


  private[ml] def javaLayers: java.util.List[Int] = {
    layers.toList.asJava
  }


  override protected def predict(features: Vector): Double = {
    ZFLabelConverter.decodeLabel(mlpModel.predict(features))
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): ZFMultilayerPerceptronClassificationModel = {
    copyValues(new ZFMultilayerPerceptronClassificationModel(uid, layers, weights), extra)
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new ZFMultilayerPerceptronClassificationModel.MultilayerPerceptronClassificationModelWriter(this)
}

@Since("2.0.0")
object ZFMultilayerPerceptronClassificationModel
  extends MLReadable[ZFMultilayerPerceptronClassificationModel] {

  @Since("2.0.0")
  override def read: MLReader[ZFMultilayerPerceptronClassificationModel] =
    new MultilayerPerceptronClassificationModelReader

  @Since("2.0.0")
  override def load(path: String): ZFMultilayerPerceptronClassificationModel = super.load(path)


  private[ZFMultilayerPerceptronClassificationModel]
  class MultilayerPerceptronClassificationModelWriter(
                                                       instance: ZFMultilayerPerceptronClassificationModel) extends MLWriter {

    private case class Data(layers: Array[Int], weights: Vector)

    override protected def saveImpl(path: String): Unit = {

      DefaultParamsWriter.saveMetadata(instance, path, sc)

      val data = Data(instance.layers, instance.weights)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class MultilayerPerceptronClassificationModelReader
    extends MLReader[ZFMultilayerPerceptronClassificationModel] {


    private val className = classOf[ZFMultilayerPerceptronClassificationModel].getName

    override def load(path: String): ZFMultilayerPerceptronClassificationModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("layers", "weights").head()
      val layers = data.getAs[Seq[Int]](0).toArray
      val weights = data.getAs[Vector](1)
      val model = new ZFMultilayerPerceptronClassificationModel(metadata.uid, layers, weights)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

}
