package org.apache.spark.ml.classification

import AccurateML.lsh.ZFHashLayer
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.ann.{ZFANNGradient, ZFANNUpdater, ZFDataStacker, ZFFeedForwardTopology}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.optimization.{ZFUpdater, ZFZipGradientDescent}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
  * Created by zhangfan on 18/4/25.
  */
@Since("1.5.0")
@Experimental
class ZFZipMLPC @Since("1.5.0")(testset: Dataset[_], zflayers: Array[Int], ratio: Double, upbound: Int, itqitN: Int, itqratioN: Int, redisHost: String,
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


  private var optimizer: ZFZipGradientDescent = new ZFZipGradientDescent(_gradient, _updater, zflayers, testset) //LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(100)


  def getWeights: Vector = _weights


  def setWeights(value: Vector): this.type = {
    _weights = value
    this
  }


  def SGDOptimizer: ZFZipGradientDescent = {
    val sgd = new ZFZipGradientDescent(_gradient, _updater, zflayers, testset)
    optimizer = sgd
    sgd
  }


  @Since("1.5.0")
  def setLayers(value: Array[Int]): this.type = set(layers, value)


  @Since("1.5.0")
  def setBlockSize(value: Int): this.type = set(blockSize, value)


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
  override def copy(extra: ParamMap): ZFZipMLPC = defaultCopy(extra)

  protected def extractLabeledPoints(dataset: Dataset[_]): RDD[LabeledPoint] = {
    dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
  }

  //  protected def extractOldLabeledPoints(dataset: Dataset[_]): RDD[OldLabeledPoint] = {
  //    dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
  //      case Row(label: Double, features: Vector) => new OldLabeledPoint(label, OldVectors.fromML(features))
  //    }
  //  }


  def train(dataset: Dataset[_]): ZFMultilayerPerceptronClassificationModel = {
    val myLayers = zflayers
    //    val lpData = extractLabeledPoints(dataset)
    val lpData = extractLabeledPoints(dataset)
    println("oldLpData.partN, " + lpData.partitions.size)
    val hashT = System.currentTimeMillis()
    val hash = new ZFHashLayer(itqitN, itqratioN, upbound, false, 0, redisHost, lpData.sparkContext, zflayers.last)
    val zipObjectData: RDD[(OldVector, Array[LabeledPoint])] = lpData.map(lp => (lp.label, lp)).groupByKey().flatMap(t => {
      hash.zfHashMapMLPC(t._2)
    }).persist(StorageLevel.MEMORY_AND_DISK)
    //    val zipObjectData: RDD[(OldVector, Array[LabeledPoint])] = lpData.mapPartitions(hash.zfHashMapMLPC).persist(StorageLevel.MEMORY_ONLY)
    zipObjectData.count()
    println("hashTime, " + (System.currentTimeMillis() - hashT))

    //    val data = lpData.map(lp => ZFLabelConverter.encodeLabeledPoint(lp, labels))


    if ($(solver) == ZFZipMLPC.GD) {
      val sgd = new ZFZipGradientDescent(_gradient, _updater, zflayers, testset)
      optimizer = sgd
      sgd
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by ZFZipMLPC.")
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
    val (newWeights, _, accTime) = optimizer.runMiniBatchSGD(zipObjectData, dataStacker, oldw, ratio)

    val model = new ZFMultilayerPerceptronClassificationModel(uid, myLayers, newWeights.asML)
    model.setOtherTime(accTime)
    model
  }
}

@Since("2.0.0")
object ZFZipMLPC
  extends DefaultParamsReadable[ZFZipMLPC] {


  private[classification] val LBFGS = "l-bfgs"


  private[classification] val GD = "gd"


  private[classification] val supportedSolvers = Array(LBFGS, GD)

  @Since("2.0.0")
  override def load(path: String): ZFZipMLPC = super.load(path)
}
