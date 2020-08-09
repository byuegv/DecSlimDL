

package org.apache.spark.ml.ann

import java.util.Random

import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, axpy => Baxpy}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.{CollectionAccumulator, LongAccumulator}


private[ann] trait ZFLayer extends Serializable {


  val weightSize: Int

  def getOutputSize(inputSize: Int): Int


  val inPlace: Boolean


  def createModel(initialWeights: BDV[Double]): ZFLayerModel


  def initModel(weights: BDV[Double], random: Random): ZFLayerModel
}


private[ann] trait ZFLayerModel extends Serializable {

  val weights: BDV[Double]

  def eval(data: BDM[Double], output: BDM[Double]): Unit


  def computePrevDelta(delta: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit


  def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit
}


private[ann] class ZFAffineLayer(val numIn: Int, val numOut: Int) extends ZFLayer {

  override val weightSize = numIn * numOut + numOut

  override def getOutputSize(inputSize: Int): Int = numOut

  override val inPlace = false

  override def createModel(weights: BDV[Double]): ZFLayerModel = new ZFAffineLayerModel(weights, this)

  override def initModel(weights: BDV[Double], random: Random): ZFLayerModel =
    ZFAffineLayerModel(this, weights, random)
}


private[ann] class ZFAffineLayerModel private[ann](
                                                    val weights: BDV[Double],
                                                    val layer: ZFAffineLayer) extends ZFLayerModel {
  val w = new BDM[Double](layer.numOut, layer.numIn, weights.data, weights.offset)
  val b =
    new BDV[Double](weights.data, weights.offset + (layer.numOut * layer.numIn), 1, layer.numOut)

  private var ones: BDV[Double] = null

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    output(::, *) := b
    BreezeUtil.dgemm(1.0, w, data, 1.0, output)
  }

  override def computePrevDelta(
                                 delta: BDM[Double],
                                 output: BDM[Double],
                                 prevDelta: BDM[Double]): Unit = {
    BreezeUtil.dgemm(1.0, w.t, delta, 0.0, prevDelta)
  }

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {

    val cumGradientOfWeights = new BDM[Double](w.rows, w.cols, cumGrad.data, cumGrad.offset)
    BreezeUtil.dgemm(1.0 / input.cols, delta, input.t, 1.0, cumGradientOfWeights)
    if (ones == null || ones.length != delta.cols) ones = BDV.ones[Double](delta.cols)

    val cumGradientOfBias = new BDV[Double](cumGrad.data, cumGrad.offset + w.size, 1, b.length)
    BreezeUtil.dgemv(1.0 / input.cols, delta, ones, 1.0, cumGradientOfBias)
  }
}


private[ann] object ZFAffineLayerModel {


  def apply(layer: ZFAffineLayer, weights: BDV[Double], random: Random): ZFAffineLayerModel = {
    randomWeights(layer.numIn, layer.numOut, weights, random)
    new ZFAffineLayerModel(weights, layer)
  }


  def randomWeights(
                     numIn: Int,
                     numOut: Int,
                     weights: BDV[Double],
                     random: Random): Unit = {
    var i = 0
    val sqrtIn = math.sqrt(numIn)
    while (i < weights.length) {
      weights(i) = (random.nextDouble * 4.8 - 2.4) / sqrtIn
      i += 1
    }
  }
}


private[ann] trait ZFActivationFunction extends Serializable {


  def eval: Double => Double


  def derivative: Double => Double
}


private[ann] object ZFApplyInPlace {


  def apply(x: BDM[Double], y: BDM[Double], func: Double => Double): Unit = {
    var i = 0
    while (i < x.rows) {
      var j = 0
      while (j < x.cols) {
        y(i, j) = func(x(i, j))
        j += 1
      }
      i += 1
    }
  }


  def apply(
             x1: BDM[Double],
             x2: BDM[Double],
             y: BDM[Double],
             func: (Double, Double) => Double): Unit = {
    var i = 0
    while (i < x1.rows) {
      var j = 0
      while (j < x1.cols) {
        y(i, j) = func(x1(i, j), x2(i, j))
        j += 1
      }
      i += 1
    }
  }
}


private[ann] class ZFSigmoidFunction extends ZFActivationFunction {

  override def eval: (Double) => Double = x => 1.0 / (1 + math.exp(-x))

  override def derivative: (Double) => Double = z => (1 - z) * z
}


private[ann] class ZFFunctionalLayer(val activationFunction: ZFActivationFunction) extends ZFLayer {

  override val weightSize = 0

  override def getOutputSize(inputSize: Int): Int = inputSize

  override val inPlace = true

  override def createModel(weights: BDV[Double]): ZFLayerModel = new ZFFunctionalLayerModel(this)

  override def initModel(weights: BDV[Double], random: Random): ZFLayerModel =
    createModel(weights)
}


private[ann] class ZFFunctionalLayerModel private[ann](val layer: ZFFunctionalLayer)
  extends ZFLayerModel {


  val weights = new BDV[Double](0)

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    ZFApplyInPlace(data, output, layer.activationFunction.eval)
  }

  override def computePrevDelta(
                                 nextDelta: BDM[Double],
                                 input: BDM[Double],
                                 delta: BDM[Double]): Unit = {
    ZFApplyInPlace(input, delta, layer.activationFunction.derivative)
    delta :*= nextDelta
  }

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {}
}


private[ann] class ZFFeedForwardTopology private(val layers: Array[ZFLayer]) extends Serializable {
  def model(weights: Vector): ZFFeedForwardModel = ZFFeedForwardModel(this, weights)

  def model(seed: Long): ZFFeedForwardModel = ZFFeedForwardModel(this, seed)
}


private[ml] object ZFFeedForwardTopology {

  def apply(layers: Array[ZFLayer]): ZFFeedForwardTopology = {
    new ZFFeedForwardTopology(layers)
  }


  def multiLayerPerceptron(
                            layerSizes: Array[Int],
                            softmaxOnTop: Boolean = true): ZFFeedForwardTopology = {
    val layers = new Array[ZFLayer]((layerSizes.length - 1) * 2)
    for (i <- 0 until layerSizes.length - 1) {
      layers(i * 2) = new ZFAffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (i == layerSizes.length - 2) {
          if (softmaxOnTop) {
            new ZFSoftmaxLayerWithCrossEntropyLoss()
          } else {
            new ZFSigmoidLayerWithSquaredError()
          }
        } else {
          new ZFFunctionalLayer(new ZFSigmoidFunction())
        }
    }
    ZFFeedForwardTopology(layers)
  }
}


private[ml] class ZFFeedForwardModel private(
                                              val weights: Vector,
                                              val topology: ZFFeedForwardTopology) {

  val layers = topology.layers
  val layerModels = new Array[ZFLayerModel](layers.length)
  private var offset = 0
  for (i <- 0 until layers.length) {
    layerModels(i) = layers(i).createModel(
      new BDV[Double](weights.toArray, offset, 1, layers(i).weightSize))
    offset += layers(i).weightSize
  }
  private var outputs: Array[BDM[Double]] = null
  private var deltas: Array[BDM[Double]] = null

  def forward(data: BDM[Double]): Array[BDM[Double]] = {

    val currentBatchSize = data.cols

    if (outputs == null || outputs(0).cols != currentBatchSize) {
      outputs = new Array[BDM[Double]](layers.length)
      var inputSize = data.rows
      for (i <- 0 until layers.length) {
        if (layers(i).inPlace) {
          outputs(i) = outputs(i - 1)
        } else {
          val outputSize = layers(i).getOutputSize(inputSize)
          outputs(i) = new BDM[Double](outputSize, currentBatchSize)
          inputSize = outputSize
        }
      }
    }
    layerModels(0).eval(data, outputs(0))
    for (i <- 1 until layerModels.length) {
      layerModels(i).eval(outputs(i - 1), outputs(i))
    }
    outputs
  }

  def computeGradient(
                       data: BDM[Double],
                       target: BDM[Double],
                       cumGradient: Vector,
                       realBatchSize: Int,
                       grads: CollectionAccumulator[Double] = null): Double = {
    val outputs = forward(data)
    val currentBatchSize = data.cols

    if (deltas == null || deltas(0).cols != currentBatchSize) {
      deltas = new Array[BDM[Double]](layerModels.length)
      var inputSize = data.rows
      for (i <- 0 until layerModels.length - 1) {
        val outputSize = layers(i).getOutputSize(inputSize)
        deltas(i) = new BDM[Double](outputSize, currentBatchSize)
        inputSize = outputSize
      }
    }
    val L = layerModels.length - 1

    val loss = layerModels.last match {
      case levelWithError: ZFLossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1))
      case _ =>
        throw new UnsupportedOperationException("Top layer is required to have objective.")
    }
    for (i <- (L - 2) to(0, -1)) {
      layerModels(i + 1).computePrevDelta(deltas(i + 1), outputs(i + 1), deltas(i))
    }

    val cumGradientArray = cumGradient.toArray
    var offset = 0
    for (i <- 0 until layerModels.length) {
      val input = if (i == 0) data else outputs(i - 1)
      layerModels(i).grad(deltas(i), input,
        new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize))

      if (grads != null) {
        if (i == 0) {
          val grad_layer_0 = new BDV[Double](layers(i).weightSize)
          layerModels(i).grad(deltas(i), input, grad_layer_0)
          val c: Double = grad_layer_0 dot grad_layer_0
          grads.add(c)
        }
      }
      offset += layers(i).weightSize

    }
    loss
  }

  def computeZipGradient(
                          data: BDM[Double],
                          target: BDM[Double],
                          //                          cumGradient: Vector,
                          realBatchSize: Int): Double = {
    val outputs = forward(data)
    val currentBatchSize = data.cols

    if (deltas == null || deltas(0).cols != currentBatchSize) {
      deltas = new Array[BDM[Double]](layerModels.length)
      var inputSize = data.rows
      for (i <- 0 until layerModels.length - 1) {
        val outputSize = layers(i).getOutputSize(inputSize)
        deltas(i) = new BDM[Double](outputSize, currentBatchSize)
        inputSize = outputSize
      }
    }
    val L = layerModels.length - 1

    val loss = layerModels.last match {
      case levelWithError: ZFLossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1))
      case _ =>
        throw new UnsupportedOperationException("Top layer is required to have objective.")
    }
    for (i <- (L - 2) to(0, -1)) {
      layerModels(i + 1).computePrevDelta(deltas(i + 1), outputs(i + 1), deltas(i))
    }
    //    val cumGradientArray = cumGradient.toArray

    var offset = 0
    var c = 0.0
    for (i <- 0 until 1) {
      val input = if (i == 0) data else outputs(i - 1)
      //      layerModels(i).grad(deltas(i), input, new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize))
      val grad_layer_0 = new BDV[Double](layers(i).weightSize)
      layerModels(i).grad(deltas(i), input, grad_layer_0)
      c = grad_layer_0 dot grad_layer_0
      offset += layers(i).weightSize
    }
    //        loss
    c
  }

  def predict(data: Vector): Vector = {
    val size = data.size
    val result = forward(new BDM[Double](size, 1, data.toArray))
    Vectors.dense(result.last.toArray)
  }
}


private[ann] object ZFFeedForwardModel {


  def apply(topology: ZFFeedForwardTopology, weights: Vector): ZFFeedForwardModel = {

    new ZFFeedForwardModel(weights, topology)
  }


  def apply(topology: ZFFeedForwardTopology, seed: Long = 11L): ZFFeedForwardModel = {
    val layers = topology.layers
    val layerModels = new Array[ZFLayerModel](layers.length)
    var totalSize = 0
    for (i <- 0 until topology.layers.length) {
      totalSize += topology.layers(i).weightSize
    }
    val weights = BDV.zeros[Double](totalSize)
    var offset = 0
    val random = new XORShiftRandom(seed)
    for (i <- 0 until layers.length) {
      layerModels(i) = layers(i).
        initModel(new BDV[Double](weights.data, offset, 1, layers(i).weightSize), random)
      offset += layers(i).weightSize
    }
    new ZFFeedForwardModel(Vectors.fromBreeze(weights), topology)
  }
}


class ZFANNGradient(ziptopology: ZFFeedForwardTopology, topology: ZFFeedForwardTopology, dataStacker: ZFDataStacker) extends Serializable {
  def compute(
               data: OldVector,
               label: Double,
               weights: OldVector,
               cumGradient: OldVector
//               t0: LongAccumulator = null, t1: LongAccumulator = null, t2: LongAccumulator = null, grads: CollectionAccumulator[Double] = null
             ): Double = {
//    val c0 = System.currentTimeMillis()
    val (input, target, realBatchSize) = dataStacker.unstack(data)
//    t0.add(System.currentTimeMillis() - c0)

//    val c1 = System.currentTimeMillis()
    val model = topology.model(weights)
//    t1.add(System.currentTimeMillis() - c1)

//    val c2 = System.currentTimeMillis()
    val r = model.computeGradient(input, target, cumGradient, realBatchSize)
//    t2.add(System.currentTimeMillis() - c2)
    r
  }

  def computeZipGrad(
                      data: OldVector,
                      //               label: Double,
                      weights: OldVector
                      //                      cumGradient: OldVector,
                      ): Double = {
//    val c0 = System.currentTimeMillis()
    val (input, target, realBatchSize) = dataStacker.unstack(data)
//    t0.add(System.currentTimeMillis() - c0)

//    val c1 = System.currentTimeMillis()
    val model = ziptopology.model(weights)
//    t1.add(System.currentTimeMillis() - c1)

//    val c2 = System.currentTimeMillis()
    val r = model.computeZipGradient(input, target, realBatchSize)
//    t2.add(System.currentTimeMillis() - c2)
    r
  }

}


class ZFDataStacker(stacksize: Int, inputSize: Int, outputSize: Int)
  extends Serializable {

  var stackSize = stacksize

  def stack(data: RDD[(Vector, Vector)]): RDD[(Double, Vector)] = {
    val stackedData = if (stackSize == 1) {
      data.map { v =>
        (0.0,
          Vectors.fromBreeze(BDV.vertcat(
            v._1.asBreeze.toDenseVector,
            v._2.asBreeze.toDenseVector))
          )
      }
    } else {
      data.mapPartitions { it =>
        it.grouped(stackSize).map { seq =>
          val size = seq.size
          val bigVector = new Array[Double](inputSize * size + outputSize * size)
          var i = 0
          seq.foreach { case (in, out) =>
            System.arraycopy(in.toArray, 0, bigVector, i * inputSize, inputSize)
            System.arraycopy(out.toArray, 0, bigVector,
              inputSize * size + i * outputSize, outputSize)
            i += 1
          }
          (0.0, Vectors.dense(bigVector))
        }
      }
    }
    stackedData
  }

  def zfStack(it: Iterator[(Vector, OldVector)]): Iterator[(Double, OldVector)] = {
    val stackedData = if (stackSize == 1) {
      it.map { v =>
        (0.0,
          OldVectors.fromBreeze(BDV.vertcat(
            v._1.asBreeze.toDenseVector,
            v._2.asBreeze.toDenseVector))
          )
      }
    } else {
      it.grouped(stackSize).map { seq =>
        val size = seq.size
        val bigVector = new Array[Double](inputSize * size + outputSize * size)
        var i = 0
        seq.foreach { case (in, out) =>
          System.arraycopy(in.toArray, 0, bigVector, i * inputSize, inputSize)
          System.arraycopy(out.toArray, 0, bigVector,
            inputSize * size + i * outputSize, outputSize)
          i += 1
        }
        (0.0, OldVectors.dense(bigVector))
      }
    }
    stackedData
  }


  def unstack(data: Vector): (BDM[Double], BDM[Double], Int) = {
    val arrData = data.toArray
    val realStackSize = arrData.length / (inputSize + outputSize)
    val input = new BDM(inputSize, realStackSize, arrData)
    val target = new BDM(outputSize, realStackSize, arrData, inputSize * realStackSize)
    (input, target, realStackSize)
  }
}


class ZFANNUpdater extends ZFUpdater {

  override def compute(
                        weightsOld: OldVector,
                        gradient: OldVector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (OldVector, Double) = {
    val thisIterStepSize = stepSize
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    Baxpy(-thisIterStepSize, gradient.asBreeze, brzWeights)
    (OldVectors.fromBreeze(brzWeights), 0)
  }
}


//private[ml] class ZFFeedForwardTrainer(
//                                        topology: ZFFeedForwardTopology,
//                                        val inputSize: Int,
//                                        val outputSize: Int,
//                                        val layers: Array[Int],
//                                        val testset: Dataset[_]) extends Serializable {
//
//  private var _seed = this.getClass.getName.hashCode.toLong
//  private var _weights: Vector = null
//  private var _stackSize = 128
//  private var dataStacker = new ZFDataStacker(_stackSize, inputSize, outputSize)
//  private var _gradient: ZFANNGradient = new ZFANNGradient(topology, dataStacker)
//  private var _updater: ZFUpdater = new ZFANNUpdater()
//
//
//  private var optimizer: ZFGradientDescent = new ZFGradientDescent(_gradient, _updater, layers, testset) //LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(100)
//
//
//  def getSeed: Long = _seed
//
//
//  def setSeed(value: Long): this.type = {
//    _seed = value
//    this
//  }
//
//
//  def getWeights: Vector = _weights
//
//
//  def setWeights(value: Vector): this.type = {
//    _weights = value
//    this
//  }
//
//
//  def setStackSize(value: Int): this.type = {
//    _stackSize = value
//    dataStacker = new ZFDataStacker(value, inputSize, outputSize)
//    this
//  }
//
//  def SGDOptimizer: ZFGradientDescent = {
//    val sgd = new ZFGradientDescent(_gradient, _updater, layers, testset)
//    optimizer = sgd
//    sgd
//  }
//
//
//  def train(data: RDD[(Vector, Vector)]): ZFFeedForwardModel = {
//    val w = if (getWeights == null) {
//      topology.model(_seed).weights
//    } else {
//      getWeights
//    }
//
//    val newWeights = optimizer.optimize(dataStacker.stack(data).map { v =>
//      (v._1, OldVectors.fromML(v._2))
//    }, w)
//    topology.model(newWeights)
//  }
//
//}
