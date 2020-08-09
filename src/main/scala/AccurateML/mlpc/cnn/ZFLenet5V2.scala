/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.neuralNetwork

import java.util.List

import AccurateML.mlpc.cnn.ZFFullConnection
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.sigmoid
import org.apache.log4j.{Level, Logger}
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer


/**
  * Convolution neural network
  */
class ZFLenet5V2(layerBuilder: CNNTopology, hiddenN: Int, outN: Int, inN: Int = 1) extends Serializable with Logging {
  private var ALPHA: Double = 0.85
  private var layers: List[CNNLayer] = null
  private var layerNum: Int = 0
  private var maxIterations = 10
  private var batchSize = 100
  val inputN: Int = inN


  val weights1 = new BDM[Double](outN, hiddenN)
  val bais1 = new BDM[Double](outN, 1)
  ZFFullConnection.initWeights(weights1, bais1)
  layers = layerBuilder.mLayers
  layerNum = layers.size
  setup

  def setup {
    var i: Int = 1
    while (i < layers.size) {
      val layer: CNNLayer = layers.get(i)
      val frontLayer: CNNLayer = layers.get(i - 1)
      val frontMapNum: Int = frontLayer.getOutMapNum
      layer.getType match {
        case "input" =>
        case "conv" =>
          val convLayer = layer.asInstanceOf[ConvCNNLayer]
          convLayer.setMapSize(frontLayer.getMapSize.subtract(convLayer.getKernelSize, 1))
          convLayer.initKernel(frontMapNum)
          convLayer.initBias(frontMapNum)
        case "samp" =>
          val sampLayer = layer.asInstanceOf[SampCNNLayer]
          sampLayer.setOutMapNum(frontMapNum)
          sampLayer.setMapSize(frontLayer.getMapSize.divide(sampLayer.getScaleSize))
        case "output" =>
          val outputLayer = layer.asInstanceOf[OutputCNNLayer]
          outputLayer.initOutputKernels(frontMapNum, frontLayer.getMapSize)
          outputLayer.asInstanceOf[OutputCNNLayer].initBias(frontMapNum)
      }
      i += 1
    }
  }

  def setMiniBatchSize(batchSize: Int): this.type = {
    this.batchSize = batchSize
    this
  }

  /**
    * Maximum number of iterations for learning.
    */
  def getMaxIterations: Int = maxIterations

  /**
    * Maximum number of iterations for learning.
    * (default = 20)
    */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  //  def trainOneByOne(trainSet: RDD[LabeledPoint]) {
  //    var t = 0
  //    val trainSize = trainSet.count().toInt
  //    val dataArr = trainSet.collect()
  //    while (t < maxIterations) {
  //      val epochsNum = trainSize
  //      var right = 0
  //      var count = 0
  //      var i = 0
  //      while (i < epochsNum) {
  //        val record = dataArr(i)
  //        val result = train(record)
  //        if (result._1) right += 1
  //        count += 1
  //        val gradient: Array[(Array[Array[BDM[Double]]], Array[Double])] = result._2
  //        updateParams(gradient, 1)
  //        i += 1
  //      }
  //      val p = 1.0 * right / count
  //      if (t % 10 == 1 && p > 0.96) {
  //        ALPHA = 0.001 + ALPHA * 0.9
  //      }
  //      t += 1
  //      logInfo(s"precision $right/$count = $p")
  //      println(s"precision $right/$count = $p")
  //    }
  //  }

  def train(trainSet: RDD[LabeledPoint], testSet: RDD[LabeledPoint], foldN: Int) {
    var t = 1
    val trainSize = trainSet.count().toInt
    val temp = train(trainSet.first)
    val gZero = temp._2
    gZero.foreach(tu => if (tu != null) {
      tu._1.foreach(m => m.foreach(x => x -= x))
      (0 until tu._2.length).foreach(i => tu._2(i) = 0)
    }
    )
    val fullZero: Array[BDM[Double]] = temp._3
    fullZero.foreach(x => x -= x)

    var totalCount = 0
    var totalRight = 0
    var accitTime = 0L



    while (t <= maxIterations) {
      val splits = trainSet.randomSplit(Array.fill(foldN)(1.0 / foldN.toDouble))
      val gAcc = trainSet.sparkContext.collectionAccumulator[Double]
      for (f <- 0 until foldN) {
        val itTime = System.currentTimeMillis()
        val foldTrain = splits(f)
        val (gradientSum, right, count, fullGradSum) = foldTrain
          //          .sample(false, batchSize.toDouble / trainSize, 42 + t)
          .treeAggregate((gZero, 0, 0, fullZero))(
          seqOp = (c, v) => {
            val result = train(v)
            val gradient = result._2
            val fullGradient: Array[BDM[Double]] = result._3

            //g
            val g1 = gradient(1)._1.map(ms => ms.map(m => m.toArray.map(a => a * a).sum).sum).sum
            gAcc.add(g1)
            //end-g

            val right = if (result._1) 1 else 0
            (ZFLenet5V2.combineGradient(c._1, gradient), c._2 + right, c._3 + 1, c._4.zip(fullGradient).map(t => t._1 + t._2))
          },
          combOp = (c1, c2) => {
            (ZFLenet5V2.combineGradient(c1._1, c2._1), c1._2 + c2._2, c1._3 + c2._3, c1._4.zip(c2._4).map(t => t._1 + t._2))
          })

        if (count > 0) {
          batchSize = count
          updateParams(gradientSum, count)
          weights1 += fullGradSum(0) / count.toDouble
          bais1 += fullGradSum(1) / count.toDouble
          val p = 1.0 * totalRight / totalCount
          if (t % 10 == 1 && p > 0.96) {
            ALPHA = 0.001 + ALPHA * 0.9
          }
          totalCount += count
          totalRight += right
          if (t % 10 == 0) {
            //            println(s"precision $totalRight/$totalCount = $p")
            //          totalCount = 0
            //          totalRight = 0
          }
        }
        accitTime += (System.currentTimeMillis() - itTime)
        //      println(t + ",itTime," + accitTime + ",pointsN," + totalCount)

      } //end-foldN

      if (t % 1 == 0) {
        val c = gAcc.value.sortWith(_ < _)
        val mean10 = c.sum.toDouble / c.size
        val ca = new Array[Int](10)
        val cinterval = (c.last - c(0)) / 10
        c.foreach(v => {
          val index = if (v == c.last) 9 else (v - c(0)) / cinterval
          ca(index.toInt) += 1
        })
        val less10 = c.filter(_ < mean10).size
        val ACC = predictFullCollection(testSet)
        println(t + ",ACC," + ACC + ",itTime," + accitTime + ",pointsN," + totalCount + "    ,," + ca.mkString(",") + " ,less mean," + less10.toDouble / c.size + ",min-max" +
          "," + c(0) + "," + c.last)
        gAcc.reset()
      }
      //      else {
      //        val c = gAcc.value.sortWith(_ < _)
      //        val mean10 = c.sum.toDouble / c.size / 10
      //        val ca = new Array[Int](10)
      //        val cinterval = (c.last - c(0)) / 10
      //        c.foreach(v => {
      //          val index = if (v == c.last) 9 else (v - c(0)) / cinterval
      //          ca(index.toInt) += 1
      //        })
      //        val less10 = c.filter(_ < mean10).size
      //        println(t + ",gradient," + ca.mkString(",") + " ,less mean/10," + less10.toDouble / c.size)
      //      }
      t += 1

    }
  }

  //  def predict(testSet: RDD[Vector]): RDD[Int] = {
  //    testSet.map(record => {
  //      val outputs: Array[Array[BDM[Double]]] = forward(record)
  //      val outputLayer = layers.get(layerNum - 1)
  //      val mapNum = outputLayer.getOutMapNum
  //      val out = new Array[Double](mapNum)
  //      for (m <- 0 until mapNum) {
  //        val outMap = outputs(layerNum - 1)(m)
  //        out(m) = outMap(0, 0)
  //      }
  //      ZFLenet5V2.getMaxIndex(out)
  //    })
  //
  //  }

  def predictFullCollection(testSet: RDD[LabeledPoint]): Double = {
    val predicts = testSet.map(_.features).map(record => {
      val outputs = forward(record)
      //zf add full connection
      val lastOut = new BDM[Double](outputs.last.size, 1, outputs.last.map(m => m(0, 0)))
      val out2 = ZFFullConnection.forwardout(lastOut, weights1, bais1)
      val index = ZFLenet5V2.getMaxIndex(out2.toArray)
      index
    })
    val rightN = predicts.zip(testSet.map(_.label)).map(t => if (t._1 == t._2) 1 else 0).sum()
    val acc = rightN.toDouble / testSet.count()
    acc
  }

  def zfsigmoid(x: Double): Double = {
    1 / (1 + math.exp(-x))
  }

  /**
    * train one record
    *
    * @return (isRight, gradient)
    */
  private def train(record: LabeledPoint): (Boolean, Array[(Array[Array[BDM[Double]]], Array[Double])], Array[BDM[Double]]) = {
    val outputs = forward(record.features)

    //zf add full connection
    val lastOut = new BDM[Double](outputs.last.size, 1, outputs.last.map(m => m(0, 0)))
    val out2 = ZFFullConnection.forwardout(lastOut, weights1, bais1)
    val c = new Array[Double](outN)
    c(record.label.toInt) = 1
    val label = new BDM[Double](outN, 1, c)

    val delta2 = label - out2
    val grad1: BDM[Double] = delta2 * lastOut.t
    val gbais1: BDM[Double] = delta2.copy

    val right = if (ZFLenet5V2.getMaxIndex(out2.toArray) == ZFLenet5V2.getMaxIndex(label.toArray)) true else false
    val delta1 = ZFFullConnection.getDelta(lastOut, weights1, delta2)
    val errors = backPropagation(delta1.toArray.map(d => new BDM[Double](1, 1, Array(d))), outputs)
    val gradient = getGradient(outputs, errors)
    (right, gradient, Array(grad1, gbais1))
  }

  /**
    * forward for one record
    *
    * @param record
    */
  private def forward(record: Vector): Array[Array[BDM[Double]]] = {

    val outputs = new Array[Array[BDM[Double]]](layers.size)
    outputs(0) = setInLayerOutput(record)
    var l: Int = 1
    while (l < layers.size) {
      val layer: CNNLayer = layers.get(l)
      outputs(l) =
        layer.getType match {
          case "conv" =>
            setConvOutput(layer.asInstanceOf[ConvCNNLayer], outputs(l - 1))
          case "samp" =>
            setSampOutput(layer.asInstanceOf[SampCNNLayer], outputs(l - 1))
          case "output" =>
            setConvOutput(layer.asInstanceOf[ConvCNNLayer], outputs(l - 1))
          case _ => null
        }
      l += 1
    }
    outputs
  }

  /**
    * run BP and get errors for all layers
    *
    * @return (right, errors for all layers)
    */
  private def backPropagation(
                               outError: Array[BDM[Double]],
                               outputs: Array[Array[BDM[Double]]]): Array[Array[BDM[Double]]] = {
    val errors = new Array[Array[BDM[Double]]](layers.size)
    //    val result1 = setOutLayerErrors(record, outputs(layerNum - 1))
    errors(layerNum - 1) = outError
    var l: Int = layerNum - 2
    while (l > 0) {
      val layer: CNNLayer = layers.get(l)
      val nextLayer: CNNLayer = layers.get(l + 1)
      errors(l) = layer.getType match {
        case "samp" =>
          setSampErrors(layer, nextLayer.asInstanceOf[ConvCNNLayer], errors(l + 1))
        case "conv" =>
          setConvErrors(layer, nextLayer.asInstanceOf[SampCNNLayer], errors(l + 1), outputs(l))
        case _ => null
      }
      l -= 1
    }
    errors
  }


  private def getGradient(
                           outputs: Array[Array[BDM[Double]]],
                           errors: Array[Array[BDM[Double]]]): Array[(Array[Array[BDM[Double]]], Array[Double])] = {
    var l: Int = 1
    val gradient = new Array[(Array[Array[BDM[Double]]], Array[Double])](layerNum)
    while (l < layerNum) {
      val layer: CNNLayer = layers.get(l)
      val lastLayer: CNNLayer = layers.get(l - 1)
      gradient(l) = layer.getType match {
        case "conv" =>
          val kernelGradient = getKernelsGradient(layer, lastLayer, errors(l), outputs(l - 1))
          val biasGradient = getBiasGradient(layer, errors(l))
          (kernelGradient, biasGradient)
        case "output" =>
          val kernelGradient = getKernelsGradient(layer, lastLayer, errors(l), outputs(l - 1))
          val biasGradient = getBiasGradient(layer, errors(l))
          (kernelGradient, biasGradient)
        case _ => null
      }
      l += 1
    }
    gradient
  }

  private def updateParams(
                            gradient: Array[(Array[Array[BDM[Double]]],
                              Array[Double])],
                            batchSize: Int): Unit = {
    var l: Int = 1
    while (l < layerNum) {
      val layer: CNNLayer = layers.get(l)
      layer.getType match {
        case "conv" =>
          updateKernels(layer.asInstanceOf[ConvCNNLayer], gradient(l)._1, batchSize)
          updateBias(layer.asInstanceOf[ConvCNNLayer], gradient(l)._2, batchSize)
        case "output" =>
          updateKernels(layer.asInstanceOf[ConvCNNLayer], gradient(l)._1, batchSize)
          updateBias(layer.asInstanceOf[ConvCNNLayer], gradient(l)._2, batchSize)
        case _ =>
      }
      l += 1
    }
  }

  private def updateKernels(
                             layer: ConvCNNLayer,
                             gradient: Array[Array[BDM[Double]]], batchSize: Int): Unit = {
    val len = gradient.length
    val width = gradient(0).length
    var j = 0
    while (j < width) {
      var i = 0
      while (i < len) {
        // update kernel
        val deltaKernel = gradient(i)(j) / batchSize.toDouble * ALPHA
        layer.getKernel(i, j) += deltaKernel
        i += 1
      }
      j += 1
    }
  }

  private def updateBias(layer: ConvCNNLayer, gradient: Array[Double], batchSize: Int): Unit = {
    val gv = new BDV[Double](gradient)
    layer.getBias += gv * ALPHA / batchSize.toDouble
  }

  /**
    * get bias gradient
    *
    * @param layer  layer to be updated
    * @param errors errors of this layer
    */
  private def getBiasGradient(layer: CNNLayer, errors: Array[BDM[Double]]): Array[Double] = {
    val mapNum: Int = layer.getOutMapNum
    var j: Int = 0
    val gradient = new Array[Double](mapNum)
    while (j < mapNum) {
      val error: BDM[Double] = errors(j)
      val deltaBias: Double = sum(error)
      gradient(j) = deltaBias
      j += 1
    }
    gradient
  }

  /**
    * get kernels gradient
    *
    * @param layer
    * @param lastLayer
    */
  private def getKernelsGradient(
                                  layer: CNNLayer,
                                  lastLayer: CNNLayer,
                                  layerError: Array[BDM[Double]],
                                  lastOutput: Array[BDM[Double]]): Array[Array[BDM[Double]]] = {
    val mapNum: Int = layer.getOutMapNum
    val lastMapNum: Int = lastLayer.getOutMapNum
    val delta = Array.ofDim[BDM[Double]](lastMapNum, mapNum)
    var j = 0
    while (j < mapNum) {
      var i = 0
      while (i < lastMapNum) {
        val error = layerError(j)
        val deltaKernel = ZFLenet5V2.convnValid(lastOutput(i), error)
        delta(i)(j) = deltaKernel
        i += 1
      }
      j += 1
    }
    delta
  }

  /**
    * set errors for sampling layer
    *
    * @param layer
    * @param nextLayer
    */
  private def setSampErrors(
                             layer: CNNLayer,
                             nextLayer: ConvCNNLayer,
                             nextLayerError: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.getOutMapNum
    val nextMapNum: Int = nextLayer.getOutMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var i = 0
    while (i < mapNum) {
      var sum: BDM[Double] = null // sum for every kernel
      var j = 0
      while (j < nextMapNum) {
        val nextError = nextLayerError(j)
        val kernel = nextLayer.getKernel(i, j)
        // rotate kernel by 180 degrees and get full convolution
        if (sum == null) {
          sum = ZFLenet5V2.convnFull(nextError, flipud(fliplr(kernel)))
        }
        else {
          sum += ZFLenet5V2.convnFull(nextError, flipud(fliplr(kernel)))
        }
        j += 1
      }
      errors(i) = sum
      i += 1
    }
    errors
  }

  /**
    * set errors for convolution layer
    *
    * @param layer
    * @param nextLayer
    */
  private def setConvErrors(
                             layer: CNNLayer,
                             nextLayer: SampCNNLayer,
                             nextLayerError: Array[BDM[Double]],
                             layerOutput: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.getOutMapNum
    val errors = new Array[BDM[Double]](mapNum)
    var m: Int = 0
    while (m < mapNum) {
      val scale: Scale = nextLayer.getScaleSize
      val nextError: BDM[Double] = nextLayerError(m)
      val map: BDM[Double] = layerOutput(m)
      var outMatrix: BDM[Double] = (1.0 - map)
      outMatrix = map :* outMatrix
      outMatrix = outMatrix :* ZFLenet5V2.kronecker(nextError, scale)
      errors(m) = outMatrix
      m += 1
    }
    errors
  }

  /**
    * set errors for output layer
    *
    * @param record
    * @return
    */
  private def setOutLayerErrors(
                                 record: LabeledPoint,
                                 output: Array[BDM[Double]]): (Boolean, Array[BDM[Double]]) = {
    val outputLayer: CNNLayer = layers.get(layerNum - 1)
    val mapNum: Int = outputLayer.getOutMapNum
    val target: Array[Double] = new Array[Double](mapNum)
    val outValues: Array[Double] = output.map(m => m(0, 0))

    val label = record.label.toInt
    target(label) = 1
    val layerError: Array[BDM[Double]] = (0 until mapNum).map(i => {
      val errorMatrix = new BDM[Double](1, 1)
      errorMatrix(0, 0) = outValues(i) * (1 - outValues(i)) * (target(i) - outValues(i))
      errorMatrix
    }).toArray
    val outClass = ZFLenet5V2.getMaxIndex(outValues)
    (label == outClass, layerError)
  }

  //  /**
  //    * set inlayer output
  //    *
  //    * @param record
  //    */
  //  private def setInLayerOutput(record: Vector): Array[BDM[Double]] = {
  //    val inputLayer: CNNLayer = layers.get(0)
  //    val mapSize = inputLayer.getMapSize
  //    if (record.size != mapSize.x * mapSize.y) {
  //      throw new RuntimeException("data size and map size mismatch!")
  //    }
  //    val m = new BDM[Double](mapSize.x, mapSize.y)
  //    var i: Int = 0
  //    while (i < mapSize.x) {
  //      var j: Int = 0
  //      while (j < mapSize.y) {
  //        m(i, j) = record(mapSize.x * i + j)
  //        j += 1
  //      }
  //      i += 1
  //    }
  //    Array(m)
  //  }

  private def setInLayerOutput(record: Vector): Array[BDM[Double]] = {
    val inputLayer: CNNLayer = layers.get(0)
    val mapSize = inputLayer.getMapSize
    if (record.size != mapSize.x * mapSize.y * inputN) {
      throw new RuntimeException("data size and map size mismatch!")
    }
    val ans = new ArrayBuffer[BDM[Double]]()
    for (in <- 0 until inputN) {
      val m = new BDM[Double](mapSize.x, mapSize.y)
      val offset = in * mapSize.x * mapSize.y
      var i: Int = 0
      while (i < mapSize.x) {
        var j: Int = 0
        while (j < mapSize.y) {
          m(i, j) = record(offset + mapSize.x * i + j)
          j += 1
        }
        i += 1
      }
      ans += m
    }
    ans.toArray
  }

  private def setConvOutput(
                             layer: ConvCNNLayer,
                             outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val mapNum: Int = layer.getOutMapNum
    val lastMapNum: Int = outputs.length
    val output = new Array[BDM[Double]](mapNum)
    var j = 0
    val oldBias = layer.getBias
    while (j < mapNum) {
      var sum: BDM[Double] = null
      var i = 0
      while (i < lastMapNum) {
        val lastMap = outputs(i)
        val kernel = layer.getKernel(i, j)
        if (sum == null) {
          sum = ZFLenet5V2.convnValid(lastMap, kernel)
        }
        else {
          sum += ZFLenet5V2.convnValid(lastMap, kernel)
        }
        i += 1
      }
      sum = sigmoid(sum + oldBias(j))
      output(j) = sum
      j += 1
    }
    output
  }

  /**
    * set output for sampling layer
    *
    * @param layer
    */
  private def setSampOutput(
                             layer: SampCNNLayer,
                             outputs: Array[BDM[Double]]): Array[BDM[Double]] = {
    val lastMapNum: Int = outputs.length
    val output = new Array[BDM[Double]](lastMapNum)
    var i: Int = 0
    while (i < lastMapNum) {
      val lastMap: BDM[Double] = outputs(i)
      val scaleSize: Scale = layer.getScaleSize
      output(i) = ZFLenet5V2.scaleMatrix(lastMap, scaleSize)
      i += 1
    }
    output
  }
}

/**
  * zf add one full-connection layer, acc in mnist is > 90%
  */
object ZFLenet5V2 {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("ZFLenet5V2")
    val sc = new SparkContext(conf)

    // /Users/zhangfan/Downloads/mnist/train.txt /Users/zhangfan/Downloads/mnist/test.txt 1000 60 6 784 true 1
    // /Users/zhangfan/Downloads/test_32x32.txt /Users/zhangfan/Downloads/test_32x32.txt 1000 60 6 3072 false 3
    val dataPath = args(0) //"/Users/zhangfan/Downloads/mnist/train.txt"
    val testPath = args(1) //"/Users/zhangfan/Downloads/mnist/test.txt"
    val numIt = args(2).toInt //5000
    val batchSize = args(3).toInt //16
    val minPart = args(4).toInt //6
    val numFeature = args(5).toInt
    val isSparse = args(6).toBoolean
    val inputN = args(7).toInt
    val foldN = args(8).toInt

    println("dataPath, " + dataPath + ",foldN, " + foldN + ",numIt," + numIt + ",batchSize," + batchSize + ",minPart," + minPart + ",numFeature," + numFeature + ",isSparse," + isSparse + ",inputN," + inputN)

    val data: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPart)
    } else {
      sc.textFile(dataPath, minPart).map(s => {
        val vs = s.split("\\s+|,").map(_.toDouble)
        new LabeledPoint(vs(0), Vectors.dense(vs.slice(1, vs.size)))
      })
    }

    val test: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPart)
    } else {
      sc.textFile(testPath, minPart).map(s => {
        val vs = s.split("\\s+|,").map(_.toDouble)
        new LabeledPoint(vs(0), Vectors.dense(vs.slice(1, vs.size)))
      })
    }
    val topology = new CNNTopology
    //    //    author little + 1-fullconnection-layer acc > 90%
    //    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
    //    topology.addLayer(CNNLayer.buildConvLayer(6, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //    topology.addLayer(CNNLayer.buildConvLayer(12, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //    topology.addLayer(CNNLayer.buildOutputLayer(10))
    //    val ZFLenet5V2: ZFLenet5V2 = new ZFLenet5V2(topology, 10, 10).setMaxIterations(numIt).setMiniBatchSize(batchSize)


    //    //    //lenet-5
    //    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
    //    topology.addLayer(CNNLayer.buildConvLayer(6, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //
    //    topology.addLayer(CNNLayer.buildConvLayer(16, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //    topology.addLayer(CNNLayer.buildConvLayer(120, new Scale(4, 4)))
    //    //    topology.addLayer(CNNLayer.buildOutputLayer(120)) //84,10
    //    val lenet5v2: ZFLenet5V2 = new ZFLenet5V2(topology, 120, 10, inputN).setMaxIterations(numIt).setMiniBatchSize(batchSize)

    //lenet-5
    //    topology.addLayer(CNNLayer.buildInputLayer(new Scale(32, 32), inputN))
    //    topology.addLayer(CNNLayer.buildConvLayer(6, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //    topology.addLayer(CNNLayer.buildConvLayer(16, new Scale(5, 5)))
    //    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
    //    topology.addLayer(CNNLayer.buildConvLayer(120, new Scale(5, 5)))
    //    //    topology.addLayer(CNNLayer.buildOutputLayer(120)) //84,10
    //    val lenet5v2: ZFLenet5V2 = new ZFLenet5V2(topology, 120, 10, inputN).setMaxIterations(numIt).setMiniBatchSize(batchSize)

    //    alex-next
//    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
//    topology.addLayer(CNNLayer.buildConvLayer(8, new Scale(11, 11)))
//    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))
//
//    topology.addLayer(CNNLayer.buildConvLayer(16, new Scale(5, 5)))
//    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))
//
//    topology.addLayer(CNNLayer.buildConvLayer(32, new Scale(3, 3)))
//    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
//
//    topology.addLayer(CNNLayer.buildConvLayer(48, new Scale(3, 3)))
//    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))
//
//    topology.addLayer(CNNLayer.buildConvLayer(32, new Scale(3, 3)))
//    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))
//
//    val lenet5v2: ZFLenet5V2 = new ZFLenet5V2(topology, 32, 10, inputN).setMaxIterations(numIt).setMiniBatchSize(batchSize)

    topology.addLayer(CNNLayer.buildInputLayer(new Scale(28, 28)))
    topology.addLayer(CNNLayer.buildConvLayer(8, new Scale(7, 7)))//8, new Scale(11, 11))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))

    topology.addLayer(CNNLayer.buildConvLayer(16, new Scale(5, 5)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))

    topology.addLayer(CNNLayer.buildConvLayer(32, new Scale(3, 3)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(2, 2)))

    topology.addLayer(CNNLayer.buildConvLayer(48, new Scale(3, 3)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(1, 1)))

    topology.addLayer(CNNLayer.buildConvLayer(32, new Scale(3, 3)))
    topology.addLayer(CNNLayer.buildSampLayer(new Scale(4, 4)))

    val lenet5v2: ZFLenet5V2 = new ZFLenet5V2(topology, 32, 10, inputN).setMaxIterations(numIt).setMiniBatchSize(batchSize)

    lenet5v2.train(data, test, foldN)
    val acc = lenet5v2.predictFullCollection(test)
    println("test ACC, " + acc)
  }

  private[neuralNetwork] def kronecker(matrix: BDM[Double], scale: Scale): BDM[Double] = {
    val ones = BDM.ones[Double](scale.x, scale.y)
    kron(matrix, ones)
  }

  /**
    * return a new matrix that has been scaled down
    *
    * @param matrix
    */
  private[neuralNetwork] def scaleMatrix(matrix: BDM[Double], scale: Scale): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val sm: Int = m / scale.x
    val sn: Int = n / scale.y
    val outMatrix = new BDM[Double](sm, sn)
    val size = scale.x * scale.y
    var i = 0
    while (i < sm) {
      var j = 0
      while (j < sn) {
        var sum = 0.0
        var si = i * scale.x
        while (si < (i + 1) * scale.x) {
          var sj = j * scale.y
          while (sj < (j + 1) * scale.y) {
            sum += matrix(si, sj)
            sj += 1
          }
          si += 1
        }
        outMatrix(i, j) = sum / size
        j += 1
      }
      i += 1
    }
    outMatrix
  }

  /**
    * full conv
    *
    * @param matrix
    * @param kernel
    * @return
    */
  private[neuralNetwork] def convnFull(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val km: Int = kernel.rows
    val kn: Int = kernel.cols
    val extendMatrix = new BDM[Double](m + 2 * (km - 1), n + 2 * (kn - 1))
    var i = 0
    var j = 0
    while (i < m) {
      while (j < n) {
        extendMatrix(i + km - 1, j + kn - 1) = matrix(i, j)
        j += 1
      }
      i += 1
    }
    convnValid(extendMatrix, kernel)
  }

  /**
    * valid conv
    *
    * @param matrix
    * @param kernel
    * @return
    */
  private[neuralNetwork] def convnValid(matrix: BDM[Double], kernel: BDM[Double]): BDM[Double] = {
    val m: Int = matrix.rows
    val n: Int = matrix.cols
    val km: Int = kernel.rows
    val kn: Int = kernel.cols
    val kns: Int = n - kn + 1
    val kms: Int = m - km + 1
    val outMatrix: BDM[Double] = new BDM[Double](kms, kns)
    var i = 0
    while (i < kms) {
      var j = 0
      while (j < kns) {
        var sum = 0.0
        for (ki <- 0 until km) {
          for (kj <- 0 until kn)
            sum += matrix(i + ki, j + kj) * kernel(ki, kj)
        }
        outMatrix(i, j) = sum
        j += 1
      }
      i += 1
    }
    outMatrix
  }

  private[neuralNetwork] def getMaxIndex(out: Array[Double]): Int = {
    var max: Double = out(0)
    var index: Int = 0
    var i: Int = 1
    while (i < out.length) {
      if (out(i) > max) {
        max = out(i)
        index = i
      }
      i += 1
    }
    index
  }

  private[neuralNetwork] def combineGradient(
                                              g1: Array[(Array[Array[BDM[Double]]], Array[Double])],
                                              g2: Array[(Array[Array[BDM[Double]]], Array[Double])]):
  Array[(Array[Array[BDM[Double]]], Array[Double])] = {

    val l = g1.length
    var li = 0
    while (li < l) {
      if (g1(li) != null) {
        // kernel
        val layer = g1(li)._1
        val x = layer.length
        var xi = 0
        while (xi < x) {
          val line: Array[BDM[Double]] = layer(xi)
          val y = line.length
          var yi = 0
          while (yi < y) {
            line(yi) += g2(li)._1(xi)(yi)
            yi += 1
          }
          xi += 1
        }

        // bias
        val b = g1(li)._2
        val len = b.length
        var bi = 0
        while (bi < len) {
          b(bi) = b(bi) + g2(li)._2(bi)
          bi += 1
        }
      }
      li += 1
    }
    g1
  }


}