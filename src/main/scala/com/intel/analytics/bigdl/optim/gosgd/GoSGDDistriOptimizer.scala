package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.{Module, _}
import com.intel.analytics.bigdl.dataset.{DataSet, Identity, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.{Container, Module, Utils}
import com.intel.analytics.bigdl.parameters.{AllReduceParameter, ParameterProcessor}
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast}
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, EachCrossEntropyCriterion}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, MklDnnContainer, MklDnnLayer, MklDnnModule}
import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import org.apache.commons.lang.exception.ExceptionUtils
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.Logger
import org.apache.spark.network.netty.SparkTransportConf
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.mkldnn.{DnnGraph, MklDnnContainer, MklDnnLayer}
import com.intel.analytics.bigdl.utils.intermediate.{ConversionUtils, IRGraph}


object GoSGDDistriOptimizer extends AbstractOptimizer {
  import Optimizer._

  val logger: Logger = Logger.getLogger(getClass)
  import DistriOptimizer.Cache

  /**
   * Train the model.
   *
   * @param dataset train dataset
   * @param coresPerNode cores per node
   * @param state state table
   * @param endWhen trigger to stop training
   * @param metrics metrics
   * @param models cached models
   * @param optimMethods optimization methods
   * @param parameters [[AllReduceParameter]]
   * @param validationTrigger validation trigger
   * @param validationDataSet validation dataset
   * @param validationMethods validation methods
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param trainSummary train summary
   * @param validationSummary validation summary
   * @param isOverWrite if overwrite the checkpoint
   * @param parameterProcessers a list of ParameterProcessor used to process parameters
   */
  private[optim] def optimize[T: ClassTag](
                                            trainingModel: Module[T],
                                            dataset: DistributedDataSet[Array[Sample[T]]],
                                            coresPerNode: Int,
                                            state: Table,
                                            endWhen: Trigger,
                                            metrics: Metrics,
                                            models: RDD[Cache[T]],
                                            optimMethods: Map[String, OptimMethod[T]],
                                            parameters: Map[String, AllReduceParameter[T]],
                                            validationTrigger: Option[Trigger],
                                            validationDataSet: Option[DataSet[MiniBatch[T]]],
                                            validationMethods: Option[Array[ValidationMethod[T]]],
                                            cacheTrigger: Option[Trigger],
                                            cachePath: Option[String],
                                            trainSummary: Option[TrainSummary],
                                            validationSummary: Option[ValidationSummary],
                                            isOverWrite: Boolean,
                                            parameterProcessers: Array[ParameterProcessor]
                                          )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L

    // driverState is needed to prevent serializing the whole optimizer
    optimMethods.values.foreach{ optimMethod =>
      if (!optimMethod.state.contains("epoch")) optimMethod.state.update("epoch", 1)
      if (!optimMethod.state.contains("neval")) optimMethod.state.update("neval", 1)
      if (!optimMethod.state.contains("Loss")) {
        optimMethod.state.update("Loss", Float.PositiveInfinity)
      }
      if (!optimMethod.state.contains("score")) optimMethod.state.update("score", 0f)
      if (!optimMethod.state.contains("recordsProcessedThisEpoch")) {
        optimMethod.state.update("recordsProcessedThisEpoch", 0)
      }
    }

    val _subModelNumber = Engine.getEngineType() match {
      case MklBlas => coresPerNode
      case MklDnn => 1
    }
    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval"),
      "Loss" -> optimMethods.values.head.state("Loss"),
      "score" -> optimMethods.values.head.state("score"),
      "parallelism" -> _subModelNumber
    )

    logger.info(s"config $state")
    var recordsProcessedThisEpoch = optimMethods.values.head.state[Int]("recordsProcessedThisEpoch")
    if (recordsProcessedThisEpoch == 0) {
      val shuffleBefore = System.nanoTime()
      logger.info("Shuffle data")
      dataset.shuffle()
      val shuffleEnd = System.nanoTime()
      logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")
    }

    var tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var threshold = Long.MaxValue
    var timeout = Long.MaxValue
    var iteration = 0
    val dropPercentage = state.get[Double]("dropPercentage").get
    val warmupIterationNum = state.get[Int]("warmupIterationNum").get
    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val maxDropPercentage = state.get[Double]("maxDropPercentage").get
    val driverSubModelNum = partitionNum * _subModelNumber
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)

    // =====================   get user define state information =========================
    val goControler = state.get[GoSGDControler]("gocontroler").get
    val jedisHelper = new JedisHelper(host = goControler.redisHost,port = goControler.redisPort,
      clusterMode = goControler.redisClusterMode)
    jedisHelper.init()
    val edgeUID = jedisHelper.signIn()
    logger.info(s"Register edge${edgeUID}")
    var minEpochTimeCost: Double = 120.0
    var maxEpochTimeCost: Double = 120.0

    logger.info("Wait for other edge to sign in")
    while(jedisHelper.currentEdgeNum() < goControler.edgeNum){
      Thread.sleep(1000)
    }
    logger.info("All edges signin, begin to train the model")

    // =====================   end       =================================================

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = false)

    // ================== The first Epoch all data point are equal import ============
    var criticalSamples = dataRDD.mapPartitions(data => {
      val originalPoints = data.toArray.flatMap(f => f.tail)
      originalPoints.toIterator
    },preservesPartitioning = true)
    var criticalTrainDataset = (DataSet.rdd(criticalSamples) -> SampleToMiniBatch(batchSize = goControler.batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
    var criticalRDD = criticalTrainDataset.data(true)
    //criticalRDD.count() // trigger an spark job to get the original data points

    logger.info("Count dataset")
    val countBefore = System.nanoTime()
    var numSamples = criticalTrainDataset.data(false).map(_.size()).reduce(_ + _)// get the number of aggregated points
    val countAfter = System.nanoTime()
    logger.info(s"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9}s")
    var epochCostStart = System.nanoTime()

    while (!endWhen(driverState)) {
      val lossSum = sc.accumulator(0.0, "loss sum")
      val recordsNum = sc.accumulator(0, "record number")
      metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("get weights for each node", mutable.ArrayBuffer[Double](), sc)
      metrics.set("computing time average", 0.0, sc, partitionNum)
      metrics.set("aggregate gradient time", 0.0, sc, partitionNum)
      metrics.set("get weights average", 0.0, sc, partitionNum)
      metrics.set("put gradient", 0.0, sc, Engine.nodeNumber())
      metrics.set("aggregrateGradientParition average executor", 0.0, sc, Engine.nodeNumber())
      metrics.set("compute weight average", 0.0, sc, Engine.nodeNumber())
      metrics.set("send weights average", 0.0, sc, Engine.nodeNumber())

      // =====================   set something           =========================
      val curEpoch = driverState.get[Int]("epoch").get
      val curIteration = driverState.get[Int]("neval").get
      // =====================   end       =====================

      val driverMetrics = metrics
      val start = System.nanoTime()

      /*
        Run the forwards/backwards pass using multiple threads in each partition, and track the
        number of model updates that finished before the thread timeout mechanism.
       */
      val numFinishedModelUpdates: Int = criticalRDD
        .zipPartitions(models, preservesPartitioning = true) { (data, modelIter) => {
          val cached = modelIter.next()
          val syWStart = System.nanoTime()
          /*
            Note: All models in `cached` share the same storage for weights, so we only need to
            copy the weights from parameter server into the first model's weights.
           */
          val weightsResults = parameters.values.map(p =>
            p.getWeights(cached.modelWeights.head.narrow(1, p.paramOffset, p.size))
          ).toArray
          val miniBatchBuffer = new Array[MiniBatch[T]](_subModelNumber)
          val batch = data.next()
          val stackSize = batch.size() / _subModelNumber
          tasks += Engine.default.invoke(() => {
            require((batch.size() >= _subModelNumber) &&
              (batch.size() % _subModelNumber == 0), "total batch size: " +
              s"${batch.size()} should be divided by total core number: ${_subModelNumber}")
            if (batch.size() < _subModelNumber * 2) {
              logger.warn("Warning: for better training speed, " +
                "total batch size is recommended to be at least two times of core number" +
                s"${_subModelNumber}, please tune your batch size accordingly")
            }
            var b = 0
            while (b < _subModelNumber) {
              miniBatchBuffer(b) = batch.slice(b * stackSize + 1, stackSize)
              b += 1
            }
          })
          Engine.default.sync(tasks)
          weightsResults.foreach(_.waitResult())
          val weightSyncTime = System.nanoTime() - syWStart
          driverMetrics.add("get weights average", weightSyncTime)
          driverMetrics.add("get weights for each node", weightSyncTime)
          tasks.clear()

          // ======================Start train models===================================
          var time = System.nanoTime()
          if (dropPercentage > 0.0 && iteration > warmupIterationNum +
            computeThresholdbatchSize - 1) {
            timeout = threshold - weightSyncTime
          }
          val pre = (iteration % computeThresholdbatchSize) * _subModelNumber
          val trainingThreads = Engine.default.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              val trainStart = System.nanoTime()
              val localModel = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val input = miniBatchBuffer(i).getInput()
              val target = miniBatchBuffer(i).getTarget()

              if (Engine.getEngineType() == MklBlas || localModel.isInstanceOf[IRGraph[T]]) {
                val output = localModel.forward(input)
                lossArray(i) = ev.toType[Double](localCriterion.forward(output, target))
                val errors = localCriterion.backward(output, target)
                localModel.backward(input, errors)
              } else {
                Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
                  val output = localModel.forward(input)
                  lossArray(i) = ev.toType[Double](localCriterion.forward(output, target))
                  val errors = localCriterion.backward(output, target)
                  localModel.backward(input, errors)
                }))
              }
              cached.moduleTimeList(i + pre) = System.nanoTime() - trainStart + weightSyncTime
              i
            }
          ), timeout)
          val computingTime = System.nanoTime() - time
          driverMetrics.add("computing time average", computingTime)
          driverMetrics.add("computing time for each node", computingTime)

          val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
          recordsNum += finishedThreads.size * stackSize
          var i = 0
          while (i < finishedThreads.size) {
            lossSum += lossArray(finishedThreads(i))
            i += 1
          }

          if (finishedThreads.nonEmpty) {
            val finishedGradients = finishedThreads.map(cached.modelGradients(_))
            parameters.values.foreach { p =>
              time = System.nanoTime()
              val pOffset = p.paramOffset
              val pLength = p.size
              val taskSize = pLength / _subModelNumber
              val extraTask = pLength % _subModelNumber

              // Aggregate multi-model's gradient to the first model's gradient
              val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
              if (parallelNum != 1) {
                Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
                  val offset = pOffset + tid * taskSize + math.min(tid, extraTask)
                  val length = taskSize + (if (tid < extraTask) 1 else 0)
                  var i = 1
                  while (i < finishedGradients.length) {
                    finishedGradients(0).narrow(1, offset, length)
                      .add(finishedGradients(i).narrow(1, offset, length))
                    i += 1
                  }
                }))
                driverMetrics.add("aggregate gradient time", System.nanoTime() - time)
              }
              val putG = System.nanoTime()
              // Put first finished model's gradient who aggregated
              // all other models' gradient to AllReduceParameter
              p.putGradients(finishedGradients(0).narrow(1, pOffset, pLength))
              driverMetrics.add("put gradient", System.nanoTime() - putG)
            }
          } else {
            val putG = System.nanoTime()
            // zero gradient in BlockManager when no thread finished.
            cached.modelGradients(0).zero()
            parameters.values.foreach{p =>
              p.putGradients(cached.modelGradients(0).narrow(1, p.paramOffset, p.size))
            }
            driverMetrics.add("put gradient", System.nanoTime() - putG)
          }

          tasks ++= Engine.default.invoke {
            (0 until _subModelNumber).map { i =>
              () => {
                cached.localModels(i).training()
                cached.localModels(i).zeroGradParameters()
              }
            }
          }
          Iterator.single(finishedThreads.size)
        }
        }.reduce(_ + _)

      dropModelNumBatch += (driverSubModelNum - numFinishedModelUpdates)
      if (dropPercentage == 0.0 ||
        numFinishedModelUpdates >= driverSubModelNum * (1.0 - maxDropPercentage)) {
        // enough records were processed for this batch, so update the model
        val value = lossSum.value / numFinishedModelUpdates

        driverState("numFinishedModel") = numFinishedModelUpdates
        // isGradientUpdated is flag to mark whether gradient is updated. May changed in the future.
        driverState("isGradientUpdated") = false
        // parameterProcesser like L2NormClippingProcessor may aggregate gradient,
        // and change the value of isGradientUpdated in driverState.
        parameters.foreach { p =>
          parameterProcessers.foreach(_.collectGlobalData(models, p._2, metrics, driverState))
        }
        val isGradientUpdated = driverState[Boolean]("isGradientUpdated")
        val stateBroadcast = sc.broadcast(driverState)

        models.mapPartitions { modelIter =>
          val modelCache = modelIter.next()
          // if parameterProcesser has aggregated gradient, we can skip this aggregation.
          if (!isGradientUpdated) {
            val getG = System.nanoTime()
            parameters.values.foreach(_.aggregateGradientPartition(numFinishedModelUpdates))
            driverMetrics.add("aggregrateGradientParition average executor",
              System.nanoTime() - getG)
          }
          parameters.foreach { p =>
            parameterProcessers.foreach(_.processParameters(p._2, modelCache, driverState))
          }
          modelCache.optimMethods.foreach{ case (name, optimMethod) =>
            var time = System.nanoTime()
            optimMethod.state.update("epoch", driverState[Int]("epoch"))
            optimMethod.state.update("neval", driverState[Int]("neval"))
            optimMethod.state.update("Loss", driverState[Float]("Loss"))
            if (validationMethods.isDefined) {
              optimMethod.state.update("score", driverState[Float]("score"))
            }

            val p = parameters(name)
            optimMethod.optimize(_ => (ev.fromType(value), p.gradientPartition),
              p.weightPartition)
            driverMetrics.add("compute weight average", System.nanoTime() - time)
            time = System.nanoTime()
            p.sendWeightPartition()
            driverMetrics.add("send weights average", System.nanoTime() - time)
          }
          Iterator.empty
        }.count()

        stateBroadcast.destroy()
        recordsProcessedThisEpoch += recordsNum.value
        val end = System.nanoTime()
        wallClockTime += end - start
        driverState("isGradientUpdated") = true
        driverState("Loss") = lossSum.value.toFloat / numFinishedModelUpdates
        optimMethods.foreach{ v =>
          v._2.updateHyperParameter()
        }
        // TODO: Support show learningrate for multiOptimMethod
        driverState(s"LearningRate") = optimMethods.head._2.getLearningRate().toFloat

        driverState("Throughput") = recordsNum.value.toFloat / ((end - start) / 1e9f)
        val _header = header(driverState[Int]("epoch"), recordsProcessedThisEpoch, numSamples,
          driverState[Int]("neval"), wallClockTime)
        logger.info(s"${_header} Trained ${recordsNum.value} records in ${(end - start) / 1e9} " +
          s"seconds. Throughput is ${driverState("Throughput")} records/second. Loss is ${
            driverState("Loss")}. ${getHyperParameterLog(optimMethods)}")
        logger.debug("\n" + metrics.summary())
        logger.debug("Dropped modules: " + (driverSubModelNum - numFinishedModelUpdates))
        lossArray = new Array[Double](_subModelNumber)

        // compute threshold
        iteration += 1
        if (dropPercentage > 0.0 && iteration > warmupIterationNum &&
          iteration % computeThresholdbatchSize == 0) {
          val moduleTimeList = models.mapPartitions { iter =>
            iter.next().moduleTimeList.iterator
          }.collect()

          val k = (dropPercentage * computeThresholdbatchSize * driverSubModelNum).toInt
          if (k > dropModelNumBatch) {
            threshold = Util.kthLargest(moduleTimeList, 0, moduleTimeList.length-1,
              k - dropModelNumBatch)
          } else {
            threshold = (threshold * 1.01).toLong
          }
          logger.info("threshold: " + threshold)

          // clear moduleTimeList in each node
          models.mapPartitions { iter =>
            val timeList = iter.next.moduleTimeList
            var i = 0
            while (i < timeList.length) {
              timeList(i) = 0
              i += 1
            }
            Iterator.empty
          }.count()
          dropModelNumBatch = 0
        }

        driverState("neval") = driverState[Int]("neval") + 1
        if (recordsProcessedThisEpoch >= numSamples) {
          // Epoch is finished
          val epochEnd = System.nanoTime()
          wallClockTime = lastEpochTime + epochEnd - epochStart
          lastEpochTime = wallClockTime
          epochStart = System.nanoTime()
          logger.info(s"${_header} Epoch finished. Wall clock time is ${wallClockTime / 1e6} ms")

          // ---------------------  calculate selection ration based on the Time/Epoch of each Edge Device +++++++++++
          // ========== save epoch time ====================
          val curEpochTimeCost = (System.nanoTime() - epochCostStart) / 1e9
          logger.info(s"${_header} Save epoch cost time of edge${edgeUID} to Redis: ${curEpochTimeCost} s")
          jedisHelper.zaddEpochTime(curEpochTimeCost)
          epochCostStart = System.nanoTime()

          val mergeModelWeightsStart = System.nanoTime()
          logger.info(s"${_header} Edge_${edgeUID} pull latest model to driver")
          getModel(models, parameters, trainingModel) // pull the latest model parameters to driver

          // save model parameters of current edge to redis
          val paramsTensor = trainingModel.getParameters()._1
          logger.info(s"${_header} Save latest model parameters to Resis")
          jedisHelper.lsetWeights[T](paramsTensor)

          var blockStartx = System.nanoTime()
          var waitTime: Double = 0.0
          while(waitTime <= 2*maxEpochTimeCost && (jedisHelper.isFinishedETSave() == false)){
            Thread.sleep(1000)
            waitTime = (System.nanoTime() - blockStartx) / 1e9
          }
          val epochTimex = jedisHelper.zrangeMinMaxEpochTime()
          jedisHelper.incrGetEpochTime()

          minEpochTimeCost = epochTimex._1
          maxEpochTimeCost = epochTimex._2
          val criticalThreshold = if(goControler.fixedCriticalRatio.isDefined){
            goControler.fixedCriticalRatio.get
          }else{
            minEpochTimeCost / curEpochTimeCost
          }
          logger.info(s"Edge_${edgeUID} pull epoch time finished: min ${minEpochTimeCost} s, max ${maxEpochTimeCost} s " +
            s"criticalThreshold: ${criticalThreshold}")
          jedisHelper.delEpochTimeKey()

          val totalEdges = jedisHelper.currentEdgeNum()
          require(totalEdges > 0, s"cast long to int shoule > 0 totalEdges = ${totalEdges}")
          var selIndex = (new scala.util.Random).nextInt(totalEdges)
          while(selIndex == (edgeUID - 1) && goControler.edgeNum > 1){
            selIndex = (new scala.util.Random).nextInt(totalEdges)
          }
          logger.info(s"${_header} Pull latest model of edge_${selIndex+1} parameters from Resis")

          blockStartx = System.nanoTime()
          waitTime = 0.0
          while(waitTime <= 2*maxEpochTimeCost && (jedisHelper.isFinishedWeightSave() == false)){
            Thread.sleep(1000)
            waitTime = (System.nanoTime() - blockStartx) / 1e9
          }
          val otherEdgeParamsTensor = jedisHelper.lindexWeightsByIndex[T](selIndex)
          jedisHelper.incrGetWeights()
          logger.info(s"${_header} Merge model parameters from other Edge_${selIndex+1}")
          jedisHelper.delWeightsKey()

          //val edgeParams = trainingModel.getParameters()._1
          models.mapPartitions { modelIter =>
            val modelCache = modelIter.next()
            modelCache.optimMethods.foreach{ case (name, optimMethod) =>
              var time = System.nanoTime()
              val p = parameters(name)
              //optimMethod.optimize(_ => (ev.fromType(value), p.gradientPartition), p.weightPartition)
              require(p.weightPartition.nElement() == otherEdgeParamsTensor.nElement(),"Only support one optimMethod")
              p.weightPartition.add(otherEdgeParamsTensor)
              p.weightPartition.asInstanceOf[Tensor[Float]].mul(0.5f)
              p.sendWeightPartition()
            }
            Iterator.empty
          }.count()
          // ++++++++++++++++++++++++++++ End ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          val mergeModelWeightsCost = (System.nanoTime() - mergeModelWeightsStart) / 1e9
          val calCriticalStart = System.nanoTime()

          val currentTrainingEpoch = driverState[Int]("epoch")
          driverState("epoch") = driverState[Int]("epoch") + 1
          recordsProcessedThisEpoch = 0

          // --------------------- Every `epochInterval` epoch select the critical part of training data
          if(currentTrainingEpoch % goControler.epochInterval == 0){
            dataset.shuffle()
            dataRDD = dataset.data(train = false)

            logger.info(s"${_header} Select the critical part data points")
            // recalculate the importance (effect value) of each aggregated points
            // and select the critical part of original data points
            if(criticalThreshold >= 1.0) {
              criticalSamples = dataRDD.mapPartitions(data => {
                val originalPoints = data.toArray.flatMap(f => f.tail)
                originalPoints.toIterator
              },preservesPartitioning = true)
            }
            else{
              criticalSamples = dataRDD.zipPartitions(models,preservesPartitioning = true){ (data,modelIter) => {
                val cached = modelIter.next()
                /*
                      Note: All models in `cached` share the same storage for weights, so we only need to
                      copy the weights from parameter server into the first model's weights.
                     */
                val weightsResults = parameters.values.map(p =>
                  p.getWeights(cached.modelWeights.head.narrow(1, p.paramOffset, p.size))
                ).toArray
                weightsResults.foreach(_.waitResult())

                val partition = data.toArray
                val aggPoints = partition.map(_.head)
                val numAgg = aggPoints.length
                val localEachClassNLLCriterion = EachClassNLLCriterionOne()
                val costInformation = mutable.ArrayBuffer[(Int,Float)]()
                val localModel = cached.localModels.head

                var slicePartIter: Int = (numAgg / goControler.batchSize).toInt
                val leftPartNumber = numAgg % goControler.batchSize
                if(leftPartNumber > 0){
                  slicePartIter += 1
                }

                for(curIterIdx <- 0 until slicePartIter){
                  val positionX: Int = curIterIdx * goControler.batchSize
                  val offsetX: Int = if(curIterIdx < (slicePartIter - 1)){
                    goControler.batchSize
                  }else{
                    leftPartNumber
                  }
                  val curAggPoints = aggPoints.slice(positionX,positionX+offsetX)
                  val aggMiniBatch = SampleToMiniBatch(offsetX,partitionNum = Some(1))
                    .apply(curAggPoints.toIterator).next()
                  require(aggMiniBatch.size() == offsetX,s"size of aggregated samples (${curAggPoints.length}) " +
                    s"should be equal to aggMiniBatch (${aggMiniBatch.size()})")

                  val input = aggMiniBatch.getInput()
                  val target = aggMiniBatch.getTarget().toTensor
                  if (Engine.getEngineType() == MklBlas || localModel.isInstanceOf[IRGraph[T]]) {
                    val output = localModel.forward(input).toTensor
                    val eachloss = localEachClassNLLCriterion.forward(output,target)
                    require(eachloss.length == offsetX,s"the number of loss ${eachloss.length} should equal to aggregated samples ${offsetX}")
                    for(cu <- 0 until offsetX){
                      val tl = ev.toType[Float](eachloss(cu))
                      costInformation.append(Tuple2(positionX+cu,if(tl.isNaN || tl.isInfinite) 0.0f else tl))
                    }

                  } else {
                    Engine.dnnComputing.invokeAndWait2(Array(0).map(_ => () => {
                      val output = localModel.forward(input).toTensor
                      val eachloss = localEachClassNLLCriterion.forward(output,target)
                      require(eachloss.length == offsetX,s"the number of loss ${eachloss.length} should equal to aggregated samples ${offsetX}")
                      for(cu <- 0 until offsetX){
                        val tl = ev.toType[Float](eachloss(cu))
                        costInformation.append(Tuple2(positionX+cu,if(tl.isNaN || tl.isInfinite) 0.0f else tl))
                      }
                    }))
                  }
                }

                val sortedInfo = costInformation.sortWith((x,y) => x._2 > y._2).map(_._1)
                val choosedCritical = sortedInfo.slice(0,(numAgg * criticalThreshold).toInt).flatMap(idx => partition(idx).tail)
                choosedCritical.toIterator
              }
              }
            }
            criticalTrainDataset = (DataSet.rdd(criticalSamples) -> SampleToMiniBatch(batchSize = goControler.batchSize))
              .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
          } else{
            criticalTrainDataset.shuffle()
          }

          criticalRDD = criticalTrainDataset.data(true)
          //criticalRDD.count() // trigger an spark job to get the original data points
          logger.info(s"${_header} Critical part selection End!")
          logger.info("Count dataset")
          val countBefore = System.nanoTime()
          numSamples = criticalTrainDataset.data(false).map(_.size()).reduce(_ + _)// get the number of aggregated points
          val countAfter = System.nanoTime()
          logger.info(s"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9}s")
          val calCriticalCost = (System.nanoTime() - calCriticalStart) / 1e9
          logger.info(s"Timecost of each part, epochTimeCost: ${curEpochTimeCost} s, " +
            s"mergeModelWeightsCost: ${mergeModelWeightsCost} s, calCriticalCost: ${calCriticalCost} s")
        }

        optimMethods.map { case (moduleName, optimMethod) =>
          optimMethod.state.update("recordsProcessedThisEpoch", recordsProcessedThisEpoch)
          optimMethod.state.update("epoch", driverState[Int]("epoch"))
          optimMethod.state.update("neval", driverState[Int]("neval"))
          optimMethod.state.update("Loss", driverState[Float]("Loss"))
          if (validationMethods.isDefined) {
            optimMethod.state.update("score", driverState[Float]("score"))
          }
        }

        validate(
          validationTrigger,
          validationDataSet,
          validationMethods,
          coresPerNode,
          models,
          driverState,
          validationSummary,
          _header,
          parameters
        )

        trainSummary.foreach { summary =>
          saveSummary(
            summary,
            models,
            driverState,
            parameters,
            trainingModel
          )
        }

        checkpoint(
          cacheTrigger,
          cachePath,
          isOverWrite,
          wallClockTime,
          models,
          driverState,
          parameters,
          optimMethods,
          trainingModel
        )

      } else {
        logger.info(s"Warning! Not enough training samples were successfully processed in this " +
          s"iteration due to some slow tasks. The gradients computed in this iteration will be " +
          s"discarded. Only $numFinishedModelUpdates/$driverSubModelNum threads successfully " +
          s"completed training.")
      }
    } // end of while(!endWhen(driverState))
    jedisHelper.shutdown()
  }


  /**
   ** Create checkpoint.
   * @param cacheTrigger cache trigger
   * @param cachePath cache path
   * @param isOverWrite whether over write
   * @param wallClockTime wall clock time
   * @param models cached models
   * @param state state table
   * @param parameters all reduce parameters
   * @param optimMethods all optim methods
   * @param trainingModel training model
   */
  override def checkpoint[T: ClassTag](
                                        cacheTrigger: Option[Trigger],
                                        cachePath: Option[String],
                                        isOverWrite: Boolean,
                                        wallClockTime: Long,
                                        models: RDD[Cache[T]],
                                        state: Table,
                                        parameters: Map[String, AllReduceParameter[T]],
                                        optimMethods: Map[String, OptimMethod[T]],
                                        trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    cacheTrigger.foreach { trigger =>
      cachePath.foreach { path =>
        if (trigger(state)) {
          saveModel(getModel(models, parameters, trainingModel), cachePath, isOverWrite,
            s".${state[Int]("epoch") - 1}")
          logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save model to $path")
          optimMethods.foreach{case (name, optimMethod) =>
            optimMethod.state.update("epoch", state[Int]("epoch"))
            optimMethod.state.update("neval", state[Int]("neval"))
            saveOptimMethod(optimMethod, cachePath, isOverWrite, s"-$name.${state[Int]("epoch") - 1}")
            logger.info(s"[Wall Clock ${wallClockTime / 1e9}s] Save optimMethod " +
              s"${optimMethod} to $path")
          }
        }
      }
    }
  }


  /**
   * Init engine and cache models, weights, gradients, criterions, state tables
   * and validation methods on worker nodes.
   *
   * @param model train model
   * @param dataset train dataset
   * @param criterion loss function
   * @param state state table
   * @param nodeNumber node number
   * @param coresPerNode cores per node
   * @param checkSingleton if checkSingleton
   * @param parameters all reduce parameter instance
   * @param validationMethods validation methods
   * @param optimMethod optimization method
   * @param parameterProcessors a list of ParameterProcessor used to process parameters
   * @return cached models
   */
  private def initThreadModels[T: ClassTag](
                                             model: Module[T],
                                             dataset: DistributedDataSet[Array[Sample[T]]],
                                             criterion: Criterion[T],
                                             state: Table,
                                             nodeNumber: Int,
                                             coresPerNode: Int,
                                             checkSingleton: Boolean,
                                             parameters: Map[String, AllReduceParameter[T]],
                                             validationMethods: Option[Array[ValidationMethod[T]]],
                                             optimMethod: Map[String, OptimMethod[T]],
                                             parameterProcessors: ArrayBuffer[ParameterProcessor]
                                           )(implicit ev: TensorNumeric[T]): (RDD[DistriOptimizer.Cache[T]], ModelBroadcast[T]) = {
    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast((criterion, state, validationMethods, optimMethod))
    // ensure model's parameter is compacted for getting a better performance when broadcasting
    model.getParameters()
    // As cloneModel is using Serialization to implement deep copy, and will throw OOMError
    // when model's size is bigger than SerializationUtils' buffer size. So we can use
    // ModelBroadcast to clone model here.
    // Notes: All models returned by modelBroadcast.value() share the same weight&bias, while
    // gradWeight&gradBias is unshared.
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, ConversionUtils.convert(model))
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => coresPerNode
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }

    require(dataset.originRDD().partitions.length == nodeNumber,
      s"Passed in rdd partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number ${nodeNumber}")


    val computeThresholdbatchSize = state.get[Int]("computeThresholdbatchSize").get
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()

    val models = dataset.originRDD().mapPartitions(_ => {
      val partitionId = TaskContext.getPartitionId
      val (broadcastCriterion, broadcastState, broadcastMethod,
      broadcastOptim) = broadcast.value
      if (!Engine.checkSingleton()) {
        if (checkSingleton) {
          require(Engine.checkSingleton(), "Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient training" +
            "data to be distributed? Set property \"bigdl.check.singleton\" to false to skip " +
            "this check")
        } else {
          logger.warn("Partitions of the training data are not evenly" +
            "distributed across the executors in the Spark cluster; are there sufficient training" +
            "data to be distributed?")
        }
      }
      Engine.setNodeAndCore(nExecutor, executorCores)
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = modelBroadcast.value(true)
        if (Engine.getEngineType() == MklDnn && !localModel.isInstanceOf[IRGraph[T]]) {
          Engine.dnnComputing.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              localModel match {
                case container: MklDnnContainer => container.compile(TrainingPhase)
                case graph: DnnGraph => graph.compile(TrainingPhase)
                case _ =>
              }
            }))
        }
        setModelId(localModel, partitionId)
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val localMethod =
          if (broadcastMethod.isDefined) Some(broadcastMethod.get.map(_.clone())) else None
        val (weights, grads) = localModel.getParameters()
        (localModel, weights, grads, localCriterion, localState, localMethod)
      }.toArray

      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val weights = cached.head._2
      parameters.foreach(v =>
        v._2.init(weights.narrow(1, v._2.paramOffset, v._2.size))
      )

      Iterator.single(Cache(
        cached.map(_._1), // models
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        new Array[Long](_subModelNumber * computeThresholdbatchSize),
        cached.map(_._6),
        broadcastOptim.map(v => (v._1, v._2.clone()))
      ))
    }).persist()
    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    models.count()
    logger.info("Cache thread models... done")
    (models, modelBroadcast)
  }

  private def setModelId[T: ClassTag](model: Module[T], partitionId: Int): Unit = {
    model.setId(partitionId)
    if (model.isInstanceOf[Container[_, _, T]]) {
      model.asInstanceOf[Container[_, _, T]].modules.
        foreach(sub => setModelId(sub, partitionId))
    }
  }

  /**
   * Fetch current model parameters to driver, and copy to trainingModel.
   *
   * @param models cached models
   * @param parameters [[AllReduceParameter]]
   * @param trainingModel the model is trained by optimizer
   * @return trained model
   */
  override protected def getModel[T: ClassTag](
                                                models: RDD[Cache[T]],
                                                parameters: Map[String, AllReduceParameter[T]],
                                                trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val partitionNum = models.partitions.length
    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
    trainingModel.setExtraParameter(extraState)

    // make sure gradient is as the same length as weight
    val parameterArray = trainingModel.parameters()
    (0 until parameterArray._2.length).foreach(i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    )

    val (parameter, gradientParameter) = trainingModel.getParameters()

    parameters.foreach { case (moduleName, p) =>
      val currentModule = trainingModel(moduleName)
      require(currentModule.isDefined, s"Couldn't find $moduleName in $trainingModel")
      val (weights, gradients) = models.mapPartitions(iter => {
        val cached = iter.next()
        val curPartitionId = TaskContext.getPartitionId()
        Iterator.single((Map(curPartitionId -> p.weightPartition),
          Map(curPartitionId -> p.gradientPartition)))
      }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

      val taskSize = p.size / partitionNum
      require(taskSize != 0, "parameter length should not less than partition number")
      val extraSize = p.size % partitionNum

      (0 until partitionNum).map(pid => {
        val start = p.paramOffset + pid * taskSize + math.min(pid, extraSize)
        val length = taskSize + (if (pid < extraSize) 1 else 0)
        parameter.narrow(1, start, length).copy(weights(pid))
        gradientParameter.narrow(1, start, length).copy(gradients(pid))
      })
    }

    trainingModel
  }

}


class GoSGDDistriOptimizer[T: ClassTag] (
                                          _model: Module[T],
                                          _dataset: DistributedDataSet[Array[Sample[T]]],
                                          _criterion: Criterion[T]
                                        )(implicit ev: TensorNumeric[T])
  extends Optimizer[T, Array[Sample[T]]](
    _model, _dataset, _criterion) {
  val metrics = new Metrics

  private var models: RDD[DistriOptimizer.Cache[T]] = null
  // this variable is used to check the models cloned when broadcast, if there're native resources,
  // it will be deleted at the end of Optimizer.
  private var modelBroadcast: ModelBroadcast[T] = null

  /**
   * Clean some internal states, so this or other optimizers can run optimize again
   *
   * This method will be called at the end of optimize. You need not call it if optimize succeed.
   * If the optimize fails, you may call it before next optimize.
   */
  def clearState() : Unit = {
    GoSGDDistriOptimizer.clearState(models)
  }


  // By default, optimMethod internal state for each worker will not be reserved and reuse.
  private var reserveOptimMethod = false
  private[bigdl] var previousOptim: RDD[Map[String, OptimMethod[T]]] = null
  /**
   * If you want to reserve optimMethod for each worker, and reuse those methods in
   * next training task, you can call it.
   */

  /**
   * If you want to reserve optimMethod for each worker and reuse those methods in
   * next training task, please set reserve = true
   * Otherwise, if just using optimMethod you set in optimizer, please set reserve = false
   * @param reserve whether to reserve optim method for each worker
   * @return
   */
  override def reserveOptim(reserve: Boolean): this.type = {
    reserveOptimMethod = reserve
    this
  }

  // replace optim methods with previous
  private def resetOptimMethods[T: ClassTag](
                                              models: RDD[DistriOptimizer.Cache[T]],
                                              previousOptimMethods: RDD[Map[String, OptimMethod[T]]]):
  RDD[DistriOptimizer.Cache[T]] = {
    models.zipPartitions(previousOptimMethods) { (m1, m2) => {
      val cache = m1.next()
      cache.optimMethods = m2.next()
      Iterator(cache)
    }
    }
  }

  private def endEpoch(): Unit = {
    GoSGDDistriOptimizer.endEpoch(optimMethods)
  }

  def setTrainData(sampleRDD: RDD[(Long,Array[Sample[T]])],
                   batchSize: Int): this.type = {
    this.dataset = (com.intel.analytics.bigdl.dataset.DataSet.rdd(sampleRDD) -> com.intel.analytics.bigdl.dataset.Identity()
      ).asInstanceOf[DistributedDataSet[Array[Sample[T]]]]
    // if current epoch is not finished, we will end the
    // current epoch and start a new epoch when optimize is called
    endEpoch()
    this
  }

  private[bigdl] def prepareInput2[T: ClassTag](dataset: DataSet[Array[Sample[T]]],
                                                validationDataSet: Option[DataSet[MiniBatch[T]]]): Unit = {
    dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]].cache()
    if (validationDataSet.isDefined) {
      validationDataSet.get.toDistributed().cache()
    }
  }

  override def prepareInput(): Unit = {
    if (!dataset.toDistributed().isCached) {
      GoSGDDistriOptimizer.logger.info("caching training rdd ...")
      prepareInput2(this.dataset, this.validationDataSet)
    }
  }

  override def optimize(): Module[T] = {

    val distDataset = dataset.toDistributed()
    val trainingModel = if (Engine.getEngineType() == MklDnn && !model.isInstanceOf[MklDnnModule]
      && !model.isInstanceOf[IRGraph[T]] && !model.isInstanceOf[Graph[T]]) {
      model.toGraph().setName(model.getName())
    } else model

    optimMethods.values.foreach { optimMethod =>
      optimMethod.clearHistory()
    }

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    state("isLayerwiseScaled") = Utils.isLayerwiseScaled(_model)

    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()

    val partitionNum = distDataset.originRDD().partitions.length
    val modelParameters = trainingModel.getParameters()
    // subModuleName -> (storageOffset, length, AllReduceParameter)
    val parameters = if (optimMethods.size != 1) {
      val p = optimMethods.map{case (subModuleName, optimMethods) =>
        val subModule = trainingModel(subModuleName)
        require(subModule.isDefined, s"Optimizer couldn't find $subModuleName in $model")
        val subModuleWeights = subModule.get.getParameters()._1
        (subModuleName, subModuleWeights)
      }
      val sortedWeights = p.values.toArray.sortWith((a, b) => a.storageOffset() < b.storageOffset())
      val compactWeights = Module.isCompact(sortedWeights)
      require(modelParameters._1 == compactWeights,
        s"DistriOptimizer: All subModules should have an OptimMethod.")
      p.map{case (subModuleName, weights) =>
        (subModuleName, AllReduceParameter.newParameter[T](
          partitionNum, weights.nElement(), weights.storageOffset()))
      }
    } else if (optimMethods.contains(trainingModel.getName())) {
      Map(trainingModel.getName() -> AllReduceParameter.newParameter[T](
        partitionNum, modelParameters._1.nElement()))
    } else {
      throw new IllegalArgumentException(s"${trainingModel.getName()} doesn't " +
        s"have corresponding OptimMethod")
    }

    prepareInput()

    val modelsAndBroadcast = GoSGDDistriOptimizer.initThreadModels(trainingModel, distDataset, criterion,
      state, nodeNumber, coresPerNode, checkSingleton, parameters, validationMethods,
      optimMethods, parameterProcessors)

    models = if (reserveOptimMethod && previousOptim != null) {
      // replace optimMethods with previous ones
      resetOptimMethods(modelsAndBroadcast._1, previousOptim)
    } else {
      modelsAndBroadcast._1
    }
    modelBroadcast = modelsAndBroadcast._2

    if (checkpointPath.isDefined) {
      val file = checkpointPath.get + "/" +
        new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime())
      new File(file).mkdir()
      checkpointPath = Some(file)
    }

    var retryNum = 0
    val maxRetry = System.getProperty("bigdl.failure.retryTimes", "5").toInt
    val retryTimeInterval = System.getProperty("bigdl.failure.retryTimeInterval", "120").toInt
    var lastFailureTimestamp = System.nanoTime()

    while (retryNum < maxRetry) {
      try {
        GoSGDDistriOptimizer.optimize(
          trainingModel,
          distDataset,
          coresPerNode,
          state,
          endWhen,
          metrics,
          models,
          optimMethods,
          parameters,
          validationTrigger,
          validationDataSet,
          validationMethods,
          checkpointTrigger,
          checkpointPath,
          trainSummary,
          validationSummary,
          isOverWrite,
          parameterProcessors.toArray
        )
        retryNum = Int.MaxValue
      } catch {
        case e: IllegalArgumentException =>
          throw e
        case t: Throwable =>
          GoSGDDistriOptimizer.logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          if (checkpointPath.isDefined) {
            /* To avoid retry number is used up by first few exceptions, we count time here.
             * If exception exceeds maxRetry times in maxRetry*retryTimeInterval seconds,
             * we will give up retry Or we will reset retryNum
             */
            if (System.nanoTime() - lastFailureTimestamp < maxRetry * retryTimeInterval * 1e9) {
              retryNum += 1
              if (retryNum == maxRetry) {
                throw t
              }
            } else {
              retryNum = 1
            }
            GoSGDDistriOptimizer.logger.info(s"Retrying $retryNum times")
            lastFailureTimestamp = System.nanoTime()

            val modelFile = getLatestFile(checkpointPath.get, "model")
            clearState()
            models.unpersist()
            val newModel = if (modelFile != null) {
              GoSGDDistriOptimizer.logger.info("Model recover from last snapshot")
              Module.load[T](modelFile)
            } else {
              GoSGDDistriOptimizer.logger.info("Model recover from origin model")
              trainingModel
            }
            optimMethods = optimMethods.map { case (moduleName, optimMethod) =>
              val methodFile = getLatestFile(checkpointPath.get, s"optimMethod-$moduleName")

              val newOptimMethod = if (methodFile != null) {
                GoSGDDistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from last snapshot")
                OptimMethod.load[T](methodFile)
              } else {
                GoSGDDistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from origin model")
                optimMethod
              }
              newOptimMethod.clearHistory()
              (moduleName, newOptimMethod)
            }
            val modelsAndBroadcast = GoSGDDistriOptimizer.initThreadModels(newModel, distDataset,
              criterion, state, nodeNumber, coresPerNode, checkSingleton, parameters,
              validationMethods, optimMethods, parameterProcessors)
            models = modelsAndBroadcast._1
            modelBroadcast = modelsAndBroadcast._2
          } else {
            throw t
          }
      }
    }

    GoSGDDistriOptimizer.getModel(models, parameters, trainingModel)

    // Reset some internal states, so this or other optimizers can run optimize again
    clearState()

    // unpersist the model because the next time optimize is called, new `models` will be
    // created
    shutdown()

    // reserve optimMethod internal state for each worker if need
    if (reserveOptimMethod) {
      previousOptim = models.map(m => m.optimMethods).cache()
      previousOptim.count()
    } else {
      if (previousOptim != null) previousOptim.unpersist()
    }
    models.unpersist()

    trainingModel
  }

  private def getLatestFile(path: String, fileName: String): String = {
    val fl = new java.io.File(path)
    val files = fl.listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean = {
        name.startsWith(fileName)
      }
    })

    var lastMod = Long.MinValue
    var choice: String = null
    files.map {file =>
      if (file.lastModified() > lastMod) {
        choice = file.getPath;
        lastMod = file.lastModified();
      }
    }
    return choice;
  }

  // this shutdown should not be called out of this scope.
  private[optim] override def shutdown(): Unit = {
    models.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }

      iter
    }.count()
    CachedModels.deleteKey(modelBroadcast.uuid)
  }
}