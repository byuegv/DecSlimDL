package slpart.models.gosgd

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, Identity, MiniBatch, Sample, SampleToIDAryMiniBatch}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import slpart.dataloader.Cifar10Loader
import wxw.dataloader.AggCIFAR10
import slpart.models.archsets.squeezenet.SqueezeNetCifarV3

object TrainSqueezeNetV3 {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.WARN)

  import slpart.models.OptionUtils._

  def cifar10Decay(epoch: Int) = if(epoch >= 81) 2.0 else if(epoch >= 51 ) 1.0 else 0.0
  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param =>{
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-${param.appName}-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val start = System.nanoTime()
      //      val trainSamples = Cifar10Loader.trainingSamplesAry(sc,param.dataset,param.zeroScore,isAggregate = param.aggregate,category = param.classes,
      //        itqbitN = param.itqbitN,itqitN = param.itqitN,itqratioN = param.itqratioN,upBound = param.upBound,splitN = param.minPartN,isSparse = param.isSparse)
      //      val validationSamples = Cifar10Loader.validateSamples(sc,param.dataset,param.zeroScore)
      val labelSel = parseLabelSelection2Set(param.labelSelection)
      val trainSamples = if(param.sampleFraction.isDefined){
        AggCIFAR10.trainCifarData(sc,labelSel,aggMethod = param.aggMethod,path = param.dataset, upBound = param.upBound)
          .sample(withReplacement = false,fraction = param.sampleFraction.get)
      } else{
        AggCIFAR10.trainCifarData(sc,labelSel,aggMethod = param.aggMethod,path = param.dataset, upBound = param.upBound)
      }
      val validationSamples = AggCIFAR10.valCifarData(sc,param.dataset)
      logger.error(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")

      val model = if(param.loadSnapshot && param.model.isDefined){
        Module.load[Float](param.model.get)
      }
      else{
        SqueezeNetCifarV3(numClass = param.classes,hasBN = param.hasBN,hasDropout = param.hasDropout)
      }

      val optimMethod = if(param.loadSnapshot && param.state.isDefined){
        OptimMethod.load[Float](param.state.get)
      }else{
        param.optMethod match {
          case "adam" => new Adam[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "radam" => new RectifiedAdam[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,weightDecay = param.weightDecay)
          case "adadelta" => new Adadelta[Float]()
          case "rmsprop" => new RMSprop[Float](param.learningRate,param.learningRateDecay)
          case "ftrl" => new Ftrl[Float](param.learningRate)
          case _ => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,
            dampening = param.dampening,nesterov = param.nesterov,learningRateSchedule = SGD.EpochDecay(cifar10Decay))
        }
      }

      if(param.storeInitModel && param.initModel.isDefined){
        logger.error(s"save initial model in ${param.initModel.get}")
        model.save(param.initModel.get,true)
        if(param.initState.isDefined){
          logger.error(s"save init state in ${param.initState.get}")
          optimMethod.save(param.initState.get,true)
        }
      }

      //      val trainDataset = (DataSet.rdd(trainSamples.map(_._2)) -> Identity()).asInstanceOf[DistributedDataSet[Array[Sample[Float]]]]
      val trainDataset = (DataSet.rdd(trainSamples) -> Identity()).asInstanceOf[DistributedDataSet[Array[Sample[Float]]]]

      val criterion = ClassNLLCriterion[Float]()
      val optimizer = new DecGoDistriOptimizer(model,trainDataset,_criterion = criterion)

      if(param.checkpoint.isDefined){
        //        optimizer.setCheckpoint(param.checkpoint.get,Trigger.severalIteration(param.checkpointIteration))
        optimizer.setCheckpoint(param.checkpoint.get,Trigger.everyEpoch)
        if(param.overwriteCheckpoint) optimizer.overWriteCheckpoint()
      }

      val goControler = new GoSGDControler()
      goControler.batchSize = param.batchSize
      goControler.redisHost = param.redisHost
      goControler.redisPort = param.redisPort
      goControler.redisDatabase = param.redisDatabase
      goControler.redisClusterMode = param.redisClusterMode
      goControler.fixedCriticalRatio = param.fixedCriticalRatio
      goControler.edgeNum = param.edgeNum
      goControler.epochInterval = param.epochInterval


      val prestate = T(
        ("gocontroler",goControler)
      )
      // set user defined state
      optimizer.setState(prestate)

      optimizer.setValidation(
        trigger = Trigger.everyEpoch,
        sampleRDD = validationSamples,
        vMethods = Array(new Top1Accuracy(),new Top5Accuracy(),new Loss(ClassNLLCriterion[Float]())),
        batchSize = param.batchSize
      )
      optimizer.setOptimMethod(optimMethod)
      optimizer.setEndWhen(Trigger.maxEpoch(param.maxEpoch))

      val trainedModel = optimizer.optimize()

      if(param.storeTrainedModel && param.trainedModel.isDefined){
        logger.error(s"save trained model in ${param.trainedModel.get}")
        trainedModel.save(param.trainedModel.get,overWrite = true)
        if(param.trainedState.isDefined) {
          logger.error(s"save trained state in ${param.trainedState.get}")
          optimMethod.save(param.trainedState.get,overWrite = true)
        }
      }

      sc.stop()
    })
  }

}
