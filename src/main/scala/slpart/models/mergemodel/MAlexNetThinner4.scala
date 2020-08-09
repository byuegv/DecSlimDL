package com.intel.analytics.bigdl.optim

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import wxw.dataloader.{AggCIFAR10, AggMNIST}
import com.intel.analytics.bigdl.numeric.NumericFloat

object MAlexNetThinner4 {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.WARN)

  import slpart.models.OptionUtils._
  def main(args: Array[String]): Unit = {
    testParser.parse(args,new TestParams()).map(param => {
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-mergeAleNetThinner4Test-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(s"${curTime}-mergeAleNetThinner4Test")
      val sc = new SparkContext(conf)
      Engine.init
      val validationSamples = AggCIFAR10.valCifarData(sc,param.dataset)
      val modelList = param.model.trim.split(",")
      val modelNum = modelList.length
      require(modelNum > 0,s"The model number should larger than 0, got ${modelNum}")

      val model = Module.load[Float](modelList.head)
      val (weights,gradients) = model.getParameters()
      for(elem <- modelList.tail){
        val tpModel = Module.load[Float](elem)
        val (tpWei,tpGrad) = tpModel.getParameters()
        weights.add(tpWei)
        gradients.add(tpGrad)
      }
      if(modelNum>1){
        weights.div(modelNum)
        gradients.div(modelNum)
      }

      val result = model.evaluate(validationSamples,
        Array(new Top1Accuracy[Float],new Top5Accuracy[Float](),new Loss[Float]()), Some(param.batchSize))

      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    })


  }
}
