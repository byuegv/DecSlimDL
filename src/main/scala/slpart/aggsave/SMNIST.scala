package slpart.aggsave

import java.text.SimpleDateFormat
import java.util.Date
import java.io.PrintWriter

import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import slpart.dataloader.MnistLoader
import wxw.dataloader.AggMNIST

object SMNIST {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.WARN)

  import slpart.models.OptionUtils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param =>{
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-${param.appName}-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val trainSamples = MnistLoader.trainingSamplesAry(sc,param.dataset,param.zeroScore,isAggregate = param.aggregate,category = param.classes,
              itqbitN = param.itqbitN,itqitN = param.itqitN,itqratioN = param.itqratioN,upBound = param.upBound,splitN = param.minPartN,isSparse = param.isSparse)
      val validationSamples = MnistLoader.validateSamples(sc,param.dataset,param.zeroScore)


//      val labelSel = parseLabelSelection2Set(param.labelSelection)
//      val trainSamples = AggMNIST.trainMnistData(sc,labelSel,aggMethod = param.aggMethod,path =
//        param.dataset, upBound = param.upBound)
//      val validationSamples = AggMNIST.valMnistData(sc,param.dataset)

      val maxlen = trainSamples.map(_._2.length).max()
      val trainOut = new PrintWriter("training.txt")
      trainSamples.map(_._2).collect().foreach(agglist => {
        val len = agglist.length
        for(i <- 0 until maxlen){
          val point = agglist(i%len)
          val label = point.label().squeeze().toArray().head
          val feature = point.feature().reshape(Array(1*28*28)).squeeze().toArray()
          trainOut.print(label)
          trainOut.print(";")
          trainOut.print(feature.mkString(","))
          if(i < (maxlen-1)){
            trainOut.print("|")
          }
        }
        trainOut.println()
        }
      )
      trainOut.close()
      val testOut = new PrintWriter(("test.txt"))
      validationSamples.collect().foreach(point => {
        val label = point.label().squeeze().toArray().head
        val feature = point.feature().reshape(Array(1*28*28)).squeeze().toArray()
        testOut.print(label)
        testOut.print(";")
        testOut.print(feature.mkString(","))
        testOut.println()
      })
      testOut.close()


      sc.stop()
    })
  }

}
