package slpart.exam

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.JedisHelper

object JedisExam {
  class ThreadDeom(threadName: String) extends Runnable{
    override def run(): Unit = {
      val jedisHelper = new JedisHelper("localhost",6379,false)
      val tp = scala.util.Random.nextInt(20)
      Thread.sleep((tp*1000).toLong)
      jedisHelper.init()
      val eid = jedisHelper.signIn()
      System.out.println(s"edge_${eid} sign in! ${jedisHelper.toString}")

      var minTime = 12.0
      var maxTime = 12.0

      for(i <- 1 to 5){
        val etime = scala.util.Random.nextInt(10) + 1
        Thread.sleep((etime*1000).toLong)

        jedisHelper.zaddEpochTime(etime)
        System.out.println(s"${threadName} save epoch time finished ${etime}")
        Thread.sleep(1200)

        val tensor = Tensor(5).randn()
        jedisHelper.lsetWeights(tensor)
        System.out.println(s"${threadName} save weights finished ${tensor.toArray().mkString(",")}")

        var blockstart = System.nanoTime()
        var waitTime = 0.0
        while(waitTime <= maxTime*2 && (jedisHelper.isFinishedETSave() == false)){
          Thread.sleep(1000)
          waitTime = (System.nanoTime() - blockstart) / 1e9
        }
        val epochTimex = jedisHelper.zrangeMinMaxEpochTime()
        jedisHelper.incrGetEpochTime()
        minTime = epochTimex._1
        maxTime = epochTimex._2
        System.out.println(s"${threadName} pull epoch time finished: min ${minTime} max ${maxTime}")
        jedisHelper.delEpochTimeKey()

        Thread.sleep(1000)
        blockstart = System.nanoTime()
        waitTime = 0.0
        while(waitTime <= maxTime*2 && (jedisHelper.isFinishedWeightSave() == false)){
          Thread.sleep(1000)
          waitTime = (System.nanoTime() - blockstart) / 1e9
        }

        var index = scala.util.Random.nextInt(100)
        while(index == (eid-1) && jedisHelper.currentEdgeNum() != 1){
          index = scala.util.Random.nextInt(100)
        }
        val rtensor = jedisHelper.lindexWeightsByIndex(index).asInstanceOf[Tensor[Float]]
        jedisHelper.incrGetWeights()
        System.out.println(s"${threadName} pull weight finished ${rtensor.toArray().mkString(",")}")

        jedisHelper.delWeightsKey()
      }
      System.out.println(s"edge_${eid} sign out!")
      jedisHelper.shutdown()
    }
  }

  def main(args: Array[String]): Unit = {
    import java.util.concurrent.{Executors, ExecutorService}
    val edgeNumber: Int = 5
    val threadPool = Executors.newFixedThreadPool(edgeNumber)
    try{
      for(idx <- 1 to edgeNumber){
        threadPool.execute(new ThreadDeom(("Thread_"+idx)))
      }
    }finally {
      threadPool.shutdown()
    }

//    val jedisHelper = new JedisHelper("localhost",6379,false)
//    jedisHelper.init()
//    System.out.println(jedisHelper.signIn())
//    jedisHelper.zaddEpochTime(79.0)
//    System.out.println(jedisHelper.zrangeMaxEpochTime())
//    System.out.println(jedisHelper.zrangeMinEpochTime())
//
//    val tensor_1 = Tensor(3,4).randn(0,1.0)
//    System.out.println((s"tensor_1:\n${tensor_1}"))
//    jedisHelper.lpushWeights(tensor_1)
//    System.out.println(jedisHelper.lindexWeithsByIndex(0))
//
//    System.out.println(jedisHelper.isFinishedETSave())
//    System.out.println(jedisHelper.isFinishedWeightSave())
//
//
//    jedisHelper.shutdown()
  }
}
