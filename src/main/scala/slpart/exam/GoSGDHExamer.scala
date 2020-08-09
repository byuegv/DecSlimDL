package slpart.exam

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.GoSGDHelper

object GoSGDHExamer {
  class ThreadDeom(threadName: String,totalEdge: Int) extends Runnable{
    override def run(): Unit = {
      val jedisHelper = new GoSGDHelper("localhost",6379,false)
      val tp = scala.util.Random.nextInt(20)
      Thread.sleep((tp*1000).toLong)
      val edgeName = jedisHelper.signIn()
      System.out.println(s"${threadName} sign in! ${jedisHelper.toString}")

      // wait for all edges sign in
      while(jedisHelper.currentEdges() < totalEdge){
        Thread.sleep(1000)
      }
      System.out.println("All Edges signed in!")

      var minTime = 12.0
      var maxTime = 12.0

      for(i <- 1 to 15){
        // ========== Training and aggregate parameters
        val etime = scala.util.Random.nextInt(10) + 1
        Thread.sleep((etime*1000).toLong)
        Thread.sleep(1200)

        val score = scala.util.Random.nextDouble()
        val tensor = Tensor(5).randn()
        val selId = jedisHelper.selEdgeID(canBeSelf = true)
        val threshold = scala.util.Random.nextDouble()
        if(threshold >= 0.5){
          System.out.println("Only insert timecost")
          jedisHelper.insertoTime(etime)
        }
        else{
          System.out.println(("Insert timecost and parameters"))
          jedisHelper.insertScoreWeightxTime[Float](etime,selId,score,tensor)
        }
        System.out.println(s"${threadName} save weights finished ${tensor.toArray().mkString(",")}")

        // ---------- wait until all edge finish save timecost and parameters --------
        while(jedisHelper.isFinishedSave() == false){
          Thread.sleep(1000)
          System.out.println(s"${threadName} wait other edges to finish save parameters and time")
        }

        val (rtensorAry, epochTimex ) = jedisHelper.popScoreWeightxTime[Float]()
        rtensorAry.foreach(rtensor => {
          val score = rtensor._1
          val weig = rtensor._2
          System.out.println(s"${threadName} pull weight finished score: ${score}, weights: ${weig.toArray().mkString(",")}")
        })

        minTime = epochTimex._1
        maxTime = epochTimex._2
        System.out.println(s"${threadName} pull epoch time finished: min ${minTime} max ${maxTime}")

        // ---------- wait until all edge finish update timecost and parameters --------
        while(jedisHelper.isFinishedUpdate() == false){
          Thread.sleep(1000)
          System.out.println(s"${threadName} wait other edges to finish update parameters and time")
        }
        jedisHelper.pruneEnvir()
      }
      System.out.println(s"${edgeName} sign out!")
      jedisHelper.shutdown()
    }
  }

  def main(args: Array[String]): Unit = {
    import java.util.concurrent.{Executors, ExecutorService}
    val edgeNumber: Int = 10
    val threadPool = Executors.newFixedThreadPool(edgeNumber)
    try{
      for(idx <- 1 to edgeNumber){
        threadPool.execute(new ThreadDeom(("Thread_"+idx),totalEdge = edgeNumber))
      }
    }finally {
      threadPool.shutdown()
    }
  }
}
