package slpart.exam

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.GossipGradHelper

object GossipExamer {

  def sampAryGenerater(n1: Int, n2: Int) = {
    val ary = Array.tabulate(n1)(i => Array.tabulate(n2)(j => {
      val feat = Tensor(2,3).rand()
      val lab = j+1
      Sample[Float](feat,lab)
    }))
    ary
  }

  class ThreadDeom(threadName: String,totalEdge: Int) extends Runnable{
    override def run(): Unit = {
      val jedisHelper = new GossipGradHelper("localhost",6379,false)
      val tp = scala.util.Random.nextInt(10)
      Thread.sleep((tp*1000).toLong)
      var edgeName = jedisHelper.signIn()
      var edgeID = jedisHelper.edgeID
      System.out.println(s"${threadName} sign in! ${jedisHelper.toString}")

      // wait for all edges sign in
      while(jedisHelper.currentEdges() < totalEdge){
        Thread.sleep(1000)
      }
      System.out.println("All Edges signed in!")
      // save samples
      var sampArys = sampAryGenerater(19,5)
//      jedisHelper.lsetSamples[Array[Array[Sample[Float]]]](sampArys)
      jedisHelper.hsetSamples(sampArys,batchSize = 4)

      var epochStartTime = System.nanoTime()
      // Epoch
      for(epoch <- 1 to 5){
        // 从其他edge拉取训练样本
        val selId = jedisHelper.selEdgeID(canBeSelf = false)
//        sampArys = jedisHelper.lindexSamples[Array[Array[Sample[Float]]]](selId)
        sampArys = jedisHelper.hvalsSamples[Array[Sample[Float]]](selId)

        var minTime = 12.0
        var maxTime = 12.0
        // iteration
        for(iteration <- 1 to 15){
          val _header= s"Epoch ${epoch} Iteration: ${iteration}  "
          if(iteration % 3 == 0){
            edgeID = jedisHelper.resetEdgeID()
            edgeName = jedisHelper.edgeName
            System.out.println(s"${_header} new edgeId: ${edgeID}, edgeName: ${edgeName}")
          }
          // ========== Training and aggregate parameters
          val etime = scala.util.Random.nextInt(10) + 1
          Thread.sleep((etime*1000).toLong)
          Thread.sleep(1200)

          val tensor = Tensor(5).randn()
          val oEdgeId = (edgeID + math.pow(2,iteration)).toInt % totalEdge + 1
          jedisHelper.lpushGradients[Float](tensor,Some(oEdgeId))
          System.out.println(s"${_header} ${threadName} save gradients finished ${tensor.toArray().mkString(",")}")

          // pull gradients
//          val otherEdgeId = jedisHelper.selEdgeID(canBeSelf = false)
          val otherEdgeId = (edgeID + totalEdge - math.pow(2,iteration)).toInt % totalEdge + 1
          val ograd = jedisHelper.brpopGradients[Float]()
          System.out.println(s"${_header} ${threadName} pull gradients: ${ograd.toArray().mkString(",")}")
          tensor.add(ograd)
          tensor.mul(0.5f)
        }
        val epochTimecost = (System.nanoTime()-epochStartTime) / 1e9
        epochStartTime = System.nanoTime()

        jedisHelper.insertoTime(epochTimecost)

        // ---------- wait until all edge finish save timecost --------
        while(jedisHelper.isFinishedSave() == false){
          Thread.sleep(1000)
          System.out.println(s"Epoch ${epoch} ${threadName} wait other edges to finish save time")
        }

        val  epochTimex = jedisHelper.popTimeCost()

        minTime = epochTimex._1
        maxTime = epochTimex._2
        System.out.println(s"${threadName} pull epoch time finished: min ${minTime} max ${maxTime}")

        // ---------- wait until all edge finish update timecost and parameters --------
        while(jedisHelper.isFinishedUpdate() == false){
          Thread.sleep(1000)
          System.out.println(s"Epoch ${epoch} ${threadName} wait other edges to finish update parameters and time")
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
