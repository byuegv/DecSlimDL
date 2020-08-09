package slpart.exam

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.optim._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.GoSGDHelper
import org.apache.spark.rdd.RDD
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}

object SCExam {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.WARN)
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("Spark Exam")
    val sc = new SparkContext(conf)
    Engine.init

    val tensorAry = Array.tabulate(5)(i => Array.tabulate(5)(j => Sample[Float](Tensor[Float](3,5).rand(),j)))
    val rdd = sc.parallelize(tensorAry)
    rdd.foreach(ten => ten.foreach(s => println(s.feature())))
    System.out.println(rdd.toString())


    val jpool = new JedisPool(new JedisPoolConfig(),"localhost",6379)
    var jedis: Jedis = null
    try{
      System.out.println("save rdd in Redis")
      jedis = jpool.getResource
      val serRdd = SerializationUtil.serialize(rdd.collect())
      jedis.set("RDD",serRdd)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }

    try{
      System.out.println("get rdd from Redis")
      jedis = jpool.getResource
      val serRdd = jedis.get("RDD")
      val derRdd = SerializationUtil.deserialize[Array[Array[Sample[Float]]]](serRdd)
      derRdd.foreach(ten => ten.foreach(s => println(s.feature())))
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }

    val ten2 = Tensor[Float](2,3).rand()
    System.out.println(ten2)
    System.out.println(ten2.asInstanceOf[Tensor[Double]])
    ten2.asInstanceOf[Tensor[Float]].add(2.0f)
    System.out.println(ten2)


    sc.stop()
  }

}
