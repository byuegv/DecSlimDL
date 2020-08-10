package com.intel.analytics.bigdl.optim

import redis.clients.jedis.{BinaryJedis, Jedis, JedisPool, JedisPoolConfig, Tuple}
import redis.clients.jedis.exceptions.JedisConnectionException
import java.io._
import java.util.Base64
import java.nio.charset.StandardCharsets.UTF_8

import com.intel.analytics.bigdl.dataset.Sample

import scala.collection.JavaConversions._
import scala.collection.JavaConversions.asScalaSet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.mutable.ArrayBuffer

class GossipGradHelper(host: String = "localhost",port: Int = 6379,
                  clusterMode:Boolean = false) extends Serializable {
  private val gosPrefix: String = "GossipGrad:"
  private val edgeIdKey: String = s"${gosPrefix}ID"
  private val edgeIDSet: String = s"${gosPrefix}EdgeIDSet"
  private val tpedgeIDSet: String = s"${gosPrefix}TPEdgeIDSet"
  var edgeID: Int = 1 // default 1
  private val finishCounter: String = s"${gosPrefix}Counter"
  private val finishUpdater: String = s"${gosPrefix}Updater"
  private val timeCostKey: String = s"${gosPrefix}TimeCost"
  var edgeName: String = ""
  private val nRoundKey: String = s"${gosPrefix}newRound"
  private val sampleListKey: String = s"${gosPrefix}samples"

  // Jedis 连接池
  private val jpool = new JedisPool(new JedisPoolConfig(),host,port)
  private var jedis: Jedis = null

  // sign in a new edge and return it's name
  def signIn() = {
    try{
      val lua =
        s"""
           |local edgeID = redis.call('incr','${edgeIdKey}')
           |redis.call('sadd','${edgeIDSet}',edgeID)
           |redis.call('lpush','${sampleListKey}','nil')
           |return edgeID
           |""".stripMargin
      jedis = jpool.getResource
      edgeID = jedis.eval(lua).toString.toInt
      edgeName = s"${gosPrefix}edge:${edgeID}"
      edgeName
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def resetEdgeID() = {
    try{
      val lua =
        s"""if redis.call('exists','${tpedgeIDSet}') == 0 then
           |  redis.call('sunionstore','${tpedgeIDSet}','${edgeIDSet}')
           |end
           |local id = redis.call('spop','${tpedgeIDSet}')
           |return id
           |""".stripMargin
      jedis = jpool.getResource
      edgeID = jedis.eval(lua).toString.toInt
      edgeName = s"${gosPrefix}edge:${edgeID}"
      edgeID
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def currentEdges() = {
    try{
      jedis = jpool.getResource
      jedis.scard(edgeIDSet).toInt
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def lpushGradients[T](gradients: Tensor[T],edge: Option[Int] = None) = {
    try{
      val serGrad = SerializationUtil.serialize(gradients)
      jedis = jpool.getResource
      var pushEdge = edgeName
      if(edge.isDefined){
        pushEdge = s"${gosPrefix}edge:${edge.get}"
      }
      jedis.lpush(edgeName,serGrad)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def brpopGradients[T](edge: Option[Int] = None) = {
    try{
      var popEdge = edgeName
      if(edge.isDefined){
        popEdge = s"${gosPrefix}edge:${edge.get}"
      }
      jedis = jpool.getResource
      val serGrad = jedis.brpop(0,popEdge)
//      System.out.println(serGrad.length)
      val grad = SerializationUtil.deserialize[Tensor[T]](serGrad.last)
      grad
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }



  def lsetSamples[T](samples: T) = {
    try{
      val serSamples = SerializationUtil.serialize(samples)
      jedis = jpool.getResource
      jedis.lset(sampleListKey,edgeID-1,serSamples)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def lindexSamples[T](index: Long) = {
    try{
      jedis = jpool.getResource
      val serSamples = jedis.lindex(sampleListKey,index-1)
      SerializationUtil.deserialize[T](serSamples)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  // 每个epoch等到所有edge运行完成，然后计算各个edge的选取比例
  def insertoTime(epochTime:Double) = {
    try{
      jedis = jpool.getResource
      val trans = jedis.multi()
      trans.zadd(timeCostKey,epochTime,edgeName)
      val finish = trans.incr(finishCounter)
      trans.exec()
      finish.get().toInt
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }
  // return (minTimeCost,maxTimeCost)
  def popTimeCost[T]() = {
    try{
      jedis = jpool.getResource
      val trans = jedis.multi()
      val ms = trans.zrangeWithScores(timeCostKey,0,-1)
      trans.incr(finishUpdater)
      trans.exec()
      val timec = Tuple2[Double,Double](ms.get().head.getScore,ms.get().last.getScore)

      timec
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  // nRoundKey == 0 finish time cost
  // nRoundKey == 1 finish update
  // nRoundKey == 2 finish prune env
  def isFinishedSave():Boolean = {
    try{
      val lua =
        s"""if redis.call('exists','${nRoundKey}') == 1 and redis.call('get','${nRoundKey}') == '0' then
           |  return 1
           |end
           |local counter = redis.call('get','${finishCounter}')
           |local edgeNum = redis.call('scard','${edgeIDSet}')
           |if tonumber(counter) == edgeNum then
           |  redis.call('setnx','${nRoundKey}','0')
           |  if redis.call('get','${nRoundKey}') == '2' then
           |    redis.call('set','${nRoundKey}','0')
           |  end
           |
           |  return 1
           |else
           |  return 0
           |end
           |""".stripMargin
      jedis = jpool.getResource
      val flag = jedis.eval(lua).toString.toInt
      flag == 1
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  // nRoundKey == 0 finish save time cost
  // nRoundKey == 1 finish update
  // nRoundKey == 2 finish prune env
  def isFinishedUpdate():Boolean = {
    try{
      val lua =
        s"""
           |local round = redis.call('get','${nRoundKey}')
           |if round == '1' or round == '2' then
           |  return 1
           |end
           |local updater = redis.call('get','${finishUpdater}')
           |local edgeNum = redis.call('scard','${edgeIDSet}')
           |if tonumber(updater) == edgeNum then
           |  if redis.call('get','${nRoundKey}') == '0' then
           |    redis.call('set','${nRoundKey}','1')
           |  end
           |
           |  return 1
           |else
           |  return 0
           |end
           |""".stripMargin
      jedis = jpool.getResource
      val flag = jedis.eval(lua).toString.toInt
      flag == 1
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  // nRoundKey == 0 finish save time cost
  // nRoundKey == 1 finish update
  // nRoundKey == 2 finish prune env
  def pruneEnvir() = {
    try{
      val lua =
        s"""
           |local round = redis.call('get','${nRoundKey}')
           |if round == '2' then
           |  return 1
           |elseif round == '1' then
           |  -- redis.call('del','${finishCounter}','${finishUpdater}','${timeCostKey}')
           |  redis.call('set','${finishCounter}',0)
           |  redis.call('set','${finishUpdater}',0)
           |  redis.call('del','${timeCostKey}')
           |  -- redis.call('del','${gosPrefix}edge:*')
           |  -- for _,k in ipairs(redis.call('keys','${gosPrefix}edge:*')) do redis.call('del',k) end
           |  redis.call('set','${nRoundKey}','2')
           |  return 1
           |else
           |  return 0
           |end
           |""".stripMargin
      jedis = jpool.getResource
      val flag = jedis.eval(lua).toString.toInt
      flag == 1
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def selEdgeID(canBeSelf: Boolean = false) = {
    try{
      jedis = jpool.getResource
      var selId = jedis.srandmember(edgeIDSet).toInt
      while(selId == edgeID && (!canBeSelf)){
        selId = jedis.srandmember(edgeIDSet).toInt
      }
      selId
    }finally{
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def shutdown() = {
    try{
      val lua =
        s"""
           |local edgeNum = redis.call('scard','${edgeIDSet}')
           |if edgeNum == 1 then
           |  redis.call('del','${edgeIdKey}','${nRoundKey}','${edgeIDSet}')
           |  redis.call('del','${finishCounter}')
           |  redis.call('del','${finishUpdater}')
           |  redis.call('del','${sampleListKey}')
           |  redis.call('del','${edgeName}')
           |else
           |  redis.call('srem','${edgeIDSet}',${edgeID})
           |  redis.call('del','${edgeName}')
           |end
           |""".stripMargin
      jedis = jpool.getResource
      jedis.eval(lua)
    }finally{
      if(jedis != null){
        jedis.close()
      }
    }
    if(!jpool.isClosed){
      jpool.close()
    }
  }

  override def toString: String = {
    s"${this.getClass.getName} ${edgeName}"
  }

}
