package com.intel.analytics.bigdl.optim

import redis.clients.jedis.{BinaryJedis, Jedis, JedisPool, JedisPoolConfig, Tuple}
import redis.clients.jedis.exceptions.JedisConnectionException
import java.io._
import java.util.Base64
import java.nio.charset.StandardCharsets.UTF_8

import scala.collection.JavaConversions._
import scala.collection.JavaConversions.asScalaSet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.mutable.ArrayBuffer

class GoSGDHelper(host: String = "localhost",port: Int = 6379,
                  clusterMode:Boolean = false) extends Serializable {
  private val goPrefix: String = "GoSGD:"
  private val edgeIdKey: String = s"${goPrefix}ID"
  private val edgeIDSet: String = s"${goPrefix}EdgeIDSet"
  private var edgeID: Int = 1 // default 1
  private val finishCounter: String = s"${goPrefix}Counter"
  private val finishUpdater: String = s"${goPrefix}Updater"
  private val timeCostKey: String = s"${goPrefix}TimeCost"
  private var edgeName: String = ""
  private val nRoundKey: String = s"${goPrefix}newRound"

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
           |return edgeID
           |""".stripMargin
      jedis = jpool.getResource
      edgeID = jedis.eval(lua).toString.toInt
      edgeName = s"${goPrefix}edge:${edgeID}"
      edgeName
    }finally {
      if(jedis!=null){
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

  def insertScoreWeightxTime[T](epochTime:Double,edge: Int, score: Double,weights: Tensor[T]) = {
    try{
      val pushEdge = s"${goPrefix}edge:${edge}"
      val serWeights = SerializationUtil.serialize(weights)
      jedis = jpool.getResource
      val trans = jedis.multi()
      trans.zadd(timeCostKey,epochTime,edgeName)
      trans.zadd(pushEdge,score,serWeights)
      val finish = trans.incr(finishCounter)
      trans.exec()
      finish.get().toInt
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }
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

  def popScoreWeightxTime[T]() = {
    try{
      jedis = jpool.getResource
      val trans = jedis.multi()
      val sw = trans.zrangeWithScores(edgeName,0,-1)
      val ms = trans.zrangeWithScores(timeCostKey,0,-1)
      trans.incr(finishUpdater)

      trans.exec()

      val arybuf = new ArrayBuffer[(Double,Tensor[T])]
      sw.get().foreach(elem => {
        val tup = elem.asInstanceOf[Tuple]
        val score = tup.getScore
        val weights = SerializationUtil.deserialize[Tensor[T]](tup.getElement)
        arybuf.append(Tuple2(score,weights))
      })
      val timec = Tuple2[Double,Double](ms.get().head.getScore,ms.get().last.getScore)

      (arybuf,timec)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

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
           |  --redis.call('del','${goPrefix}edge:*')
           |  for _,k in ipairs(redis.call('keys','${goPrefix}edge:*')) do redis.call('del',k) end
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

  // nRoundKey == 0 finish save parameters and time
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
