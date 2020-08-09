package com.intel.analytics.bigdl.optim

import redis.clients.jedis.{BinaryJedis, Jedis, JedisPool, JedisPoolConfig}
import redis.clients.jedis.exceptions.JedisConnectionException
import java.io._
import java.util.Base64
import java.nio.charset.StandardCharsets.UTF_8

import scala.collection.JavaConversions._
import scala.collection.JavaConversions.asScalaSet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

class JedisHelper(host: String = "localhost",port: Int = 6379,
                  clusterMode:Boolean = false) extends Serializable {
  private val edgeIdKey: String = "edgeID"
  private val edgePrefix: String = "edge_"
  private val edgePatternDel: String = s"for _,k in ipairs(redis.call('keys','${edgePrefix}*')) do redis.call('del',k) end"
  private val epochTimeKey: String = "epochTime"
  private val weightsKey: String = "modelWeights"
  private val getEpochTimeKey: String = "getEpochTime"
  private val saveWeightsKey: String = "saveWeights"
  private val getWeightsKey: String = "getModelWeights"
  private val initKey: String = new String(Base64.getEncoder.encode("initDecSlimDL".getBytes),UTF_8)

  private val jpool = new JedisPool(new JedisPoolConfig(),host,port)
  private var jedis: Jedis = null

  private var currentEdgeKey: String = ""
  private var currentEdgeIdx: Int = 0

  def init() = {
    try{
      jedis = jpool.getResource
      jedis.setnx(initKey,initKey)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def signIn() = {
    try{
      jedis = jpool.getResource
      val eid = jedis.incr(edgeIdKey).toInt
      currentEdgeKey = s"${edgePrefix}${eid}"
      currentEdgeIdx= eid - 1
      jedis.setnx(currentEdgeKey,"1")
      jedis.rpush(weightsKey,"'nil'")
      eid
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }
  def signOut() ={
    try{
      jedis = jpool.getResource
      val lua =
        s"""
           |redis.call('del','${currentEdgeKey}')
           |if tonumber(redis.call('exists','${getEpochTimeKey}')) == 1 then
           |   redis.call('decr','${getEpochTimeKey}')
           |end
           |if tonumber(redis.call('exists','${getWeightsKey}')) == 1 then
           |   redis.call('decr','${getWeightsKey}')
           |end
           |""".stripMargin
      jedis.eval(lua)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def zaddEpochTime(epochTime: Double) = {
    try{
      jedis = jpool.getResource
      jedis.zadd(epochTimeKey,epochTime,currentEdgeKey)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def lsetWeights[T](weights: Tensor[T]) = {
    try{
      jedis = jpool.getResource
      val ts = jedis.multi()
      val serWeights = SerializationUtil.serialize(weights)
      ts.lset(weightsKey,currentEdgeIdx,serWeights)
      ts.incr(saveWeightsKey)
      ts.exec()
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def currentEdgeNum() = {
    try{
      jedis = jpool.getResource
      jedis.keys(s"${edgePrefix}*").asInstanceOf[java.util.HashSet[String]].size()
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def isFinishedETSave():Boolean = {
    try{
      jedis = jpool.getResource
      val transaction = jedis.multi()
      transaction.keys(s"${edgePrefix}*")
      transaction.zcard(epochTimeKey)
      val resAry = transaction.exec().toArray()

      val edgeNum = resAry(0).asInstanceOf[java.util.HashSet[String]].size()
      val epochTimeNum = resAry(1).toString.toInt
      edgeNum == epochTimeNum
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def incrGetEpochTime() ={
    try{
      jedis = jpool.getResource
      jedis.incr(getEpochTimeKey)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }
  def incrGetWeights() = {
    try{
      jedis = jpool.getResource
      jedis.incr(getWeightsKey)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def isFinishedWeightSave():Boolean = {
    try{
      jedis = jpool.getResource
      val transaction = jedis.multi()
      transaction.keys(s"${edgePrefix}*")
      transaction.get(saveWeightsKey)
      val resAry = transaction.exec().toArray()

      val edgeNum = resAry(0).asInstanceOf[java.util.HashSet[String]].size()
      val epochTimeNum = resAry(1).toString.toInt
      edgeNum == epochTimeNum
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }

  def delEpochTimeKey()= {
    try{
      val lua =
        s"""
           |local edgeNum = #redis.call('keys','${edgePrefix}*')
           |local epochNum = tonumber(redis.call('get','${getEpochTimeKey}') or "0")
           |if edgeNum <= epochNum then
           |   redis.call('del','${epochTimeKey}','${getEpochTimeKey}')
           |   return 1
           |else
           |   return 0
           |end
           |""".stripMargin
      jedis = jpool.getResource
      jedis.eval(lua)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }
  def delWeightsKey()= {
    try{
      jedis = jpool.getResource
      val lua =
        s"""
           |local edgeNum = #redis.call('keys','${edgePrefix}*')
           |local epochNum = tonumber(redis.call('get','${getWeightsKey}') or "0")
           |if edgeNum <= epochNum then
           |   redis.call('del','${getWeightsKey}','${saveWeightsKey}')
           |   return 1
           |else
           |   return 0
           |end
           |""".stripMargin
      jedis.eval(lua)
    }finally {
      if(jedis != null){
        jedis.close()
      }
    }
  }


  def lindexWeightsByIndex[T](index: Int) = {
    try{
      jedis = jpool.getResource
      val idx = index % jedis.llen(weightsKey)
      val serWeiths = jedis.lindex(weightsKey,idx)
      SerializationUtil.deserialize[Tensor[T]](serWeiths)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def zrangeMinMaxEpochTime() = {
    try{
      jedis = jpool.getResource
      val ms = jedis.zrangeWithScores(epochTimeKey,0,-1)
      (ms.head.getScore,ms.last.getScore)
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  def closePool() = {
    if(!jpool.isClosed){
      jpool.close()
    }
  }

  def shutdown() = {
    signOut()
    closePool()
  }

  def destory() ={
    try{
      jedis = jpool.getResource
      val ts = jedis.multi()
      ts.eval(edgePatternDel)
      ts.del(weightsKey,epochTimeKey,initKey)
      ts.exec()
    }finally {
      if(jedis!=null){
        jedis.close()
      }
    }
  }

  override def toString: String = {
    s"${this.getClass.getName} ${currentEdgeKey}"
  }

}
