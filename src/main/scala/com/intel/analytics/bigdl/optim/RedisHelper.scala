package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.concurrent.duration._
import scala.util.control.Breaks
import com.redis._
import serialization._

import scala.concurrent.Await

class RedisHelper(host: String = "localhost",port: Int = 6379,
                  database: Int = 0,secret: Option[Any] = None, clusterMode:Boolean = false) extends Serializable {
  val redis = new RedisClient(host = host,port = port,database = database,secret = secret)
  private val edgeId: String = "edgeID"
  private val edges: String = "edges"
  private val epochTime: String = "epochTime"
  private val parameters: String = "parameters"

  def register()={
    val edgeID = redis.incr(edgeId)
    redis.lpush(edges,edgeID).get
  }
  def checkout() = {
    redis.lpop(edges).get
  }

  def totalEdge()={
    redis.llen(edges).get
  }

  def zaddEpochTime(edgeName: String, time: Double) = {
    val totalEdge = redis.llen(edges).get
    val curFinishEdge = redis.zcard(epochTime).get
    if(totalEdge == curFinishEdge){
      redis.del(epochTime)
    }
    redis.zadd(epochTime,time,edgeName)
  }
  def zrangeMinEpochTime(timeout: Int = 120) = {
    var curCost: Double = 0.0
    val loop = new Breaks
    loop.breakable{
      while(true){
        //      val pipres = redis.pipelineNoMulti(List(
        //        {() => redis.llen("edges")},
        //        {() => redis.zcard("epochtime")}
        //      ))
        //      val result = pipres.map{a => Await.result(a.future, pairIntToDuration(1,SECONDS))}
        val totalEdge = redis.llen(edges).get
        val curFinishEdge = redis.zcard(epochTime).get
        if(totalEdge > curFinishEdge){
          Thread.sleep(100)
          curCost += 0.1
        }
        else{
          loop.break()
        }
        if(curCost > timeout){
          loop.break()
        }
      }
    }
    redis.zrangeWithScore(epochTime,0,1).get.head._2
  }

  def pushTParams[T](params: Tensor[T]) = {
    val totalEdge = redis.llen(edges).get
    val curFinishEdge = redis.llen(parameters).get
    if(totalEdge == curFinishEdge){
      redis.del(parameters)
    }
    val serl = SerializationUtil.serialize(params)
    redis.lpush(parameters,serl).get
  }

  def pullTParams[T](index: Int,timeout: Int = 240) = {
    var curCost: Double = 0.0
    val loop = new Breaks
    loop.breakable{
      while(true){
        val totalEdge = redis.llen(edges).get
        val curFinishEdge = redis.llen(parameters).get
        if(totalEdge > curFinishEdge){
          Thread.sleep(1000)
          curCost += 1
        }
        else{
          loop.break()
        }
        if(curCost > timeout){
          loop.break()
        }
      }
    }
    val stRes = redis.lindex(parameters,index).get
    SerializationUtil.deserialize[Tensor[T]](stRes)
  }

  def rpopTParams[T]() = {
    val stRes = redis.rpop(parameters).get
    SerializationUtil.deserialize[Tensor[T]](stRes)
  }

  def pushParams[T](params: List[Tensor[T]]) = {
    val totalEdge = redis.llen(edges).get
    val curFinishEdge = redis.llen(parameters).get
    if(totalEdge == curFinishEdge){
      redis.del(parameters)
    }
    val serl = SerializationUtil.serialize(params)
    redis.lpush(parameters,serl).get
  }

  def pullParams[T](index: Int,timeout: Int = 240) = {
    var curCost: Double = 0.0
    val loop = new Breaks
    loop.breakable{
      while(true){
        val totalEdge = redis.llen(edges).get
        val curFinishEdge = redis.llen(parameters).get
        if(totalEdge > curFinishEdge){
          Thread.sleep(1000)
          curCost += 1
        }
        else{
          loop.break()
        }
        if(curCost > timeout){
          loop.break()
        }
      }
    }
    val stRes = redis.lindex(parameters,index).get
    SerializationUtil.deserialize[List[Tensor[T]]](stRes)
  }

  def rpopParams[T]() = {
    val stRes = redis.rpop(parameters).get
    SerializationUtil.deserialize[List[Tensor[T]]](stRes)
  }

  def flushData() = {
    redis.flushdb
  }
  def close() = {
    redis.close()
  }
}
