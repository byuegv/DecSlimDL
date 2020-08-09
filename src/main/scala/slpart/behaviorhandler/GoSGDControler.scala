package com.intel.analytics.bigdl.optim

class GoSGDControler extends Serializable {
  var batchSize: Int = 64
  var redisHost: String = "localhost"
  var redisPort: Int = 6379
  var redisDatabase: Int = 0
  var redisSecret: Option[Any] = None
  var redisClusterMode:Boolean = false
  var fixedCriticalRatio: Option[Double] = None
  var edgeNum: Int = 2
  var epochInterval: Int = 1
  var exchangeProb: Double = 0.05

  override def toString: String = {
      s"${getClass.getName}:{" +
      s"\nbatchSize: ${batchSize}," +
      s"\nredisHost: ${redisHost}," +
      s"\nredisPort: ${redisPort}," +
      s"\nredisDatabase: ${redisDatabase}" +
      s"\nredisSecret: ${redisSecret}" +
      s"\nredisClusterMode: ${redisClusterMode}," +
      s"\nfixedCriticalRatio: ${fixedCriticalRatio}" +
      s"\nedgeNum: ${edgeNum}" +
      s"\nepochInterval: ${epochInterval}" +
      s"\nexchangeProb: ${exchangeProb}"
  }

  def this(batchSize: Int = 64,redisHost: String = "localhost", redisPort: Int=6379,
           redisDatabase: Int = 0,redisSecret: Option[Any] = None,
           redisClusterMode:Boolean = false,fixedCriticalRatio: Option[Double] = None,edgeNum: Int = 2,
           epochInterval: Int = 1,exchangeProb: Double = 0.05) {
    this()
    this.batchSize = batchSize
    this.redisHost = redisHost
    this.redisPort = redisPort
    this.redisDatabase = redisDatabase
    this.redisSecret = redisSecret
    this.redisClusterMode = redisClusterMode
    this.fixedCriticalRatio = fixedCriticalRatio
    this.edgeNum = edgeNum
    this.epochInterval = epochInterval
    this.exchangeProb = exchangeProb
  }
}
