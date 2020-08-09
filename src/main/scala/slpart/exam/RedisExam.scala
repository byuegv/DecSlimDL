package slpart.exam

import com.redis._
import serialization._
import slpart.models.lenet.LeNet5
import Parse.Implicits._


object RedisExam {
  private val edgeId: String = "edgeID"
  private val edges: String = "edges"
  private val epochTime: String = "epochTime"
  private val parameters: String = "parameters"

  def main(args: Array[String]): Unit = {
    val redis = new RedisClient("localhost",6379)

    val lua1 =
      s""" if redis.call('exists','${edges}') == 0 then
         |    redis.call('set','${edges}',0)
         | else
         |    redis.call('incr','${edges}')
         | end
         |""".stripMargin
    System.out.println(lua1)

    val lua2 =
      s""" if redis.call('exists',KEYS[1]) == 0 then
         |    redis.call('set',KEYS[1],0)
         | else
         |    redis.call('incr',KEYS[1])
         | end
         |""".stripMargin
    System.out.println(lua2)

    val lua3 =
      s""" if redis.call('exists',KEYS[1]) == 1 and redis.call('exists',KEYS[2]) == 1 then
         |    local a = redis.call('get',KEYS[1])
         |    local b = redis.call('get',KEYS[2])
         |    return a == b
         | else
         |    return 0
         | end
         |""".stripMargin
    System.out.println(lua3)

    val res = redis.evalBulk[Int](lua3,keys = List[String](edges,edges),args=List[String]()).get
    System.out.println(s"Redis result: ${res}")


    /*redis.set("test","hello scala-redis!")
    System.out.println(redis.get("test").get)
    val tensor_1 = Tensor(3,4).randn(0,1.0)
    System.out.println((s"tensor_1:\n${tensor_1}"))

    redis.set("tensor_1",tensor_1)
    val rten_1 = redis.get("tensor_1").get
    System.out.println(s"tensor_1 from redis:\n${rten_1}")

    val tensor_2 = Tensor(2,3).rand()
    System.out.println(s"tensor_2:\n${tensor_2}")
    val ary = Array(tensor_1,tensor_2)

    redis.set("stensor_1",SerializeUtil.serialize(tensor_1))
    val dst1 = SerializeUtil.deserialize[Tensor[Float]](redis.get("stensor_1").get)
    dst1.add(2.0f)
    System.out.println(s"dser tensor_1:\n${dst1}")


    val lenet = LeNet5(10)
    val pt = lenet.getParametersTable()
    redis.del("list_1")
    redis.lpush("list_1",SerializeUtil.serialize(pt))

    val rpt = SerializeUtil.deserialize[Table](redis.lindex("list_1",0).get)
    System.out.println(s"rpt:\n${rpt.get[Table]("conv_1").get("bias")}")

    //val pary = Array(1,2,3,4,9).toList
    val pary = lenet.parameters()._2.toList
    redis.del("pary")
    redis.lpush("pary",SerializeUtil.serialize(pary))
    val rpary = SerializeUtil.deserialize[List[Tensor[Float]]](redis.lindex("pary",0).get)
    val res = pary.zip(rpary).map(x => x._1.add(x._2))
    System.out.println(res.mkString("\n"))*/

   /* //val pary = Array(1,2,3,4,9).toList
    val pary = lenet.parameters()._2
    val tpxx = pary.zipWithIndex.map(x => (x._2,x._1)).toMap
//    System.out.println(pary)
    redis.del("pary")
    redis.lpush("pary",SerializeUtil.serialize(tpxx))
    val rpary = SerializeUtil.deserialize[Map[Int,Tensor[Float]]](redis.lindex("pary",0).get)
      .toArray.sortBy(_._1).map(_._2)
    val res = pary.zip(rpary).map(x => x._1.add(x._2))
    System.out.println(res.mkString("\n"))*/

  }
}
