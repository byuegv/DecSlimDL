package com.intel.analytics.bigdl.optim

import java.io._
import java.util.Base64
import java.nio.charset.StandardCharsets.UTF_8

object SerializationUtil {
  def serialize[T](value: T): String = {
    val stream: ByteArrayOutputStream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(value)
    oos.close()
    /*new String(
      Base64.getEncoder().encode(stream.toByteArray()),
      UTF_8
    )*/
    Base64.getEncoder().encodeToString(stream.toByteArray())
  }

  def deserialize[T](str: String) = {
//    val bytes = Base64.getDecoder().decode(str.getBytes(UTF_8))
    val bytes = Base64.getDecoder().decode(str)
    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
    val value = ois.readObject().asInstanceOf[T]
    ois.close()
    value
  }
}