package slpart.models.archsets.mobilenet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer

//1865002
object MobileNetV1 {
  /**
   * Build depthwise separable architecture: depth-wise -> point-wise(conv)
   * @param depParam depthwise param: inputChannel,outputChannel,depthMultiplier,kw,kh,sw,sh,pw,ph
   * @param hasBN
   */
  def depthwiseSeparable(depParam: Array[Int],hasBN: Boolean = false) = {
    require(depParam.length >= 9 ,"you should provide enough param for depthwise and point-wise layers")
    val cont = Sequential[Float]()
    cont.add(SpatialSeparableConvolution(nInputChannel = depParam(0),nOutputChannel = depParam(1),depthMultiplier = depParam(2),
      kW = depParam(3),kH = depParam(4),sW = depParam(5),sH = depParam(6),pW = depParam(7),pH = depParam(8)))
    if(hasBN) cont.add(SpatialBatchNormalization(depParam(1)))
    cont.add(ReLU(true))

    cont
  }

  /**
   * Build depthwise separable architecture: depth-wise -> point-wise(conv)
   * @param depParam depthwise param: inputChannel,outputChannel,depthMultiplier,kw,kh,sw,sh,pw,ph
   * @param convParam conv param: inputChannel,outputChannel,kw,kh,sw,sh,pw,ph
   * @param hasBN
   */
  def depthwiseSeparable2(depParam: Array[Int],convParam: Array[Int],hasBN: Boolean = false) = {
    require(depParam.length >= 9 && convParam.length >= 8,"you should provide enough param for depthwise and point-wise layers")
    val cont = Sequential[Float]()
    cont.add(SpatialSeparableConvolution(nInputChannel = depParam(0),nOutputChannel = depParam(1),depthMultiplier = depParam(2),
      kW = depParam(3),kH = depParam(4),sW = depParam(5),sH = depParam(6),pW = depParam(7),pH = depParam(8)))
    if(hasBN) cont.add(SpatialBatchNormalization(depParam(1)))
    cont.add(ReLU(true))

    //    cont.add(SpatialConvolution(nInputPlane = convParam(0) * depParam(2),nOutputPlane = convParam(1),kernelW = convParam(2),kernelH = convParam(3),
    //      strideW = convParam(4),strideH = convParam(5),padW = convParam(6),padH = convParam(7)))
    //    if(hasBN) cont.add(SpatialBatchNormalization(convParam(1)))
    //    cont.add(ReLU(true))
    cont
  }

  def apply(classNum:Int = 10,hasDropout: Boolean = false,hasBN: Boolean = false,shallow: Boolean = false,depthMultiplier: Int = 1): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,32,3,3,2,2,1,1).setName("conv_1")) // (32 - 3 + 2 * 1)/2 + 1 = 16
    if(hasBN) model.add(SpatialBatchNormalization(32).setName("sbn_1"))
    model.add(ReLU(true))

    model.add(depthwiseSeparable(depParam = Array(32,64,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_2")) // 16x16x64
    model.add(depthwiseSeparable(depParam = Array(64,128,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_3")) // 8x8x128
    model.add(depthwiseSeparable(depParam = Array(128,128,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_4")) // 8x8x128
    model.add(depthwiseSeparable(depParam = Array(128,256,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_5")) // 4x4x256
    model.add(depthwiseSeparable(depParam = Array(256,256,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_6")) // 4x4x256
    model.add(depthwiseSeparable(depParam = Array(256,512,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_5")) // 2x2x512

    if(!shallow){
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_7")) // 2x2x512
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_8")) // 2x2x512
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_9")) // 2x2x512
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_10")) // 2x2x512
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_11")) // 2x2x512
    }

    model.add(depthwiseSeparable(depParam = Array(512,1024,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_12")) // 1x1x1024
    model.add(depthwiseSeparable(depParam = Array(1024,1024,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_13")) // 1x1x1024

    model.add(SpatialAveragePooling(kW = 1,kH = 1,dW = 1,dH = 1,globalPooling = true).setName("globalAvgPool_14"))
    model.add(Reshape(Array(1024)))
    model.add(Linear(1024,classNum).setName("fc_15"))
    model
  }
  def main(args: Array[String]): Unit = {
    val model = MobileNetV1(classNum = 10,hasDropout = false,hasBN = false,shallow = true,depthMultiplier = 1)
    System.out.println(model.parameters()._1.map(_.nElement()).reduce(_ + _))
  }
}

//1871850/1865002
object MobileNetThinner {
  /**
   * Build depthwise separable architecture: depth-wise -> point-wise(conv)
   * @param depParam depthwise param: inputChannel,outputChannel,depthMultiplier,kw,kh,sw,sh,pw,ph
   * @param hasBN
   */
  def depthwiseSeparable(depParam: Array[Int],hasBN: Boolean = false) = {
    require(depParam.length >= 9 ,"you should provide enough param for depthwise and point-wise layers")
    val cont = Sequential[Float]()
    cont.add(SpatialSeparableConvolution(nInputChannel = depParam(0),nOutputChannel = depParam(1),depthMultiplier = depParam(2),
      kW = depParam(3),kH = depParam(4),sW = depParam(5),sH = depParam(6),pW = depParam(7),pH = depParam(8)))
    if(hasBN) cont.add(SpatialBatchNormalization(depParam(1)))
    cont.add(ReLU(true))

    cont
  }

  def apply(classNum:Int = 10,hasDropout: Boolean = false,hasBN: Boolean = false,shallow: Boolean = false,depthMultiplier: Int = 1): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,32,3,3,1,1,1,1).setName("conv_1")) // (32 - 3 + 2 * 1)/1 + 1 = 32
    if(hasBN) model.add(SpatialBatchNormalization(32).setName("sbn_1"))
    model.add(ReLU(true))

    model.add(depthwiseSeparable(depParam = Array(32,64,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_2")) // 16x16x64
    model.add(depthwiseSeparable(depParam = Array(64,128,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_3")) // 16x16x128
    model.add(depthwiseSeparable(depParam = Array(128,128,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_4")) // 16x16x128
    model.add(depthwiseSeparable(depParam = Array(128,256,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_5")) // 8x8x256
    model.add(depthwiseSeparable(depParam = Array(256,256,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_6")) // 8x8x256
    model.add(depthwiseSeparable(depParam = Array(256,512,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_5")) // 4x4x512

    if(!shallow){
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_7")) //
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_8")) //
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_9")) //
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_10")) //
      model.add(depthwiseSeparable(depParam = Array(512,512,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_11")) // 4x4x512
    }

    model.add(depthwiseSeparable(depParam = Array(512,1024,depthMultiplier,3,3,2,2,1,1),hasBN).setName("condw_12")) // 2x2x1024
    model.add(depthwiseSeparable(depParam = Array(1024,1024,depthMultiplier,3,3,1,1,1,1),hasBN).setName("condw_13")) // 2x2x1024

    model.add(SpatialAveragePooling(kW = 2,kH = 2,dW = 2,dH = 2,globalPooling = true).setName("globalAvgPool_14"))
    model.add(Reshape(Array(1024)))
    model.add(Linear(1024,classNum).setName("fc_15"))
    model
  }

  def main(args: Array[String]): Unit = {
    val model = MobileNetThinner(classNum = 10,hasDropout = false,hasBN = true,shallow = true,depthMultiplier = 1)
    System.out.println(model.parameters()._1.map(_.nElement()).reduce(_ + _))
    //    model.save("mobile.model")
  }

}


object MobileNetV1ImageNet {
  def apply(classes: Int = 1000,hashBN: Boolean = false): Module[Float] = {
    val mobileNet = Sequential[Float]()
    mobileNet.add(Reshape(Array(3,224,224)))
    mobileNet.add(SpatialConvolution(3,32,3,3,2,2,1,1).setName("conv_1")) // (224-3 + 2*1) / 2 + 1 = 112
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(32))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(32,32,1,3,3,1,1,1,1).setName("convdw_2")) // (112 - 3 + 2 * 1)/1 + 1 = 112
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(32))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(32,64,1,1,1,1,0,0).setName("conv_3")) // ( 112 - 1 +0)/1 + 1 = 112
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(64))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(64,64,1,3,3,2,2,1,1).setName("convdw_4")) // (112-3+2)/2+1 = 56
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(64))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(64,128,1,1,1,1,0,0).setName("conv_5")) // ( 56 - 1 +0)/1 + 1 = 56
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(128))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(128,128,1,3,3,1,1,1,1).setName("convdw_6")) // (56-3+2)/1+1 = 56
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(128))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(128,128,1,1,1,1,0,0).setName("conv_7")) // ( 56 - 1 +0)/1 + 1 = 56
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(128))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(128,128,1,3,3,2,2,1,1).setName("convdw_8")) // (56-3+2)/2+1 = 28
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(128))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(128,256,1,1,1,1,0,0).setName("conv_9")) // ( 28 - 1 +0)/1 + 1 = 28
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(256))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(256,256,1,3,3,1,1,1,1).setName("convdw_10")) // (28-3+2)/1+1 = 28
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(256))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(256,256,1,1,1,1,0,0).setName("conv_11")) // ( 28 - 1 +0)/1 + 1 = 28
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(256))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(256,256,1,3,3,2,2,1,1).setName("convdw_12")) // (28-3+2)/2+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(256))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(256,512,1,1,1,1,0,0).setName("conv_13")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,1,1,1,1).setName("convdw_14")) // (14-3+2)/1+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,512,1,1,1,1,0,0).setName("conv_15")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,1,1,1,1).setName("convdw_16")) // (14-3+2)/1+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,512,1,1,1,1,0,0).setName("conv_17")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,1,1,1,1).setName("convdw_18")) // (14-3+2)/1+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,512,1,1,1,1,0,0).setName("conv_19")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,1,1,1,1).setName("convdw_20")) // (14-3+2)/1+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,512,1,1,1,1,0,0).setName("conv_21")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,1,1,1,1).setName("convdw_22")) // (14-3+2)/1+1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,512,1,1,1,1,0,0).setName("conv_23")) // ( 14 - 1 +0)/1 + 1 = 14
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(512,512,1,3,3,2,2,1,1).setName("convdw_24")) // (14-3+2)/2+1 = 7
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(512))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(512,1024,1,1,1,1,0,0).setName("conv_25")) // ( 7 - 1 +0)/1 + 1 = 7
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(1024))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialSeparableConvolution(1024,1024,1,3,3,1,1,1,1).setName("convdw_26")) // (7-3+2)/1+1 = 7
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(1024))}
    mobileNet.add(ReLU(true))
    mobileNet.add(SpatialConvolution(1024,1024,1,1,1,1,0,0).setName("conv_27")) // ( 7 - 1 +0)/1 + 1 = 7
    if(hashBN){ mobileNet.add(SpatialBatchNormalization(1024))}
    mobileNet.add(ReLU(true))

    mobileNet.add(SpatialAveragePooling(7,7,7,7,0,0,globalPooling = true).setName("avgpool_28")) // (7-7+0)/7+1 = 1

    mobileNet.add(Reshape(Array(1024)))
    mobileNet.add(Linear(1024,classes).setName("fc_29")) // 1x1x1000
    mobileNet.add(SoftMax())
  }
}
