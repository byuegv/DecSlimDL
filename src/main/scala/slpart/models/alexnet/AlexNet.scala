package slpart.models.alexnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer


// parameters.element: 62378344

object AlexNetImageNet {
  def apply(classNum:Int = 1000,hasDropout: Boolean = false,hasbn: Boolean = false): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,224,224)).setName("reshape_1"))

    model.add(SpatialConvolution(3,96,11,11,4,4,3,3).setName("conv_1")) // (224 - 11 + 2*3)/4 + 1 = 55
    if(hasbn){
      model.add(SpatialBatchNormalization(96).setName("bn_1"))
    }
    model.add(ReLU(true).setName("relu_1"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("mp_1")) // (55 - 3 +0) / 2  + 1 = 27

    model.add(SpatialConvolution(96,256,5,5,1,1,2,2).setName("conv_2")) // (27 -5 + 2*2)/1 + 1 = 27
    if(hasbn){
      model.add(SpatialBatchNormalization(256).setName("bn_2"))
    }
    model.add(ReLU(true).setName("relu_2"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("mp_2")) // (27-3)/2 + 1 = 13

    model.add(SpatialConvolution(256,384,3,3,1,1,1,1).setName("conv_3")) // (13 -3 + 2*1)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(384).setName("bn_3"))
    }
    model.add(ReLU(true).setName("relu_3"))

    model.add(SpatialConvolution(384,384,3,3,1,1,1,1).setName("conv_4")) // (13 -3 + 2*2)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(384).setName("bn_4"))
    }
    model.add(ReLU(true).setName("relu_4"))

    model.add(SpatialConvolution(384,256,3,3,1,1,1,1).setName("conv_5")) // (13 -3 + 2*2)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(256).setName("bn_5"))
    }
    model.add(ReLU(true).setName("relu_5"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("mp_5")) // (13-3+0)/2 + 1 =6

    model.add(Reshape(Array(6*6*256)).setName("reshpae_2"))

    model.add(Linear(6*6*256,4096).setName("fc_6"))
    model.add(ReLU(true).setName("relu_6"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_1"))
    }
    model.add(Linear(4096,4096).setName("fc_7"))
    model.add(ReLU(true).setName("relu_7"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_2"))
    }

    model.add(Linear(4096,classNum).setName("fc_8"))
    model.add(LogSoftMax())
    model
  }
}


// parameters.element: 58297834
object AlexNetCifar10 {
  def apply(classNum:Int = 10,hasDropout: Boolean = false,hasbn: Boolean = false): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,32,32)).setName("reshape_1"))

    model.add(SpatialConvolution(3,96,6,6,1,1).setName("conv_1")) // (32 - 6 + 2*0)/1 + 1 = 27
    if(hasbn){
      model.add(SpatialBatchNormalization(96).setName("bn_1"))
    }
    model.add(ReLU(true).setName("relu_1"))
    //model.add(SpatialMaxPooling(3,3,2,2).setName("mp_1")) // (55 - 3 +0) / 2  + 1 = 27

    model.add(SpatialConvolution(96,256,5,5,1,1,2,2).setName("conv_2")) // (27 -5 + 2*2)/1 + 1 = 27
    if(hasbn){
      model.add(SpatialBatchNormalization(256).setName("bn_2"))
    }
    model.add(ReLU(true).setName("relu_2"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("mp_2")) // (27-3)/2 + 1 = 13

    model.add(SpatialConvolution(256,384,3,3,1,1,1,1).setName("conv_3")) // (13 -3 + 2*1)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(384).setName("bn_3"))
    }
    model.add(ReLU(true).setName("relu_3"))

    model.add(SpatialConvolution(384,384,3,3,1,1,1,1).setName("conv_4")) // (13 -3 + 2*2)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(384).setName("bn_4"))
    }
    model.add(ReLU(true).setName("relu_4"))

    model.add(SpatialConvolution(384,256,3,3,1,1,1,1).setName("conv_5")) // (13 -3 + 2*2)/1 + 1 = 13
    if(hasbn){
      model.add(SpatialBatchNormalization(256).setName("bn_5"))
    }
    model.add(ReLU(true).setName("relu_5"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("mp_5")) // (13-3+0)/2 + 1 =6

    model.add(Reshape(Array(6*6*256)).setName("reshpae_2"))

    model.add(Linear(6*6*256,4096).setName("fc_6"))
    model.add(ReLU(true).setName("relu_6"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_1"))
    }
    model.add(Linear(4096,4096).setName("fc_7"))
    model.add(ReLU(true).setName("relu_7"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_2"))
    }

    model.add(Linear(4096,classNum).setName("fc_8"))
    model.add(LogSoftMax())

    model
  }
}
// parameters: 5488794/5488106
object AlexNetCifar10Thinner4 {
  def apply(classNum:Int = 10,hasDropout: Boolean = false,hasbn: Boolean = false): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,32,32)).setName("reshape_1"))

    model.add(SpatialConvolution(3,24,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_1").setInitMethod(MsraFiller())) // (32 - 3 + 2*1)/1 + 1 = 32
    if(hasbn){
      model.add(SpatialBatchNormalization(24).setName("bn_1"))
    }
    model.add(ReLU(true).setName("relu_1"))

    model.add(SpatialConvolution(24,64,5,5,1,1,2,2,wRegularizer = L2Regularizer(1e-4)).setName("conv_2").setInitMethod(MsraFiller())) // 32x32
    if(hasbn){
      model.add(SpatialBatchNormalization(64).setName("bn_2"))
    }
    model.add(ReLU(true).setName("relu_2"))
    model.add(SpatialMaxPooling(3,3,2,2,1,1).setName("mp_2")) // (32-3+2)/2 + 1 = 16

    model.add(SpatialConvolution(64,96,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_3").setInitMethod(MsraFiller())) // 16x16
    if(hasbn){
      model.add(SpatialBatchNormalization(96).setName("bn_3"))
    }
    model.add(ReLU(true).setName("relu_3"))

    model.add(SpatialConvolution(96,96,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_4").setInitMethod(MsraFiller())) // 16
    if(hasbn){
      model.add(SpatialBatchNormalization(96).setName("bn_4"))
    }
    model.add(ReLU(true).setName("relu_4"))

    model.add(SpatialConvolution(96,64,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_5").setInitMethod(MsraFiller())) // 16
    if(hasbn){
      model.add(SpatialBatchNormalization(64).setName("bn_5"))
    }
    model.add(ReLU(true).setName("relu_5"))
    model.add(SpatialMaxPooling(3,3,2,2,1,1).setName("mp_5")) // (16-3+2)/2 + 1 = 8x8*64

    model.add(Reshape(Array(8*8*64)).setName("reshpae_2"))

    model.add(Linear(8*8*64,1024).setName("fc_6"))
    model.add(ReLU(true).setName("relu_6"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_1"))
    }
    model.add(Linear(1024,1024).setName("fc_7"))
    model.add(ReLU(true).setName("relu_7"))
    if(hasDropout){
      model.add(Dropout(0.5).setName("dropout_2"))
    }

    model.add(Linear(1024,classNum).setName("fc_8"))
    model.add(LogSoftMax())
    model
  }

  def main(args: Array[String]): Unit = {
    val model = AlexNetCifar10Thinner4(10,false,false)
    System.out.println(model.parameters()._1.map(_.nElement()).reduce(_ + _))
  }
}