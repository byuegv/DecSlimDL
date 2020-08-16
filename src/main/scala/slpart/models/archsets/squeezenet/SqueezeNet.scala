package slpart.models.archsets.squeezenet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.L2Regularizer

object SqueezeNetCifarV1 {
  object FireModule {
    def apply(squeezeInput: Int, squeezeOutput: Int, expandOutput: Int, hasBN: Boolean = true) = {
      val module = Sequential()

      module.add(SpatialConvolution(squeezeInput, squeezeOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        module.add(SpatialBatchNormalization(squeezeOutput))
      }
      module.add(ReLU(true))

      val concat = Concat(2)

      val left = Sequential()
      left.add(SpatialConvolution(squeezeOutput, expandOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        left.add(SpatialBatchNormalization(expandOutput))
      }
      left.add(ReLU(true))

      val right = Sequential()
      right.add(SpatialConvolution(squeezeOutput, expandOutput, 3, 3, 1, 1, 1, 1))
      if (hasBN) {
        right.add(SpatialBatchNormalization(expandOutput))
      }
      right.add(ReLU(true))

      concat.add(left)
      concat.add(right)
      module.add(concat)

      module
    }
  }

  def apply(numClass: Int = 10,hasBN: Boolean = false,hasDropout: Boolean = true): Module[Float] = {
    val squeezeNet = Sequential()
    squeezeNet.add(Reshape(Array(3,32,32)))
    squeezeNet.add(SpatialConvolution(3,96,3,3,2,2,1,1).setName("conv_1")) // 16 x 16 x96
    if(hasBN){squeezeNet.add(SpatialBatchNormalization(96))}
    squeezeNet.add(ReLU(true).setName("relu_1"))
    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_1")) // 8 x 8 x 96

    squeezeNet.add(FireModule(96,16,64,hasBN).setName("fire_2"))
    squeezeNet.add(FireModule(128,16,64,hasBN).setName("fire_3"))
    squeezeNet.add(FireModule(128,32,128,hasBN).setName("fire_4"))

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_4")) // 4 x 4

    squeezeNet.add(FireModule(256,32,128,hasBN).setName("fire_5"))
    squeezeNet.add(FireModule(256,48,192,hasBN).setName("fire_6"))
    squeezeNet.add(FireModule(384,48,192,hasBN).setName("fire_7"))
    squeezeNet.add(FireModule(384,64,256,hasBN).setName("fire_8"))   // 4 x 4

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_8")) // 2 x 2

    squeezeNet.add(FireModule(512,64,256,hasBN).setName("fire_9"))  // 2x2x512

    if(hasDropout){squeezeNet.add(Dropout(0.5).setName("drop_9"))}

    squeezeNet.add(SpatialConvolution(512,numClass,1,1,1,1,0,0).setName("conv_10")) // 2 x 2 x numClass
    if(hasBN){squeezeNet.add(SpatialBatchNormalization(numClass))}
    squeezeNet.add(ReLU(true))

    squeezeNet.add(SpatialAveragePooling(2,2,2,2,globalPooling = true).setName("avg_11"))
    squeezeNet.add(Reshape(Array(numClass)))
    squeezeNet.add(Linear(numClass,numClass).setName("fc_12"))

    squeezeNet
  }
}

object SqueezeNetCifarV2 {
  object FireModule {
    def apply(squeezeInput: Int, squeezeOutput: Int, expandOutput: Int, hasBN: Boolean = true) = {
      val module = Sequential()

      module.add(SpatialConvolution(squeezeInput, squeezeOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        module.add(SpatialBatchNormalization(squeezeOutput))
      }
      module.add(ReLU(true))

      val concat = Concat(2)

      val left = Sequential()
      left.add(SpatialConvolution(squeezeOutput, expandOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        left.add(SpatialBatchNormalization(expandOutput))
      }
      left.add(ReLU(true))

      val right = Sequential()
      right.add(SpatialConvolution(squeezeOutput, expandOutput, 3, 3, 1, 1, 1, 1))
      if (hasBN) {
        right.add(SpatialBatchNormalization(expandOutput))
      }
      right.add(ReLU(true))

      concat.add(left)
      concat.add(right)
      module.add(concat)

      module
    }
  }

  def apply(numClass: Int = 10,hasBN: Boolean = false,hasDropout: Boolean = true): Module[Float] = {
    val squeezeNet = Sequential()
    squeezeNet.add(Reshape(Array(3,32,32)))
    squeezeNet.add(SpatialConvolution(3,64,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_1").setInitMethod(Xavier)) // 32 x 32 x 64
    if(hasBN){squeezeNet.add(SpatialBatchNormalization(64))}
    squeezeNet.add(ReLU(true).setName("relu_1"))
    //    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_1")) // 16 x 16 x 96

    squeezeNet.add(FireModule(64,16,64,hasBN).setName("fire_2"))
    squeezeNet.add(FireModule(128,16,64,hasBN).setName("fire_3"))
    squeezeNet.add(FireModule(128,32,128,hasBN).setName("fire_4"))

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_4")) // 16 x 16

    squeezeNet.add(FireModule(256,32,128,hasBN).setName("fire_5"))
    squeezeNet.add(FireModule(256,48,192,hasBN).setName("fire_6"))
    squeezeNet.add(FireModule(384,48,192,hasBN).setName("fire_7"))
    squeezeNet.add(FireModule(384,64,256,hasBN).setName("fire_8"))   // 16 x 16

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_8")) // 8 x 8

    squeezeNet.add(FireModule(512,64,256,hasBN).setName("fire_9"))  // 8x8x512

    if(hasDropout){squeezeNet.add(Dropout(0.5).setName("drop_9"))}

    //squeezeNet.add(SpatialConvolution(512,numClass,1,1,1,1,0,0).setName("conv_10")) // 2 x 2 x numClass
    //if(hasBN){squeezeNet.add(SpatialBatchNormalization(numClass))}
    squeezeNet.add(ReLU(true))

    squeezeNet.add(SpatialAveragePooling(8,8,8,8,globalPooling = true).setName("avg_11"))
    squeezeNet.add(Reshape(Array(512)))
    squeezeNet.add(Linear(512,numClass).setName("fc_12"))

    squeezeNet
  }
}



object SqueezeNetCifarV3 {
  object FireModule {
    def apply(squeezeInput: Int, squeezeOutput: Int, expandOutput: Int, hasBN: Boolean = true) = {
      val module = Sequential()

      module.add(SpatialConvolution(squeezeInput, squeezeOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        module.add(SpatialBatchNormalization(squeezeOutput))
      }
      module.add(ReLU(true))

      val concat = Concat(2)

      val left = Sequential()
      left.add(SpatialConvolution(squeezeOutput, expandOutput, 1, 1, 1, 1, 0, 0))
      if (hasBN) {
        left.add(SpatialBatchNormalization(expandOutput))
      }
      left.add(ReLU(true))

      val right = Sequential()
      right.add(SpatialConvolution(squeezeOutput, expandOutput, 3, 3, 1, 1, 1, 1))
      if (hasBN) {
        right.add(SpatialBatchNormalization(expandOutput))
      }
      right.add(ReLU(true))

      concat.add(left)
      concat.add(right)
      module.add(concat)

      module
    }
  }

  def apply(numClass: Int = 10,hasBN: Boolean = false,hasDropout: Boolean = true): Module[Float] = {
    val squeezeNet = Sequential()
    squeezeNet.add(Reshape(Array(3,32,32)))
    squeezeNet.add(SpatialConvolution(3,64,3,3,1,1,1,1,wRegularizer = L2Regularizer(1e-4)).setName("conv_1").setInitMethod(Xavier)) // 32 x 32 x 64
    if(hasBN){squeezeNet.add(SpatialBatchNormalization(64))}
    squeezeNet.add(ReLU(true).setName("relu_1"))
    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_1")) // 16 x 16 x 64

    squeezeNet.add(FireModule(64,16,64,hasBN).setName("fire_2"))
    squeezeNet.add(FireModule(128,16,64,hasBN).setName("fire_3"))
    squeezeNet.add(FireModule(128,32,128,hasBN).setName("fire_4"))

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_4")) // 8 x 8

    squeezeNet.add(FireModule(256,32,128,hasBN).setName("fire_5"))
    squeezeNet.add(FireModule(256,48,192,hasBN).setName("fire_6"))
    squeezeNet.add(FireModule(384,48,192,hasBN).setName("fire_7"))
    squeezeNet.add(FireModule(384,64,256,hasBN).setName("fire_8"))   // 8 x 8

    squeezeNet.add(SpatialMaxPooling(3,3,2,2,1,1).setName("maxpool_8")) // 4 x 4

    squeezeNet.add(FireModule(512,64,256,hasBN).setName("fire_9"))  // 4x4x512

    if(hasDropout){squeezeNet.add(Dropout(0.5).setName("drop_9"))}

    //squeezeNet.add(SpatialConvolution(512,numClass,1,1,1,1,0,0).setName("conv_10")) // 2 x 2 x numClass
    //if(hasBN){squeezeNet.add(SpatialBatchNormalization(numClass))}
    squeezeNet.add(ReLU(true))

    squeezeNet.add(SpatialAveragePooling(4,4,4,4,globalPooling = true).setName("avg_11"))
    squeezeNet.add(Reshape(Array(512)))
    squeezeNet.add(Linear(512,numClass).setName("fc_12"))

    squeezeNet
  }
}
