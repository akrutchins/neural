package edu.stanford.scalann

import breeze.linalg._

/**
 * A central manager for multiple weighted units that share the same weights
 */
class WeightsManager(initInputSize : Int, initOutputSize : Int) {
  val inputSize = initInputSize
  val outputSize = initOutputSize

  var contributors : Int = 0

  val weights : DenseMatrix[Double] = DenseMatrix.rand(outputSize,inputSize) * 0.01
  val intercepts : DenseVector[Double] = DenseVector.zeros[Double](outputSize)

  val gradients : DenseMatrix[Double] = DenseMatrix.zeros[Double](outputSize,inputSize)
  val interceptGradients : DenseVector[Double] = DenseVector.zeros[Double](outputSize)

  def contributeGradient(g : DenseMatrix[Double], i : DenseVector[Double]) {
    this.synchronized {
      gradients += g
      interceptGradients += i
      contributors += 1
    }
  }

  def postProcessGradient(alpha : Double, lambda : Double) {
    this.synchronized {
      gradients :*= (alpha / contributors.asInstanceOf[Double])
      interceptGradients :*= (alpha / contributors.asInstanceOf[Double])
      gradients :+= (weights * lambda)
    }
  }

  def adjustWeights() {
    weights :-= gradients
    intercepts :-= interceptGradients
  }

  def clearGradient() {
    this.synchronized {
      gradients :*= 0.0
      interceptGradients :*= 0.0
      contributors = 0
    }
  }
}
