package edu.stanford.scalann

import breeze.linalg._

/**
 * A central manager for multiple weighted units that share the same weights
 */
class WeightsManager(initInputSize : Int, initOutputSize : Int) {
  val weights : DenseMatrix[Double] = DenseMatrix.rand(initOutputSize,initInputSize) * 0.01
  val gradients : DenseMatrix[Double] = DenseMatrix.zeros[Double](initOutputSize,initInputSize)

  def adjustWeights() {
    weights :-= gradients
  }

  def clearGradient() {
    gradients :*= 0.0
  }
}
