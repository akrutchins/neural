package edu.stanford.scalann.abstractunits

import breeze.linalg._
import edu.stanford.scalann.NeuralNetwork

/**
 * Does the actual work of the neural network, calculating neuron activations
 */
abstract class WeightedUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int) extends AbstractUnit(network) {

  val weights : DenseMatrix[Double] = DenseMatrix.rand(initOutputSize,initInputSize) * 0.01

  val f : (Double => Double)
  val df : (Double => Double)

  var alpha : Double = 2.0

  override def feedForward() {
    val z = weights * childInterface.activationView(this)
    parentInterface.activationView(this) := z.map(f)
  }

  override def backProp() {
    // Complete the parent's backprop calculation with the derivative of the inputs
    parentInterface.deltaView(this) :*= parentInterface.activationView(this).map(df)
    // Propagate down a partial backprop, without the derivative of the inputs (next layer will handle that)
    childInterface.deltaView(this) := (weights.t * parentInterface.deltaView(this))
  }

  override def adjustWeights() {
    val delta : DenseMatrix[Double] = parentInterface.deltaView(this).toDenseMatrix.t * childInterface.activationView(this).toDenseMatrix
    delta :*= alpha
    weights :-= delta
  }

  override val outputSize: Int = initOutputSize
  override val inputSize: Int = initInputSize
}
