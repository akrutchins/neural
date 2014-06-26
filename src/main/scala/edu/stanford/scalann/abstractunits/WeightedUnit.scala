package edu.stanford.scalann.abstractunits

import breeze.linalg._
import edu.stanford.scalann.{WeightsManager, NeuralNetwork}

/**
 * Does the actual work of the neural network, calculating neuron activations
 */
abstract class WeightedUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int, initWeightsManager : WeightsManager = null) extends AbstractUnit(network) {

  val f : (Double => Double)
  val df : (Double => Double)

  override def feedForward() {
    val z = (weightsManager.weights * childInterface.activationView(this)) + weightsManager.intercepts
    parentInterface.activationView(this) := z.map(f)
  }

  override def backProp() {
    // Complete the parent's backprop calculation with the derivative of the inputs
    parentInterface.deltaView(this) :*= parentInterface.activationView(this).map(df)
    // Propagate down a partial backprop, without the derivative of the inputs (next layer will handle that)
    childInterface.deltaView(this) := (weightsManager.weights.t * parentInterface.deltaView(this))
  }

  override val outputSize: Int = initOutputSize
  override val inputSize: Int = initInputSize

  override def saveGradient() {
    val delta : DenseMatrix[Double] = parentInterface.deltaView(this).toDenseMatrix.t * childInterface.activationView(this).toDenseMatrix
    weightsManager.contributeGradient(delta, parentInterface.deltaView(this))
  }

  val weightsManager : WeightsManager = if (initWeightsManager != null) initWeightsManager else new WeightsManager(inputSize,outputSize)
}
