package edu.stanford.scalann.units

import breeze.linalg.DenseVector
import edu.stanford.scalann.NeuralNetwork
import edu.stanford.scalann.abstractunits.AbstractUnit

/**
 * Created by Keenon on 6/22/14.
 */
class OutputUnit(network : NeuralNetwork, size : Int) extends AbstractUnit(network) {

  var outputs : DenseVector[Double] = DenseVector.zeros[Double](size)
  var goldOutputs : DenseVector[Double] = DenseVector.zeros[Double](size)

  override val outputSize: Int = size
  override val inputSize: Int = size

  override def feedForward() {
    assert(childInterface.activationView(this) != null, "Cannot have a null activationView")
    assert(outputs != null, "Outputs cannot be null")
    outputs := childInterface.activationView(this)
  }
  override def backProp() {
    assert(goldOutputs != null, "Must have non-null gold outputs for backprop")
    assert(goldOutputs.length == outputs.length)
    childInterface.deltaView(this) := (outputs - goldOutputs)
  }
  override def adjustWeights() {}
}
