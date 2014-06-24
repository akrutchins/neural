package edu.stanford.scalann.units

import breeze.linalg.DenseVector
import edu.stanford.scalann.NeuralNetwork
import edu.stanford.scalann.abstractunits.AbstractUnit

/**
 * The base unit of every network, a programmable input point
 */
class InputUnit(network : NeuralNetwork, size : Int) extends AbstractUnit(network) {

  var inputs : DenseVector[Double] = DenseVector.zeros[Double](size)
  var deltas : DenseVector[Double] = DenseVector.zeros[Double](size)

  override def feedForward() {
    parentInterface.activationView(this) := inputs
  }
  override def backProp() {
    deltas := parentInterface.deltaView(this)
  }

  override val outputSize: Int = size
  override val inputSize: Int = 0
}
