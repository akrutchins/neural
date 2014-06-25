package edu.stanford.scalann.abstractunits

import edu.stanford.scalann.{Interface, NeuralNetwork}

/**
 * Handles all the common functionality for elements of the neural network
 */
abstract class AbstractUnit(network : NeuralNetwork) extends Serializable {

  def feedForward() {}
  def backProp() {}
  def saveGradient() {}

  val inputSize : Int
  val outputSize : Int

  def >>(unit : AbstractUnit) : AbstractUnit = {
    assert(outputSize == unit.inputSize, "Shorthand >> can only be used when output size matches input size, "+outputSize+" != "+inputSize)
    val interface : Interface = network.interface(outputSize)
    unit.setChildInterface(interface,0)
    setParentInterface(interface,0)
    unit
  }

  var parentInterface : Interface = null
  var childInterface : Interface = null

  def setParentInterface(interface : Interface, offset : Int) {
    assert(interface != null, "Cannot setParent to null")
    parentInterface = interface
    parentInterface.addView(this,offset,offset+outputSize-1)
  }

  def setChildInterface(interface : Interface, offset : Int) {
    assert(interface != null, "Cannot setChild to null")
    childInterface = interface
    childInterface.addView(this,offset,offset+inputSize-1)
  }
}
