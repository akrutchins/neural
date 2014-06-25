package edu.stanford.scalann

import edu.stanford.scalann.units._
import edu.stanford.scalann.abstractunits.{WeightedUnit, AbstractUnit}

import breeze.linalg._

/**
 * Central orchestration for the neural network architectures
 *
 * Created by Keenon on 6/22/14.
 */
class NeuralNetwork {

  // Public interface

  def inputUnit(size : Int) : InputUnit = {
    val u = new InputUnit(this, size)
    units :+= u
    u
  }
  def outputUnit(size : Int) : OutputUnit = {
    val u = new OutputUnit(this, size)
    units :+= u
    outputUnits :+= u
    u
  }
  def logisticUnit(inputSize : Int, outputSize : Int) : LogisticUnit = {
    val u = new LogisticUnit(this, inputSize, outputSize)
    units :+= u
    weightsManagers :+= u.weightsManager
    u
  }
  def tanhUnit(inputSize : Int, outputSize : Int) : TanhUnit = {
    val u = new TanhUnit(this, inputSize, outputSize)
    units :+= u
    weightsManagers :+= u.weightsManager
    u
  }
  def linearUnit(inputSize : Int, outputSize : Int) : LinearUnit = {
    val u = new LinearUnit(this, inputSize, outputSize)
    units :+= u
    weightsManagers :+= u.weightsManager
    u
  }
  def interface(size : Int) : Interface = {
    new Interface(size)
  }

  // Creates a duplicate unit, using the existing weights manager
  // Can clone units from another network

  def cloneUnit[T <: WeightedUnit](unit : T) : T = {
    val clone : T = unit match {
      case u : LogisticUnit => new LogisticUnit(this, u.inputSize, u.outputSize, u.weightsManager).asInstanceOf[T]
      case u : TanhUnit => new TanhUnit(this, u.inputSize, u.outputSize, u.weightsManager).asInstanceOf[T]
      case u : LinearUnit => new LinearUnit(this, u.inputSize, u.outputSize, u.weightsManager).asInstanceOf[T]
    }
    units :+= clone
    if (!weightsManagers.contains(clone.weightsManager)) weightsManagers :+= clone.weightsManager
    clone
  }

  def saveGradient() {
    feedForward()
    backProp()
    units.foreach(_.saveGradient())
  }

  def feedForward() {
    var waiting : List[AbstractUnit] = units
    while (waiting.size > 0) {
      val ready : List[AbstractUnit] = waiting.filter(u => {
        (u.childInterface == null) ||
        (waiting.count(c => {
          !(c eq u) && (c.parentInterface eq u.childInterface)
        }) == 0)
      })
      assert(ready.size > 0,"Feedforward ready list shouldn't be empty")
      waiting = waiting.diff(ready)
      ready.foreach(_.feedForward())
    }
  }

  def backProp() {
    var waiting : List[AbstractUnit] = units
    while (waiting.size > 0) {
      val ready : List[AbstractUnit] = waiting.filter(u => {
        (u.parentInterface == null) ||
        (waiting.count(p => {
          !(p eq u) && (p.childInterface eq u.parentInterface)
        }) == 0)
      })
      assert(ready.size > 0,"Backprop ready list shouldn't be empty")
      waiting = waiting.diff(ready)
      ready.foreach(_.backProp())
    }
  }

  def squaredError() : Double = {
    sum(outputUnits.map(_.squaredError()))
  }

  var units : List[AbstractUnit] = List()
  var outputUnits : List[OutputUnit] = List()
  var weightsManagers : List[WeightsManager] = List()
}
