package edu.stanford.scalann

import edu.stanford.scalann.units._
import edu.stanford.scalann.abstractunits.AbstractUnit

/**
 * Central orchestration for the neural network architectures
 *
 * Created by Keenon on 6/22/14.
 */
class NeuralNetwork {

  // Public interface

  def inputUnit(size : Int) : InputUnit = {
    val u = new InputUnit(this, size)
    units = units :+ u
    u
  }
  def outputUnit(size : Int) : OutputUnit = {
    val u = new OutputUnit(this, size)
    units = units :+ u
    u
  }
  def logisticUnit(inputSize : Int, outputSize : Int) : LogisticUnit = {
    val u = new LogisticUnit(this, inputSize, outputSize)
    units = units :+ u
    u
  }
  def tanhUnit(inputSize : Int, outputSize : Int) : TanhUnit = {
    val u = new TanhUnit(this, inputSize, outputSize)
    units = units :+ u
    u
  }
  def linearUnit(inputSize : Int, outputSize : Int) : LinearUnit = {
    val u = new LinearUnit(this, inputSize, outputSize)
    units = units :+ u
    u
  }
  def interface(size : Int) : Interface = {
    new Interface(size)
  }

  def train() {
    clearGradient()

    feedForward()
    backProp()
    saveGradient()

    adjustWeights()
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

  def adjustWeights() {
    units.foreach(_.adjustWeights())
  }

  def saveGradient() {
    units.foreach(_.saveGradient())
  }

  def clearGradient() {
    units.foreach(_.clearGradient())
  }

  private var units : List[AbstractUnit] = List()
}
