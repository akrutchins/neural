package edu.stanford.scalann

import org.scalatest._
import breeze.linalg._
import edu.stanford.scalann.units._

/**
 * Neural Network test cases
 */
class NeuralNetworkTests extends FlatSpec with Matchers {

  "A Trivial Neural Network" should "map inputs to outputs" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(4)
    val output : OutputUnit = neuralNet.outputUnit(4)
    input >> output

    input.inputs = DenseVector.ones[Double](4)

    neuralNet.feedForward()

    output.outputs should equal (DenseVector.ones[Double](4))
  }

  it should "combine inputs" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input1 : InputUnit = neuralNet.inputUnit(2)
    val input2 : InputUnit = neuralNet.inputUnit(2)
    val output : OutputUnit = neuralNet.outputUnit(4)
    val interface : Interface = neuralNet.interface(4)
    input1.setParentInterface(interface,0)
    input2.setParentInterface(interface,2)
    output.setChildInterface(interface,0)

    input1.inputs = DenseVector[Double](2,2)
    input2.inputs = DenseVector[Double](4,4)

    neuralNet.feedForward()

    output.outputs should equal (DenseVector[Double](2,2,4,4))
  }

  it should "split outputs" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(4)
    val output1 : OutputUnit = neuralNet.outputUnit(2)
    val output2 : OutputUnit = neuralNet.outputUnit(2)
    val interface : Interface = neuralNet.interface(4)
    input.setParentInterface(interface,0)
    output1.setChildInterface(interface,0)
    output2.setChildInterface(interface,2)

    input.inputs = DenseVector[Double](2,2,4,4)

    neuralNet.feedForward()

    output1.outputs should equal (DenseVector[Double](2,2))
    output2.outputs should equal (DenseVector[Double](4,4))
  }

  it should "consistently pattern crossing splits" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input1 : InputUnit = neuralNet.inputUnit(3)
    val input2 : InputUnit = neuralNet.inputUnit(1)
    val output1 : OutputUnit = neuralNet.outputUnit(1)
    val output2 : OutputUnit = neuralNet.outputUnit(3)
    val interface : Interface = neuralNet.interface(4)
    input1.setParentInterface(interface,0)
    input2.setParentInterface(interface,3)
    output1.setChildInterface(interface,0)
    output2.setChildInterface(interface,1)

    input1.inputs = DenseVector[Double](2,2,2)
    input2.inputs = DenseVector[Double](3)

    neuralNet.feedForward()

    output1.outputs should equal (DenseVector[Double](2))
    output2.outputs should equal (DenseVector[Double](2,2,3))
  }

  it should "propagate error derivatives back correctly" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val output : OutputUnit = neuralNet.outputUnit(2)
    input >> output

    input.inputs = DenseVector[Double](1,2)
    output.goldOutputs = DenseVector[Double](2,1)

    neuralNet.feedForward()
    neuralNet.backProp()

    output.outputs should equal (DenseVector[Double](1,2))
    input.deltas should equal (DenseVector[Double](-1,1)) // Direction of most rapid increase of the error
  }

  "A Single Layer Linear Neural Network" should "feedforward through an identity layer" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val linear : LinearUnit = neuralNet.linearUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> linear
    linear >> output

    input.inputs = DenseVector[Double](1,2)
    linear.weights(0,0) = 1
    linear.weights(0,1) = 0
    linear.weights(1,0) = 0
    linear.weights(1,1) = 1

    neuralNet.feedForward()

    output.outputs should equal (DenseVector[Double](1,2))
  }

  it should "backprop through an identity layer" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val linear : LinearUnit = neuralNet.linearUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> linear
    linear >> output

    input.inputs = DenseVector[Double](1,2)
    linear.weights(0,0) = 1
    linear.weights(0,1) = 0
    linear.weights(1,0) = 0
    linear.weights(1,1) = 1
    output.goldOutputs = DenseVector[Double](2,1)

    neuralNet.feedForward()
    neuralNet.backProp()

    output.outputs should equal (DenseVector[Double](1,2))
    input.deltas should equal (DenseVector[Double](-1,1))
  }

  it should "correct weights on a linear layer" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val linear : LinearUnit = neuralNet.linearUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> linear
    linear >> output

    input.inputs = DenseVector[Double](1,2)
    linear.weights(0,0) = 1
    linear.weights(0,1) = 2
    linear.weights(1,0) = 3
    linear.weights(1,1) = 4
    output.goldOutputs = DenseVector[Double](2,1)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      linear.adjustWeights()
    }

    error should be < 0.05
  }

  "A Single Layer Logistic Neural Network" should "correct weights on a logistic layer" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val logistic : LogisticUnit = neuralNet.logisticUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> logistic
    logistic >> output

    input.inputs = DenseVector[Double](1,2)
    logistic.weights(0,0) = 1
    logistic.weights(0,1) = 0
    logistic.weights(1,0) = 0
    logistic.weights(1,1) = 1
    output.goldOutputs = DenseVector[Double](0.75,0.5)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      logistic.adjustWeights()
    }

    error should be < 0.05
  }

  "A Double Layer Logistic Neural Network" should "correct weights on both logistic layers" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val logistic1 : LogisticUnit = neuralNet.logisticUnit(2,2)
    val logistic2 : LogisticUnit = neuralNet.logisticUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> logistic1
    logistic1 >> logistic2
    logistic2 >> output

    input.inputs = DenseVector[Double](1,2)
    logistic1.weights(0,0) = 1
    logistic1.weights(0,1) = 0
    logistic1.weights(1,0) = 0
    logistic1.weights(1,1) = 1
    logistic2.weights(0,0) = 1
    logistic2.weights(0,1) = 0
    logistic2.weights(1,0) = 0
    logistic2.weights(1,1) = 1
    output.goldOutputs = DenseVector[Double](0.75,0.5)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      logistic1.adjustWeights()
      logistic2.adjustWeights()
    }

    println("Logistic 1 weights: \n"+logistic1.weights)
    println("Logistic 2 weights: \n"+logistic2.weights)

    error should be < 0.05
  }

  "A Complex Logistic Neural Network" should "converge to the correct output value" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val logistic1 : LogisticUnit = neuralNet.logisticUnit(2,3)
    val logistic2 : LogisticUnit = neuralNet.logisticUnit(2,1)
    val logistic3 : LogisticUnit = neuralNet.logisticUnit(1,1)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> logistic1
    val logisticInterface : Interface = neuralNet.interface(3)
    logistic1.setParentInterface(logisticInterface,0)
    logistic2.setChildInterface(logisticInterface,0)
    logistic3.setChildInterface(logisticInterface,2)
    val outputInterface : Interface = neuralNet.interface(2)
    logistic2.setParentInterface(outputInterface,0)
    logistic3.setParentInterface(outputInterface,1)
    output.setChildInterface(outputInterface,0)

    input.inputs = DenseVector[Double](1,2)
    output.goldOutputs = DenseVector[Double](0.75,0.5)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      logistic1.adjustWeights()
      logistic2.adjustWeights()
      logistic3.adjustWeights()
    }

    println("Logistic 1 weights: \n"+logistic1.weights)
    println("Logistic 2 weights: \n"+logistic2.weights)
    println("Logistic 3 weights: \n"+logistic3.weights)

    error should be < 0.05
  }

  "A Single Layer Tanh Neural Network" should "correct weights on a tanh layer" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val tanh : TanhUnit = neuralNet.tanhUnit(2,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> tanh
    tanh >> output

    input.inputs = DenseVector[Double](1,2)
    tanh.weights(0,0) = 1
    tanh.weights(0,1) = 0
    tanh.weights(1,0) = 0
    tanh.weights(1,1) = 1
    output.goldOutputs = DenseVector[Double](-0.75,0.5)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      tanh.adjustWeights()
    }

    error should be < 0.05
  }

  "A Mixed Tanh and Logistic Neural Network" should "converge to the correct outputs" in {
    val neuralNet : NeuralNetwork = new NeuralNetwork
    val input : InputUnit = neuralNet.inputUnit(2)
    val logistic : LogisticUnit = neuralNet.logisticUnit(2,3)
    val tanh : TanhUnit = neuralNet.tanhUnit(3,2)
    val output : OutputUnit = neuralNet.outputUnit(2)

    input >> logistic
    logistic >> tanh
    tanh >> output

    input.inputs = DenseVector[Double](1,2)
    output.goldOutputs = DenseVector[Double](-0.75,0.5)

    var error : Double = 0.0

    for (i <- 0 to 100) {
      neuralNet.feedForward()
      error = (output.outputs - output.goldOutputs).map(x => Math.abs(x)).sum
      println(i+": "+output.outputs+" ~ "+error)
      neuralNet.backProp()
      tanh.adjustWeights()
    }

    error should be < 0.05
  }
}
