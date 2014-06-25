package edu.stanford.scalann.data

import edu.stanford.scalann.{NeuralNetwork, WeightsManager}
import breeze.linalg._

object DataSet {
  def createSimpleDataset(parent : NeuralNetwork, input : List[DenseVector[Double]], output : List[DenseVector[Double]]) : DataSet = {
    assert(parent.inputUnits.length == 1, "createSimpleDataset only works with a single input unit")
    assert(parent.outputUnits.length == 1, "createSimpleDataset only works with a single output unit")
    assert(input.length > 0, "createSimpleDataset needs atleast one input instance")
    assert(output.length > 0, "createSimpleDataset needs atleast one output instance")
    assert(output.length == input.length, "createSimpleDataset needs the same number of input and output instances")
    assert(input.forall(v => v.length == parent.inputUnits(0).outputSize), "createSimpleDataset needs the size of input unit and input data to match")
    assert(output.forall(v => v.length == parent.outputUnits(0).inputSize), "createSimpleDataset needs the size of output unit and output data to match")

    val ds = new DataSet()
    (0 to input.length-1).foreach(i => {
      val clone = NeuralNetwork.clone(parent)
      clone.inputUnits(0).inputs := input(i)
      clone.outputUnits(0).goldOutputs := output(i)
      ds.addNetwork(clone)
    })
    ds
  }
}

/**
 * Contains all the different interesting things about training data, and a means to train it up.
 *
 * Can vary structure of network per datapoint, as well as link weight matrices as identical, both within structures and across datapoints
 */
class DataSet extends Serializable {

  // Holds a list of the weights managers that are used for this dataset

  var weightsManagers : List[WeightsManager] = List()

  // Holds a list of all the networks that are used in training. Each datapoint gets its own network

  var networks : List[NeuralNetwork] = List()

  def addNetwork(network : NeuralNetwork) {
    networks :+= network
    weightsManagers = weightsManagers.union(network.weightsManagers)
  }

  def update() {
    weightsManagers.foreach(_.clearGradient())
    networks.foreach(_.saveGradient())
    weightsManagers.foreach(_.adjustWeights())
  }

  def train(iterations : Int, debug : Boolean = false) : Double = {
    for (i <- 0 to iterations) {
      update()
      if (debug) {
        println("--------\nROUND "+i+"\n----------")
        println("ERROR: "+squaredError())
        for (net <- networks) {
          println("---")
          println("Inputs: "+net.inputUnits(0).inputs.toDenseMatrix)
          println("Gold Outputs: "+net.outputUnits(0).goldOutputs.toDenseMatrix)
          println("Actual Outputs: "+net.outputUnits(0).outputs.toDenseMatrix)
        }
      }
    }
    squaredError()
  }

  def squaredError() : Double = {
    sum(networks.map(_.squaredError()))
  }
}
