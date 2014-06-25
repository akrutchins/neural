package edu.stanford.scalann.data

import edu.stanford.scalann.{NeuralNetwork, WeightsManager}
import breeze.linalg._

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

  def train(iterations : Int) : Double = {
    for (i <- 0 to iterations) {
      update()
      println()
    }
    squaredError()
  }

  def squaredError() : Double = {
    sum(networks.map(_.squaredError()))
  }
}
