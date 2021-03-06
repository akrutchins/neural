package edu.stanford.scalann.data

import edu.stanford.scalann.{NeuralNetwork, WeightsManager}
import breeze.linalg._
import edu.stanford.scalann.abstractunits.WeightedUnit

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
    weightsManagers = weightsManagers.union(network.weightsManagers).distinct
  }

  def saveUnregularizedGradients() {
    weightsManagers.foreach(_.clearGradient())
    networks.foreach(_.saveGradient())
    weightsManagers.foreach(_.averageGradientsFromContributors())
  }

  def regularizeGradients(lambda : Double) {
    weightsManagers.foreach(wm => wm.gradients :+= (wm.weights*lambda))
  }

  def adjustWeights(alpha : Double) {
    weightsManagers.foreach(_.adjustWeights(alpha))
  }

  def squaredError(lambda : Double) : Double = {
    networks.foreach(_.feedForward())
    val squaredError : Double = sum(networks.map(_.squaredError()))
    val regularization : Double = sum(weightsManagers.map(wm => sum(wm.weights.map(w => w*w))))
    (squaredError + (regularization * lambda)) / 2
  }

  // A cute test using the definition of the derivative to check that all the computed derivatives are right on target

  def checkGradient(lambda : Double) : Double = {
    val backpropToBrute : Map[WeightsManager,WeightsManager] = Map(weightsManagers.map(w => (w, new WeightsManager(w.inputSize,w.outputSize))) : _*)

    // Do regularized backprop

    saveUnregularizedGradients()
    regularizeGradients(lambda)

    // Do a brute force approach

    val epsilon = 0.001

    for (w <- weightsManagers) {
      for (i <- 0 to w.inputSize-1) {
        for (o <- 0 to w.outputSize-1) {
          val baseWeight = w.weights(o,i)
          w.weights(o,i) = baseWeight + epsilon
          val jPlus = squaredError(lambda)
          w.weights(o,i) = baseWeight - epsilon
          val jMinus = squaredError(lambda)
          w.weights(o,i) = baseWeight
          backpropToBrute(w).gradients(o,i) = (jPlus - jMinus) / (2*epsilon)
        }
      }
      for (o <- 0 to w.outputSize-1) {
        val baseWeight = w.intercepts(o)
        w.intercepts(o) = baseWeight + epsilon
        val jPlus = squaredError(lambda)
        w.intercepts(o) = baseWeight - epsilon
        val jMinus = squaredError(lambda)
        w.intercepts(o) = baseWeight
        backpropToBrute(w).interceptGradients(o) = (jPlus - jMinus) / (2*epsilon)
      }
    }

    // Scale gradients by number of contributing units

    backpropToBrute.foreach(wm => {
      val contributors : Int = networks.flatMap(_.units).filter(_.isInstanceOf[WeightedUnit]).asInstanceOf[List[WeightedUnit]].count(_.weightsManager eq wm._1)
      wm._2.gradients /= contributors.asInstanceOf[Double]
      wm._2.interceptGradients /= contributors.asInstanceOf[Double]
    })

    // Print the differences

    backpropToBrute.foreach{
      case (bp,brute) =>
        println("--------")
        println("Derivative Delta:")
        println(bp.gradients - brute.gradients)
        println("Intercept Delta: "+(bp.interceptGradients.toDenseMatrix - brute.interceptGradients.toDenseMatrix))
    }

    max(
      backpropToBrute.map(pair => {
        val ws : Double = max((pair._1.gradients-pair._2.gradients).map(Math.abs))
        val is : Double = max((pair._1.interceptGradients-pair._2.interceptGradients).map(Math.abs))
        max(ws,is)
      })
    )
  }
}
