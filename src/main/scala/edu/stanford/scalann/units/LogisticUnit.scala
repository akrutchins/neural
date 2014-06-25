package edu.stanford.scalann.units

import edu.stanford.scalann.{WeightsManager, NeuralNetwork}
import edu.stanford.scalann.abstractunits.WeightedUnit

/**
 * A weighted unit that applies a logistic function at the end
 */
class LogisticUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int, initWeightsManager : WeightsManager = null) extends WeightedUnit(network,initInputSize,initOutputSize,initWeightsManager) {
  override val f : (Double => Double) = (x : Double) => 1 / (1 + Math.exp(-x))
  override val df : (Double => Double) = (f : Double) => f*(1-f)
}
