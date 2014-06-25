package edu.stanford.scalann.units

import edu.stanford.scalann.{WeightsManager, NeuralNetwork}
import edu.stanford.scalann.abstractunits.WeightedUnit

/**
 * A linear weighted unit. Useful only for testing that WeightedUnit is performing as expected
 */
class LinearUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int, initWeightsManager : WeightsManager = null) extends WeightedUnit(network,initInputSize,initOutputSize,initWeightsManager) {
  alpha = 0.1
  override val f : (Double => Double) = (x : Double) => x
  override val df : (Double => Double) = (f : Double) => 1
}

