package edu.stanford.scalann.units

import edu.stanford.scalann.{WeightsManager, NeuralNetwork}
import edu.stanford.scalann.abstractunits.WeightedUnit

/**
 * A weighted unit that uses Tanh to scale outputs
 */
class TanhUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int, initWeightsManager : WeightsManager = null) extends WeightedUnit(network,initInputSize,initOutputSize,initWeightsManager) {
  override val f : (Double => Double) = (x : Double) => Math.tanh(x)
  override val df : (Double => Double) = (f : Double) => 1-(f*f)
}
