package edu.stanford.scalann.units

import edu.stanford.scalann.NeuralNetwork
import edu.stanford.scalann.abstractunits.WeightedUnit

/**
 * Created by Keenon on 6/24/14.
 */
class TanhUnit(network : NeuralNetwork, initInputSize : Int, initOutputSize : Int) extends WeightedUnit(network,initInputSize,initOutputSize) {
  override val f : (Double => Double) = (x : Double) => Math.tanh(x)
  override val df : (Double => Double) = (f : Double) => 1-(f*f)
}
