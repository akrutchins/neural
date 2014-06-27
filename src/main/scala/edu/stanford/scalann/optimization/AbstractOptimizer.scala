package edu.stanford.scalann.optimization

import edu.stanford.scalann.data.DataSet

/**
 * Creates the abstract implementation for all the optimizer classes to be implemented
 */
abstract class AbstractOptimizer {
  var alpha : Double = 1.0
  var lambda : Double = 0.0
  var iterations : Int = 1000
  def optimize(dataSet : DataSet, debug : Boolean = false) : Double
}
