package edu.stanford.scalann.optimization

import edu.stanford.scalann.data.DataSet

/**
 * The most basic of all optimizers, this just blindly follows the gradient around at a fixed rate
 */
class GradientDescentOptimizer extends AbstractOptimizer {

  override def optimize(dataSet: DataSet, debug: Boolean = false) : Double = {
    for (i <- 0 to iterations) {
      dataSet.saveUnregularizedGradients()
      dataSet.regularizeGradients(lambda)
      dataSet.adjustWeights(alpha)
      if (debug) println(i+": "+dataSet.squaredError(lambda))
    }
    dataSet.squaredError(lambda)
  }

}
