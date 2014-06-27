package edu.stanford.scalann.optimization

import edu.stanford.scalann.data.DataSet
import breeze.linalg.{DenseVector, DenseMatrix}
import edu.stanford.scalann.WeightsManager

/**
  * This implements the raw AdaGrad scaling for gradient descent
  * as described in Chris Dyer's "Notes on AdaGrad"
  */
class AdaGradOptimizer extends AbstractOptimizer {

  override def optimize(dataSet: DataSet, debug: Boolean = false) : Double = {

    // Setup the running sums

    val squareGradientSum : Map[WeightsManager,DenseMatrix[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseMatrix.zeros[Double](wm.weights.rows,wm.weights.cols))) : _*)
    val squareInterceptSum : Map[WeightsManager,DenseVector[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseVector.zeros[Double](wm.intercepts.length))) : _*)

    for (t <- 1 to iterations) {
      dataSet.saveUnregularizedGradients()

      dataSet.weightsManagers.foreach(wm => {

        // Keep track of the gradient sum and sum of squares, for the AdaGrad process

        squareGradientSum(wm) :+= wm.gradients.map(x => x*x)
        squareInterceptSum(wm) :+= wm.interceptGradients.map(x => x*x)

        // Implement a step of the gradient descent algorithm

        wm.weights :-= (wm.gradients :* squareGradientSum(wm).map(Math.sqrt) :* alpha)
        wm.intercepts :-= (wm.interceptGradients :* squareInterceptSum(wm).map(Math.sqrt) :* alpha)
      })

      if (debug) {
        println(t+": "+dataSet.squaredError(lambda))
      }
    }
    dataSet.squaredError(lambda)
  }

}
