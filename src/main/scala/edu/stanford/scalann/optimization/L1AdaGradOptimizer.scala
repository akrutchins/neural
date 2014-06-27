package edu.stanford.scalann.optimization

import edu.stanford.scalann.data.DataSet
import breeze.linalg.{DenseVector, DenseMatrix}
import edu.stanford.scalann.WeightsManager

/**
 * This implements the AdaGrad algorithm, adapted for L1 regularization, using Xiao's 2010 avg. gradient approach
 * as described in Chris Dyer's "Notes on AdaGrad"
 */
class L1AdaGradOptimizer extends AbstractOptimizer {

  def updateRule(timesteps : Int, linearSum : DenseMatrix[Double], squareSum : DenseMatrix[Double]) : DenseMatrix[Double] = {
    val adaGradScaling : DenseMatrix[Double] = squareSum :* (1/(alpha*timesteps))
    val signAndMask : DenseMatrix[Double] = linearSum.map(x => if (Math.abs(x)/timesteps <= lambda) 0.0 else { if (x > 0.0) -1.0 else 1.0 })
    signAndMask :* adaGradScaling :* linearSum.map(x => (Math.abs(x)/timesteps) - lambda)
  }

  override def optimize(dataSet: DataSet, debug: Boolean = false) : Double = {

    // Setup the running sums

    val squareGradientSum : Map[WeightsManager,DenseMatrix[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseMatrix.zeros[Double](wm.weights.rows,wm.weights.cols))) : _*)
    val squareInterceptSum : Map[WeightsManager,DenseVector[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseVector.zeros[Double](wm.intercepts.length))) : _*)
    val linearGradientSum : Map[WeightsManager,DenseMatrix[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseMatrix.zeros[Double](wm.weights.rows,wm.weights.cols))) : _*)
    val linearInterceptSum : Map[WeightsManager,DenseVector[Double]] = Map(dataSet.weightsManagers.map(wm => (wm, DenseVector.zeros[Double](wm.intercepts.length))) : _*)

    for (t <- 1 to iterations) {
      dataSet.saveUnregularizedGradients()

      dataSet.weightsManagers.foreach(wm => {

        // Keep track of the gradient sum and sum of squares, for the AdaGrad process

        squareGradientSum(wm) :+= wm.gradients.map(x => x*x)
        squareInterceptSum(wm) :+= wm.interceptGradients.map(x => x*x)
        linearGradientSum(wm) :+= wm.gradients
        linearInterceptSum(wm) :+= wm.interceptGradients

        // Implement a step of the gradient descent algorithm

        wm.weights := updateRule(t,linearGradientSum(wm),squareGradientSum(wm))
        wm.intercepts := updateRule(t,linearInterceptSum(wm).toDenseMatrix,squareInterceptSum(wm).toDenseMatrix).toDenseVector
      })

      if (debug) {
        println(t+": "+dataSet.squaredError(lambda))
      }
    }
    dataSet.squaredError(lambda)
  }

}
