package edu.stanford.scalann

import breeze.linalg._
import scala.collection.parallel.mutable
import edu.stanford.scalann.abstractunits.AbstractUnit

/**
 * Allows readers and writers to all claim views of the same vector, to simplify interfaces
 *
 * Keeps track of mappings internally
 */
class Interface(size : Int) {
  private val activation : DenseVector[Double] = DenseVector.zeros[Double](size)
  private val delta : DenseVector[Double] = DenseVector.zeros[Double](size)

  private val map : mutable.ParMap[AbstractUnit,(Int,Int)] = mutable.ParMap()

  def activationView(unit : AbstractUnit) : DenseVector[Double] = {
    assert(map.contains(unit), "Interface must contain mapping for unit requesting an activation view")
    activation(map(unit)._1 to map(unit)._2)
  }

  def deltaView(unit : AbstractUnit) : DenseVector[Double] = {
    assert(map.contains(unit), "Interface must contain mapping for unit requesting an delta view")
    delta(map(unit)._1 to map(unit)._2)
  }

  def addView(unit : AbstractUnit, start : Int, end : Int) {
    assert(start >= 0 && end >= 0 && start <= end && end < size, "addView parameters must be within bounds: ("+start+","+end+"), size "+size)
    assert(!map.contains(unit), "Interface can't have multiple mappings for a unit")
    map.put(unit,(start,end))
  }
}
