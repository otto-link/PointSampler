/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

#include <iostream>

namespace ps
{

/**
 * @brief Randomly retains a fixed number of points from the input set.
 *
 * This function returns a subset of the input points of size `target_count`,
 * selected uniformly at random without replacement. If `target_count` is
 * greater than or equal to the number of input points, the full input is
 * returned.
 *
 * @tparam T           Scalar type (e.g., float or double)
 * @tparam N           Dimensionality of the space
 * @param  points       Input vector of points
 * @param  target_count Desired number of points in the output (≤ points.size())
 * @return              std::vector<Point<T, N>> containing `target_count`
 *                      randomly selected points
 *
 * @code
 * std::vector<Point<float, 2>> pts = ps::random<float, 2>(1000, {{0,1},{0,1}});
 * auto reduced = ps::random_rejection_filter(pts, 300); // Keep 300 points
 * @endcode
 *
 * @image html out_random_rejection_filter.csv.jpg
 */
template <typename T, std::size_t N>
std::vector<Point<T, N>> random_rejection_filter(const std::vector<Point<T, N>> &points,
                                                 std::size_t target_count)
{
  if (target_count >= points.size())
  {
    return points; // nothing to remove
  }

  std::vector<std::size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937       gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);

  // Take the first `target_count` shuffled indices
  std::vector<Point<T, N>> result;
  result.reserve(target_count);
  for (std::size_t i = 0; i < target_count; ++i)
  {
    result.push_back(points[indices[i]]);
  }

  return result;
}

/**
 * @brief Randomly retains a fraction of the input points.
 *
 * This is a convenience overload of `random_rejection_filter` that accepts a
 * floating-point `keep_fraction` instead of an absolute count. Internally, it
 * computes the number of points to retain and calls the count-based version.
 *
 * @tparam T            Scalar type (e.g., float or double)
 * @tparam N            Dimensionality of the space
 * @param  points        Input vector of points
 * @param  keep_fraction Fraction of points to retain (between 0.0 and 1.0)
 * @return               std::vector<Point<T, N>> containing `keep_fraction *
 *                       points.size()` randomly selected points
 *
 * @code
 * std::vector<Point<double, 3>> cloud = ps::random<double, 3>(10000,
 * {{-1,1},{-1,1},{-1,1}}); auto sparse = ps::random_rejection_filter(cloud, 0.25); //
 * Keep 25% of the points
 * @endcode
 */
template <typename T, std::size_t N>
std::vector<Point<T, N>> random_rejection_filter(const std::vector<Point<T, N>> &points,
                                                 float keep_fraction)
{
  assert(keep_fraction >= 0.0 && keep_fraction <= 1.0);
  std::size_t target_count = static_cast<std::size_t>(keep_fraction * points.size());
  return random_rejection_filter(points, target_count);
}

} // namespace ps