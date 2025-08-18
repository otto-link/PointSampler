/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/jittered_grid.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Generates a point set via importance resampling from a quasi-random
 * oversampled grid.
 *
 * This function uses a Halton sequence to create an oversampled set of
 * candidate points in the domain. Each point is assigned a weight based on the
 * provided density function. A discrete distribution is then used to resample
 * `count` points according to these weights.
 *
 * The higher the `oversampling_ratio`, the better the approximation to the
 * target density, at the cost of performance.
 *
 * @tparam T             Scalar type (e.g., float or double)
 * @tparam N             Dimensionality of the space
 * @tparam DensityFn     A callable with signature `T(const Point<T, N>&)`
 * returning a non-negative density value
 * @param  count              Number of points to return after resampling
 * @param  oversampling_ratio Number of candidate points to generate as a
 *                            multiple of `count`
 * @param  axis_ranges        Axis-aligned bounding box defining the domain of
 *                            the points
 * @param  density_fn         Function mapping a point to a (non-negative)
 *                            density value
 * @param  seed               Optional seed to control the random number
 *                            generator
 * @return                    std::vector<Point<T, N>> The resulting resampled
 *                            point set
 *
 * @par Example
 * @code
 * auto density = [](const Point<float, 2> &p) {
 *     return std::exp(-10.0f * (p[0]*p[0] + p[1]*p[1]));
 * };
 *
 * auto points = ps::importance_resampling<float, 2>(
 *     500, 5,
 *     {{{-1, 1}, {-1, 1}}}, density, 42
 * );
 * @endcode
 *
 * @image html out_importance_resampling.csv.jpg
 */
template <typename T, size_t N, typename DensityFn>
std::vector<Point<T, N>> importance_resampling(
    size_t                                count,
    size_t                                oversampling_ratio,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    DensityFn                             density_fn,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  // Generate grid
  size_t                   count_grid = count * oversampling_ratio;
  std::vector<Point<T, N>> grid_points = halton<T, N>(count_grid, axis_ranges, seed);

  // Weights
  std::vector<T> weights;
  weights.reserve(count_grid);

  for (auto &p : grid_points)
    weights.push_back(density_fn(p));

  // Normalize weights
  T sum = std::accumulate(weights.begin(), weights.end(), T(0));
  for (auto &w : weights)
    w /= sum;

  // Resample
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  std::vector<Point<T, N>>           samples;
  samples.reserve(count);
  for (size_t i = 0; i < count; ++i)
    samples.push_back(grid_points[dist(gen)]);

  return samples;
}

} // namespace ps
