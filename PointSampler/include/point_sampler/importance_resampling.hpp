/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>

#include "point_sampler/jittered_grid.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N, typename DensityFn>
std::vector<Point<T, N>> importance_resampling(
    std::size_t                           count,
    std::size_t                           oversampling_ratio,
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
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
  std::vector<Point<T, N>>                samples;
  samples.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    samples.push_back(grid_points[dist(gen)]);

  return samples;
}

} // namespace ps