/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N>
std::vector<Point<T, N>> generate_random_points_jittered_grid(
    std::size_t                           count,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    const std::array<T, N>               &jitter_amount,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uniform01(0.0, 1.0);

  // Estimate grid resolution
  std::array<std::size_t, N> resolution;
  std::size_t                total_cells = 1;

  T volume = 1;
  for (const auto &[min, max] : axis_ranges)
    volume *= (max - min);

  T target_cell_volume = volume / static_cast<T>(count);
  T cell_size_estimate = std::pow(target_cell_volume, static_cast<T>(1.0) / N);

  for (std::size_t i = 0; i < N; ++i)
  {
    T range = axis_ranges[i].second - axis_ranges[i].first;
    resolution[i] = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(range / cell_size_estimate));
    total_cells *= resolution[i];
  }

  std::vector<Point<T, N>> points;
  points.reserve(std::min(count, total_cells));

  // Prepare shuffled grid indices
  std::vector<std::array<std::size_t, N>> cell_indices;
  for (std::size_t linear = 0; linear < total_cells; ++linear)
  {
    std::array<std::size_t, N> index{};
    std::size_t                div = 1;
    for (std::size_t i = 0; i < N; ++i)
    {
      index[i] = (linear / div) % resolution[i];
      div *= resolution[i];
    }
    cell_indices.push_back(index);
  }

  std::shuffle(cell_indices.begin(), cell_indices.end(), gen);
  std::size_t limit = std::min(count, cell_indices.size());

  for (std::size_t i = 0; i < limit; ++i)
  {
    const auto &idx = cell_indices[i];
    Point<T, N> p;

    for (std::size_t d = 0; d < N; ++d)
    {
      T range_min = axis_ranges[d].first;
      T range_max = axis_ranges[d].second;
      T cell_size = (range_max - range_min) / static_cast<T>(resolution[d]);

      T jitter_range = jitter_amount[d] * cell_size;
      T jitter_center = (1.0 - jitter_amount[d]) * 0.5 * cell_size;
      T jitter = uniform01(gen) * jitter_range;

      p[d] = range_min + idx[d] * cell_size + jitter_center + jitter;
    }

    points.push_back(p);
  }

  return points;
}

// overload for full-jitter
template <typename T, std::size_t N>
std::vector<Point<T, N>> generate_random_points_jittered_grid(
    std::size_t                           count,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::array<T, N> full_jitter;
  full_jitter.fill(static_cast<T>(1.0));
  return generate_random_points_jittered_grid<T, N>(count,
                                                    axis_ranges,
                                                    full_jitter,
                                                    seed);
}

} // namespace ps