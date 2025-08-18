/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Generates a point set on a jittered and optionally staggered grid.
 *
 * This function divides the domain into a grid and places one point in each
 * selected cell. Each point is jittered within its cell, and staggered offsets
 * may be applied depending on the index of higher-dimensional axes. The result
 * is a semi-regular sampling pattern with randomness.
 *
 * Jittering prevents aliasing, and staggering introduces a controlled shift
 * between alternating cells to improve uniformity and avoid alignment
 * artifacts.
 *
 * @tparam T            Scalar type (e.g., float or double)
 * @tparam N            Dimensionality of the space
 * @param  count         Number of output points (best effort, may be capped by
 *                       total available cells)
 * @param  axis_ranges   Axis-aligned bounding box defining the sampling domain
 * @param  jitter_amount Per-dimension jitter factor ∈ [0, 1]. A value of 1.0
 *                       means full jitter in the cell.
 * @param  stagger_ratio Per-dimension stagger ratio, indicating how much to
 *                       offset points based on higher dimension parity
 * @param  seed          Optional seed for deterministic jittering and shuffling
 * @return               std::vector<Point<T, N>> Sampled points
 *
 * @par Example
 * @code
 * std::array<std::pair<float, float>, 2> bounds = {{{0.0f, 1.0f}, {0.0f, 1.0f}}};
 * std::array<float, 2> jitter = {0.8f, 0.8f};
 * std::array<float, 2> stagger = {0.2f, 0.0f};
 *
 * auto samples = ps::jittered_grid<float, 2>(256, bounds, jitter, stagger, 42);
 * @endcode
 *
 * @image html out_jittered_grid.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> jittered_grid(size_t                                count,
                                       const std::array<std::pair<T, T>, N> &axis_ranges,
                                       const std::array<T, N>     &jitter_amount,
                                       const std::array<T, N>     &stagger_ratio,
                                       std::optional<unsigned int> seed = std::nullopt)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uniform01(0.0, 1.0);

  std::array<size_t, N> resolution;
  size_t                total_cells = 1;

  T volume = 1;
  for (const auto &[min, max] : axis_ranges)
    volume *= (max - min);

  T target_cell_volume = volume / static_cast<T>(count);
  T cell_size_estimate = std::pow(target_cell_volume, static_cast<T>(1.0) / N);

  for (size_t i = 0; i < N; ++i)
  {
    T range = axis_ranges[i].second - axis_ranges[i].first;
    resolution[i] = std::max<size_t>(1, static_cast<size_t>(range / cell_size_estimate));
    total_cells *= resolution[i];
  }

  std::vector<Point<T, N>> points;
  points.reserve(std::min(count, total_cells));

  std::vector<std::array<size_t, N>> cell_indices;
  for (size_t linear = 0; linear < total_cells; ++linear)
  {
    std::array<size_t, N> index{};
    size_t                div = 1;
    for (size_t i = 0; i < N; ++i)
    {
      index[i] = (linear / div) % resolution[i];
      div *= resolution[i];
    }
    cell_indices.push_back(index);
  }

  std::shuffle(cell_indices.begin(), cell_indices.end(), gen);
  size_t limit = std::min(count, cell_indices.size());

  for (size_t i = 0; i < limit; ++i)
  {
    const auto &idx = cell_indices[i];
    Point<T, N> p;

    for (size_t d = 0; d < N; ++d)
    {
      T range_min = axis_ranges[d].first;
      T range_max = axis_ranges[d].second;
      T cell_size = (range_max - range_min) / static_cast<T>(resolution[d]);

      T jitter_range = jitter_amount[d] * cell_size;
      T jitter_center = (1.0 - jitter_amount[d]) * 0.5 * cell_size;
      T jitter = uniform01(gen) * jitter_range;

      // Compute stagger offset from higher dimensions
      T stagger_offset = 0;
      for (size_t k = d + 1; k < N; ++k)
      {
        if (idx[k] % 2 == 1)
          stagger_offset += stagger_ratio[d] * cell_size;
      }

      p[d] = range_min + idx[d] * cell_size + jitter_center + jitter + stagger_offset;
    }

    points.push_back(p);
  }

  return points;
}

/**
 * @brief Generates a jittered grid of points with full jitter and no stagger.
 *
 * This overload defaults to jittering each dimension fully within its cell and
 * applies no staggering. It is equivalent to calling the full version with
 * `jitter_amount` filled with 1.0 and `stagger_ratio` filled with 0.0.
 *
 * @tparam T          Scalar type (e.g., float or double)
 * @tparam N          Dimensionality of the space
 * @param  count       Number of points to generate
 * @param  axis_ranges Axis-aligned bounding box defining the sampling domain
 * @param  seed        Optional seed for deterministic jittering
 * @return             std::vector<Point<T, N>> Jittered point samples
 *
 * @par Example
 * @code
 * std::array<std::pair<double, double>, 3> bounds = {{{0, 1}, {0, 1}, {0, 1}}};
 * auto points = ps::jittered_grid<double, 3>(1000, bounds, 1234);
 * @endcode
 */
template <typename T, size_t N>
std::vector<Point<T, N>> jittered_grid(size_t                                count,
                                       const std::array<std::pair<T, T>, N> &axis_ranges,
                                       std::optional<unsigned int> seed = std::nullopt)
{
  std::array<T, N> full_jitter;
  std::array<T, N> stagger_ratio;
  full_jitter.fill(static_cast<T>(1.0));
  stagger_ratio.fill(static_cast<T>(0.0));

  return jittered_grid<T, N>(count, axis_ranges, full_jitter, stagger_ratio, seed);
}

} // namespace ps
