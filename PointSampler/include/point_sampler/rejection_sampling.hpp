/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N, typename DensityFn>
std::vector<Point<T, N>> rejection_sampling(
    std::size_t                           count,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    DensityFn                             density_fn,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  std::array<std::uniform_real_distribution<T>, N> coord_dists;
  for (std::size_t i = 0; i < N; ++i)
    coord_dists[i] = std::uniform_real_distribution<T>(axis_ranges[i].first,
                                                       axis_ranges[i].second);

  std::uniform_real_distribution<T> accept_dist(0.0, 1.0);

  std::vector<Point<T, N>> result;
  result.reserve(count);

  while (result.size() < count)
  {
    Point<T, N> p;
    for (std::size_t i = 0; i < N; ++i)
      p[i] = coord_dists[i](gen);

    T prob = density_fn(p);
    T threshold = accept_dist(gen);

    if (prob >= threshold)
      result.push_back(p);
  }

  return result;
}

} // namespace ps