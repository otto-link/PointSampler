/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>
#include <random>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N>
std::vector<Point<T, N>> random(std::size_t                           count,
                                const std::array<std::pair<T, T>, N> &axis_ranges,
                                std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  // Create N distributions, one per dimension
  std::array<std::uniform_real_distribution<T>, N> dists;
  for (std::size_t i = 0; i < N; ++i)
  {
    const auto &[min_val, max_val] = axis_ranges[i];
    if (min_val > max_val)
      throw std::invalid_argument("Invalid axis range: min > max");
    dists[i] = std::uniform_real_distribution<T>(min_val, max_val);
  }

  std::vector<Point<T, N>> points;
  points.reserve(count);

  for (std::size_t i = 0; i < count; ++i)
  {
    Point<T, N> p;
    for (std::size_t j = 0; j < N; ++j)
      p[j] = dists[j](gen);
    points.push_back(p);
  }

  return points;
}

} // namespace ps