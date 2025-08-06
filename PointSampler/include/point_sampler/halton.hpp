/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"
#include "point_sampler/range.hpp"

namespace ps
{

template <typename T, std::size_t N>
std::vector<Point<T, N>> halton_sequence(std::size_t count, size_t shift)
{
  auto halton = [](std::size_t index, std::size_t base) -> T
  {
    T result = 0;
    T f = 1;
    while (index > 0)
    {
      f = f / base;
      result += f * (index % base);
      index = index / base;
    }
    return result;
  };

  constexpr std::size_t primes[15] =
      {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  constexpr std::size_t n_primes = 15;

  std::vector<Point<T, N>> points(count);
  for (std::size_t i = 0; i < count; ++i)
  {
    for (std::size_t d = 0; d < N; ++d)
      points[i][d] = halton(i + 1 + shift, primes[std::min(n_primes - 1, d)]);
  }
  return points;
}

template <typename T, std::size_t N>
std::vector<Point<T, N>> halton(std::size_t                           count,
                                const std::array<std::pair<T, T>, N> &axis_ranges,
                                std::optional<unsigned int>           seed = std::nullopt)
{
  // seed-like effect...
  size_t shift = seed ? *seed : 0;
  auto   points = halton_sequence<T, N>(count, shift);
  rescale_points(points, axis_ranges);
  return points;
}

} // namespace ps