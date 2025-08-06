/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"
#include "point_sampler/range.hpp"

namespace ps
{

template <typename T, size_t N>
std::vector<Point<T, N>> hammersley_sequence(size_t count, size_t shift)
{
  auto van_der_corput = [](size_t n, size_t base) -> T
  {
    T q = 0;
    T bk = 1.0 / base;
    while (n > 0)
    {
      q += T(n % base) * bk;
      n /= base;
      bk /= base;
    }
    return q;
  };

  constexpr size_t primes[15] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  constexpr size_t n_primes = 15;

  std::vector<Point<T, N>> points(count);
  for (size_t i = 0; i < count; ++i)
  {
    points[i][0] = static_cast<T>(i) / static_cast<T>(count);
    for (size_t d = 1; d < N; ++d)
      points[i][d] = van_der_corput(i + shift, primes[std::min(n_primes - 1, d - 1)]);
  }
  return points;
}

template <typename T, size_t N>
std::vector<Point<T, N>> hammersley(size_t                                count,
                                    const std::array<std::pair<T, T>, N> &axis_ranges,
                                    std::optional<unsigned int> seed = std::nullopt)
{
  // seed-like effect...
  size_t shift = seed ? *seed : 0;
  auto   points = hammersley_sequence<T, N>(count, shift);
  rescale_points(points, axis_ranges);
  return points;
}

} // namespace ps