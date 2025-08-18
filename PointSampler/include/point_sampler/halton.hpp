/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"
#include "point_sampler/range.hpp"

namespace ps
{

template <typename T, size_t N>
std::vector<Point<T, N>> halton_sequence(size_t count, size_t shift)
{
  auto halton = [](size_t index, size_t base) -> T
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

  constexpr size_t primes[15] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
  constexpr size_t n_primes = 15;

  std::vector<Point<T, N>> points(count);
  for (size_t i = 0; i < count; ++i)
  {
    for (size_t d = 0; d < N; ++d)
      points[i][d] = halton(i + 1 + shift, primes[std::min(n_primes - 1, d)]);
  }
  return points;
}

/**
 * @brief Generates a set of quasi-random points using the Halton sequence in N
 * dimensions.
 *
 * This function generates `count` points in the unit hypercube using the Halton
 * sequence, then rescales them to fit within the specified axis-aligned
 * bounding box. An optional `seed` is used as a starting index offset (i.e., a
 * shift) in the sequence to decorrelate multiple calls.
 *
 * @tparam T           Scalar type (e.g., float or double)
 * @tparam N           Dimensionality of the space
 * @param  count       Number of points to generate
 * @param  axis_ranges Axis-aligned bounding box for each dimension, as min/max
 *                     pairs
 * @param  seed        Optional seed that offsets the sequence start index
 * @return             std::vector<Point<T, N>> The generated Halton points
 *                     rescaled to the bounding box
 *
 * @par Example
 * @code
 * auto points = ps::halton<float, 2>(
 *     1000,
 *     {{{0, 1}, {0, 1}}}, 42
 * );
 * @endcode
 *
 * @image html out_halton.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> halton(size_t                                count,
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
