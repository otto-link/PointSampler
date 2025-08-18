/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>
#include <random>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Generates a specified number of uniformly distributed random points in
 * N-dimensional space.
 *
 * This function creates `count` random points where each coordinate is
 * independently sampled from a uniform distribution defined by `axis_ranges`
 * per dimension.
 *
 * @tparam T The numeric type for coordinates (e.g., float, double).
 * @tparam N The dimensionality of the points.
 *
 * @param  count       The number of random points to generate.
 * @param  axis_ranges An array of N pairs specifying the min and max range for
 *                     each axis.
 * @param  seed        Optional seed for the random number generator. If not
 *                     provided, a nondeterministic random seed is used.
 *
 * @return             A vector containing `count` randomly generated points
 *                     within the specified axis ranges.
 *
 * @throws std::invalid_argumentIfanyaxisrangehasmin>max.
 *
 * @note The points are generated independently per axis using
 * uniform_real_distribution.
 *
 * @code{.cpp}
 * #include <iostream>
 * #include "point_sampler/random.hpp"
 *
 * int main()
 * {
 *   constexpr size_t dim = 3;
 *   size_t count = 5;
 *   std::array<std::pair<float, float>, dim> ranges = {{{0.f, 1.f},
 *                                                       {0.f, 2.f},
 *                                                       {-1.f, 1.f}}};
 *
 *   // Generate points with a fixed seed for reproducibility
 *   auto points = ps::random<float, dim>(count, ranges, 42);
 *
 *   for (const auto& p : points)
 *   {
 *     for (size_t i = 0; i < dim; ++i) std::cout << p[i] << ' ';
 *     std::cout << '\n';
 *   }
 *
 *   return 0;
 * }
 * @endcode
 *
 * @image html out_random.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> random(size_t                                count,
                                const std::array<std::pair<T, T>, N> &axis_ranges,
                                std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  // Create N distributions, one per dimension
  std::array<std::uniform_real_distribution<T>, N> dists;
  for (size_t i = 0; i < N; ++i)
  {
    const auto &[min_val, max_val] = axis_ranges[i];
    if (min_val > max_val)
      throw std::invalid_argument("Invalid axis range: min > max");
    dists[i] = std::uniform_real_distribution<T>(min_val, max_val);
  }

  std::vector<Point<T, N>> points;
  points.reserve(count);

  for (size_t i = 0; i < count; ++i)
  {
    Point<T, N> p;
    for (size_t j = 0; j < N; ++j)
      p[j] = dists[j](gen);
    points.push_back(p);
  }

  return points;
}

} // namespace ps
