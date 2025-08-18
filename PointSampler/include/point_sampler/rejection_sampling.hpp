/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/function_rejection_filter.hpp"

namespace ps
{

/**
 * @brief Generates random points using rejection sampling based on a
 * user-defined density function.
 *
 * This function uniformly samples candidate points within the given
 * axis-aligned bounds and retains them based on the output of a user-provided
 * density function. The
 * `density_fn` should return a probability in the range [0, 1] for each point.
 *
 * @tparam T          Numeric type (e.g., float or double).
 * @tparam N          Number of dimensions.
 * @tparam DensityFn  Callable with signature T(Point<T, N>) returning a
 * probability in [0, 1].
 *
 * @param  count       Desired number of accepted points.
 * @param  axis_ranges Ranges for each axis in the form of an array of (min,
 *                     max) pairs.
 * @param  density_fn  Function that returns a probability for accepting a
 *                     point.
 * @param  seed        Optional seed for reproducibility.
 *
 * @return             std::vector<Point<T, N>> A vector of accepted points
 *                     based on rejection sampling.
 *
 * @throws std::invalid_argumentifanyaxisrangeisinvalid(min>max).
 *
 * @par Example
 * @code
 * #include "point_sampler/rejection_sampling.hpp"
 * #include <iostream>
 *
 * float radial_density(const Point<float, 2>& p)
 * {
 *     float r2 = p[0] * p[0] + p[1] * p[1];
 *     return std::exp(-r2); // higher near origin, drops with radius
 * }
 *
 * int main()
 * {
 *     std::array<std::pair<float, float>, 2> bounds = { { {-2.0f, 2.0f},
 *                                                         {-2.0f, 2.0f} }};
 *     auto pts = ps::rejection_sampling<float, 2>(1000, bounds, radial_density, 42);
 *     std::cout << "Generated " << pts.size() << " points.\n";
 * }
 * @endcode
 *
 * @note Rejection sampling can be inefficient if `density_fn` returns low
 * values over most of the domain, as many candidate samples will be discarded.
 *
 * @image html out_rejection_sampling.csv.jpg
 */
template <typename T, size_t N, typename DensityFn>
std::vector<Point<T, N>> rejection_sampling(
    size_t                                count,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    DensityFn                             density_fn,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  std::array<std::uniform_real_distribution<T>, N> coord_dists;
  for (size_t i = 0; i < N; ++i)
    coord_dists[i] = std::uniform_real_distribution<T>(axis_ranges[i].first,
                                                       axis_ranges[i].second);

  std::vector<Point<T, N>> candidates;
  candidates.reserve(count * 2); // overgenerate

  while (candidates.size() < count * 2)
  {
    Point<T, N> p;
    for (size_t i = 0; i < N; ++i)
      p[i] = coord_dists[i](gen);

    candidates.push_back(p);
  }

  return function_rejection_filter<T, N>(candidates, density_fn, seed);
}

} // namespace ps
