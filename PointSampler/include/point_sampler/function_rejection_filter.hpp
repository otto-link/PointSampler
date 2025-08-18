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
 * @brief Filters points based on a spatial probability (density) function.
 *
 * Each point is accepted with a probability given by `density_fn(p)`, which
 * should return a value in [0, 1]. This is useful for sampling from a
 * non-uniform spatial distribution.
 *
 * @tparam T         Scalar type (e.g., float or double)
 * @tparam N         Dimensionality of the space
 * @tparam DensityFn Callable type with signature `T(const Point<T, N>&)`
 * @param  points     Input candidate points
 * @param  density_fn Function returning acceptance probability for each point
 * @param  seed       Optional random seed for reproducibility
 * @return            std::vector<Point<T, N>> of accepted points
 *
 * @code
 * auto field = [](const Point<float, 2>& p) {
 *     return 0.5f + 0.5f * std::sin(p[0] * 10.0f); // Spatial density
 * };
 * auto accepted = ps::function_rejection_filter(pts, field);
 * @endcode
 */
template <typename T, size_t N, typename DensityFn>
std::vector<Point<T, N>> function_rejection_filter(
    const std::vector<Point<T, N>> &points,
    DensityFn                       density_fn,
    std::optional<unsigned int>     seed = std::nullopt)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> accept_dist(0.0, 1.0);

  std::vector<Point<T, N>> result;
  result.reserve(points.size());

  for (const auto &p : points)
  {
    T prob = density_fn(p);
    T threshold = accept_dist(gen);

    if (prob >= threshold)
      result.push_back(p);
  }

  return result;
}

} // namespace ps