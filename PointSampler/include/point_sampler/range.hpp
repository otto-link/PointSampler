/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Filters points that lie within the specified axis-aligned bounding
 * box.
 *
 * @tparam T           Numeric type for coordinates (e.g., float or double).
 * @tparam N           Number of dimensions.
 *
 * @param  points      Vector of input points to filter.
 * @param  axis_ranges Axis-aligned bounding box ranges for each dimension
 *                     (inclusive).
 *
 * @return             A vector containing only the points that lie within all
 *                     specified axis ranges.
 *
 * @throws std::invalid_argumentifaxis_rangesareill-formed(e.g., min > max).
 *
 * @code
 * std::vector<Point<float, 2>> pts = { {0.5f, 0.5f}, {2.f, 3.f}, {-1.f, 0.f} };
 * auto filtered = ps::filter_points_in_range<float, 2>(pts, {{{0.f, 1.f},
 *                                                            {0.f, 1.f}}});
 * // filtered now contains only { {0.5f, 0.5f} }
 * @endcode
 */
template <typename T, size_t N>
std::vector<Point<T, N>> filter_points_in_range(
    const std::vector<Point<T, N>>       &points,
    const std::array<std::pair<T, T>, N> &axis_ranges)
{
  std::vector<Point<T, N>> filtered;
  filtered.reserve(points.size());

  for (const auto &p : points)
  {
    bool inside = true;
    for (size_t i = 0; i < N; ++i)
    {
      const auto &[min_val, max_val] = axis_ranges[i];
      if (p[i] < min_val || p[i] > max_val)
      {
        inside = false;
        break;
      }
    }
    if (inside)
      filtered.push_back(p);
  }

  return filtered;
}

/**
 * @brief Filters points using a user-provided function.
 *
 * Keeps only the points for which the provided function `fn(p)` does not return
 * zero. This can be used to apply custom masks, implicit surface functions,
 * etc.
 *
 * @tparam T   Numeric type for coordinates.
 * @tparam N   Number of dimensions.
 * @tparam Func Callable that takes a Point<T, N> and returns a value
 * convertible to T.
 *
 * @param  points Vector of input points.
 * @param  fn     Unary function that returns a non-zero value if the point
 *                should be kept.
 *
 * @return        A vector of filtered points.
 *
 * @code
 * std::vector<Point<float, 2>> pts = { {0.f, 0.f}, {1.f, 1.f}, {2.f, 2.f}};
 * auto lambda = [](const Point<float, 2> &p) {
 *     return (p[0] + p[1] < 2.5f) ? 1.f : 0.f;
 * };
 * auto filtered = ps::filter_points_function<float, 2>(pts, lambda);
 * // Keeps only { {0.f, 0.f}, {1.f, 1.f} }
 * @endcode
 */
template <typename T, std::size_t N, typename Func>
std::vector<Point<T, N>> filter_points_function(const std::vector<Point<T, N>> &points,
                                                Func                            fn)
{
  std::vector<Point<T, N>> filtered;
  filtered.reserve(points.size());

  for (const auto &p : points)
  {
    if (fn(p) != T(0)) // keep point if function value is not zero
      filtered.push_back(p);
  }

  return filtered;
}

/**
 * @brief Linearly remap a set of points to fit within the specified
 * axis-aligned ranges.
 *
 * This function computes the axis-aligned bounding box (AABB) of the input
 * points and linearly rescales each point so that all dimensions lie in the
 * given
 * `target_ranges`.
 *
 * @tparam T           Numeric type (e.g., float or double).
 * @tparam N           Number of dimensions.
 *
 * @param[in,out] points        Vector of input points to modify in-place.
 * @param[in]     target_ranges Desired output min/max per dimension.
 *
 * @par Example
 * @code
 * std::vector<Point<float, 2>> pts = generate_random_points<float, 2>(100,
 *      { {{0.f, 1.f}, {0.f, 1.f} } }, 42);
 * // Refit to a new range: [10, 20] × [50, 100]
 * refit_points_to_range<float, 2>(pts, { { {10.f, 20.f}, {50.f, 100.f} } });
 * @endcode
 *
 * @note If a dimension has constant value (min == max), the center of the
 * target range is used.
 */
template <typename T, size_t N>
void refit_points_to_range(std::vector<Point<T, N>>             &points,
                           const std::array<std::pair<T, T>, N> &target_ranges)
{
  if (points.empty())
    return;

  std::array<T, N> min_vals, max_vals;

  // Initialize min/max
  for (size_t d = 0; d < N; ++d)
  {
    min_vals[d] = points[0][d];
    max_vals[d] = points[0][d];
  }

  // Compute bounding box
  for (const auto &p : points)
  {
    for (size_t d = 0; d < N; ++d)
    {
      min_vals[d] = std::min(min_vals[d], p[d]);
      max_vals[d] = std::max(max_vals[d], p[d]);
    }
  }

  // Apply linear mapping to each point
  for (auto &p : points)
  {
    for (size_t d = 0; d < N; ++d)
    {
      const T in_min = min_vals[d];
      const T in_max = max_vals[d];
      const T out_min = target_ranges[d].first;
      const T out_max = target_ranges[d].second;

      if (std::abs(in_max - in_min) < T(1e-12))
      {
        // Degenerate axis: center the target range
        p[d] = (out_min + out_max) / T(2);
      }
      else
      {
        const T t = (p[d] - in_min) / (in_max - in_min);
        p[d] = out_min + t * (out_max - out_min);
      }
    }
  }
}

/**
 * @brief Rescales normalized points (in [0, 1]) to specified axis-aligned
 * ranges.
 *
 * Each coordinate in every point is mapped from [0, 1] to a new range defined
 * per axis. This is useful after generating normalized samples (e.g., Poisson
 * disk, jittered grid).
 *
 * @tparam T            Numeric type (e.g., float, double).
 * @tparam N            Number of dimensions.
 *
 * @param[in,out] points Vector of normalized points to be modified in-place.
 * @param[in]     ranges Target value ranges for each dimension.
 *
 * @code
 * std::vector<Point<float, 2>> pts = { {0.f, 0.f}, {1.f, 1.f}, {0.5f, 0.5f} };
 * ps::rescale_points<float, 2>(pts, { { {10.f, 20.f}, {100.f, 200.f} } });
 * // pts is now { {10.f, 100.f}, {20.f, 200.f}, {15.f, 150.f} }
 * @endcode
 *
 * @note Assumes points are in [0, 1]^N. Does not check bounds.
 */
template <typename T, size_t N>
void rescale_points(std::vector<Point<T, N>>             &points,
                    const std::array<std::pair<T, T>, N> &ranges)
{
  for (auto &pt : points)
    for (size_t d = 0; d < N; ++d)
      pt[d] = ranges[d].first + pt[d] * (ranges[d].second - ranges[d].first);
}

} // namespace ps
