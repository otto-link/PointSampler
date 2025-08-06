/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Filters points that lie within the given axis-aligned ranges.
 *
 * @tparam T Type of coordinates.
 * @tparam N Number of dimensions.
 * @param points        Input points.
 * @param axis_ranges   Ranges for each dimension (min, max) inclusive.
 * @return std::vector<Point<T, N>> Filtered points that lie within the given ranges.
 */
template <typename T, std::size_t N>
std::vector<Point<T, N>> filter_points_in_range(
    const std::vector<Point<T, N>>       &points,
    const std::array<std::pair<T, T>, N> &axis_ranges)
{
  std::vector<Point<T, N>> filtered;
  filtered.reserve(points.size());

  for (const auto &p : points)
  {
    bool inside = true;
    for (std::size_t i = 0; i < N; ++i)
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
 * @brief Linearly remap a set of points to fit within the specified axis-aligned ranges.
 *
 * This function computes the axis-aligned bounding box (AABB) of the input points
 * and linearly rescales each point so that all dimensions lie in the given
 * `target_ranges`.
 *
 * @tparam T           Numeric type (e.g., float or double).
 * @tparam N           Number of dimensions.
 *
 * @param[in,out] points        Vector of input points to modify in-place.
 * @param[in]      target_ranges Desired output min/max per dimension.
 *
 * ### Example
 * @code
 * std::vector<Point<float, 2>> pts = generate_random_points<float, 2>(100, { {
 * {0.f, 1.f}, {0.f, 1.f} } }, 42);
 * // Refit to a new range: [10, 20] × [50, 100]
 * refit_points_to_range<float, 2>(pts, { { {10.f, 20.f}, {50.f, 100.f} } });
 * @endcode
 *
 * @note If a dimension has constant value (min == max), the center of the target range is
 * used.
 */
template <typename T, std::size_t N>
void refit_points_to_range(std::vector<Point<T, N>>             &points,
                           const std::array<std::pair<T, T>, N> &target_ranges)
{
  if (points.empty())
    return;

  std::array<T, N> min_vals, max_vals;

  // Initialize min/max
  for (std::size_t d = 0; d < N; ++d)
  {
    min_vals[d] = points[0][d];
    max_vals[d] = points[0][d];
  }

  // Compute bounding box
  for (const auto &p : points)
  {
    for (std::size_t d = 0; d < N; ++d)
    {
      min_vals[d] = std::min(min_vals[d], p[d]);
      max_vals[d] = std::max(max_vals[d], p[d]);
    }
  }

  // Apply linear mapping to each point
  for (auto &p : points)
  {
    for (std::size_t d = 0; d < N; ++d)
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

} // namespace ps