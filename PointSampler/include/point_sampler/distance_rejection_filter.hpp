/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

#include <iostream>

namespace ps
{

/**
 * @brief Filters a set of points using a greedy distance-based rejection.
 *
 * Points are added one by one. If a point is at least `min_dist` away from all
 * previously accepted points, it is kept. Otherwise, it is rejected.
 *
 * @tparam T        Scalar type (e.g., float or double)
 * @tparam N        Dimensionality of the space
 * @param  points   Vector of candidate points
 * @param  min_dist Minimum allowed distance between two accepted points
 * @return          std::vector<Point<T, N>> of filtered points
 *
 * @par Example
 * @code {.cpp}
 * std::vector<Point<float, 2>> pts = ps::random<float, 2>(1000, {{0,1},{0,1}});
 * auto filtered = ps::distance_rejection_filter(pts, 0.05f);
 * @endcode
 *
 * @image html out_distance_rejection_filter.csv.jpg
 */
template <typename T, std::size_t N>
std::vector<Point<T, N>> distance_rejection_filter(const std::vector<Point<T, N>> &points,
                                                   T min_dist)
{
  if (points.empty())
    return {};

  std::vector<Point<T, N>> result;
  result.reserve(points.size());
  result.push_back(points.front());

  for (std::size_t i = 1; i < points.size(); ++i)
  {
    const auto &p = points[i];

    PointCloudAdaptor<T, N> adaptor(result);
    KDTree<T, N>            index(N, adaptor);
    index.buildIndex();

    std::array<T, N> query;
    for (std::size_t d = 0; d < N; ++d)
      query[d] = p[d];

    std::vector<nanoflann::ResultItem<unsigned int, T>> matches;
    nanoflann::SearchParameters                         params;
    const T                                             radius = min_dist * min_dist;

    const size_t found = index.radiusSearch(query.data(), radius, matches, params);

    if (found == 0)
      result.push_back(p);
  }

  return result;
}

/**
 * @brief Filters points based on spatially-varying minimal distance
 * constraints.
 *
 * A scale function is used to modulate the minimum allowed distance for each
 * point. The base distance `base_min_dist` is scaled by the value returned by
 * `scale_fn(p)`, allowing for adaptive sampling densities.
 *
 * @tparam T        Scalar type (e.g., float or double)
 * @tparam N        Dimensionality of the space
 * @tparam ScaleFn  Callable returning a scalar scale factor for a given point
 *
 * @param  points        Vector of candidate points
 * @param  base_min_dist Base minimum allowed distance
 * @param  scale_fn      Function providing a local scale factor per point
 * @return               std::vector<Point<T, N>> A filtered set of points with
 *                       variable spacing
 *
 * @par Example
 * @code
 * auto scale_fn = [](const Point<float, 2>& p) {
 *     return 0.5f + 0.5f * std::sin(p[0] * 3.1415f); // Varies between 0.5 and 1
 * };
 *
 * std::vector<Point<float, 2>> pts = ps::random<float, 2>(1000, {{0,1},{0,1}});
 * auto filtered = ps::distance_rejection_filter_warped(pts, 0.05f, scale_fn);
 * @endcode
 *
 * @image html out_distance_rejection_filter_warped.csv.jpg
 */
template <typename T, std::size_t N, typename ScaleFn>
std::vector<Point<T, N>> distance_rejection_filter_warped(
    const std::vector<Point<T, N>> &points,
    T                               base_min_dist,
    ScaleFn                         scale_fn)
{
  if (points.empty())
    return {};

  std::vector<Point<T, N>> result;
  result.reserve(points.size());
  result.push_back(points.front());

  for (std::size_t i = 1; i < points.size(); ++i)
  {
    const auto &p = points[i];
    const T     local_min_dist = base_min_dist * scale_fn(p);

    PointCloudAdaptor<T, N> adaptor(result);
    KDTree<T, N>            index(N, adaptor);
    index.buildIndex();

    std::array<T, N> query;
    for (std::size_t d = 0; d < N; ++d)
      query[d] = p[d];

    std::vector<nanoflann::ResultItem<unsigned int, T>> matches;
    nanoflann::SearchParameters                         params;
    const T radius = local_min_dist * local_min_dist;

    const size_t found = index.radiusSearch(query.data(), radius, matches, params);
    if (found == 0)
      result.push_back(p);
  }

  return result;
}

} // namespace ps
