/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <cstddef>
#include <optional>

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Computes the squared distance to the nearest neighbor for each point.
 *
 * This function builds a KD-tree from a given set of N-dimensional points and
 * computes, for each point, the squared distance to its closest neighbor.
 *
 * The KD-tree search uses `nanoflann` for efficient nearest neighbor queries.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of each point.
 *
 * @param  points Vector of N-dimensional points. Note:** This vector is passed
 *                by non-const reference because the KD-tree adaptor may require
 *                mutable access, but the function does not modify the contents.
 *
 * @return        A vector containing the squared distances to the first
 *                neighbor for each point, in the same order as the input
 *                points.
 *
 * @code
 * // Example usage:
 * #include <array>
 * #include <vector>
 *
 * using Point3f = Point<float, 3>;
 *
 * std::vector<Point3f> points = {
 *     {0.0f, 0.0f, 0.0f},
 *     {1.0f, 0.0f, 0.0f},
 *     {0.0f, 1.0f, 0.0f},
 *     {1.0f, 1.0f, 0.0f}
 * };
 *
 * std::vector<float> distances_sq = first_neighbor_distance(points);
 *
 * // distances_sq[i] contains the squared distance to the closest neighbor of
 * points[i].
 * @endcode
 */
template <typename T, size_t N>
std::vector<T> first_neighbor_distance(std::vector<Point<T, N>> &points)
{
  PointCloudAdaptor<T, N> adaptor(points);
  KDTree<T, N>            index(N, adaptor);
  index.buildIndex();

  std::vector<T> distance_sq;
  distance_sq.reserve(points.size());

  // first neighbor search only
  const size_t k_neighbors = 1;

  for (size_t i = 0; i < points.size(); ++i)
  {
    const auto &p = points[i];

    std::vector<size_t> ret_indexes(k_neighbors + 1);
    std::vector<T>      out_dists_sqr(k_neighbors + 1);

    nanoflann::KNNResultSet<T> result_set(k_neighbors + 1);
    result_set.init(ret_indexes.data(), out_dists_sqr.data());

    std::array<T, N> query;
    for (size_t d = 0; d < N; ++d)
      query[d] = p[d];

    index.findNeighbors(result_set, query.data(), nanoflann::SearchParameters());

    T dist_sq = 0.f;
    if (ret_indexes.size() > 1)
    {
      const auto &q = points[ret_indexes[1]];
      auto        delta = p - q;
      dist_sq = length_squared(delta) + T(1e-6);
    }

    distance_sq.push_back(dist_sq);
  }

  return distance_sq;
}

} // namespace ps