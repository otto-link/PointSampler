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
 * @brief Relax a point set using a k-nearest neighbor repulsion algorithm with
 * a KD-tree.
 *
 * This function performs iterative relaxation on a set of N-dimensional points
 * by pushing each point away from its nearest neighbors. It uses a KD-tree for
 * efficient neighbor lookup. The goal is to reduce clustering and obtain a more
 * uniform or blue-noise-like distribution.
 *
 * Each point is offset based on inverse-distance-weighted repulsion from its
 * k-nearest neighbors, normalized and scaled by a step size. The point set is
 * updated over a number of iterations.
 *
 * @tparam T           Numeric type for coordinates (e.g., float or double).
 * @tparam N           Number of dimensions.
 *
 * @param[in,out] points      The point set to relax. This vector will be
 *                            modified in place.
 * @param[in]     k_neighbors Number of neighbors to consider (default is 8).
 * @param[in]     step_size   How far to move a point per iteration (default is
 *                            0.1).
 * @param[in]     iterations  Number of relaxation iterations (default is 10).
 *
 * @note The KD-tree is rebuilt on each iteration to reflect the updated
 * positions.
 *
 * @par Example
 * @code
 * #include <point_sampler/relaxation.hpp>
 *
 * std::vector<Point<float, 2>> pts = generate_random_points<float, 2>(
 *     1000, { { {0.f, 1.f}, {0.f, 1.f} } }, 42);
 *
 * // Apply 10 iterations of relaxation relaxation_ktree<float, 2>(pts, 8, 0.1f, 10);
 * @endcode
 *
 * ### How it works:
 * - For each point:
 *   - Find its `k_neighbors` nearest neighbors using a KD-tree.
 *   - Compute offset vectors from the current point to each neighbor.
 *   - Weight the vectors by the inverse square distance (stronger push from
 * closer neighbors).
 *   - Accumulate the offset, normalize, and scale by `step_size`.
 *   - Apply the movement to each point.
 * - Repeat for `iterations` steps.
 *
 * @image html out_relaxation_ktree_refit.csv.jpg
 */
template <typename T, size_t N>
void relaxation_ktree(std::vector<Point<T, N>> &points,
                      size_t                    k_neighbors = 8,
                      T                         step_size = T(0.1),
                      size_t                    iterations = 10)
{
  for (size_t iter = 0; iter < iterations; ++iter)
  {
    PointCloudAdaptor<T, N> adaptor(points);
    KDTree<T, N>            index(N, adaptor);
    index.buildIndex();

    std::vector<Point<T, N>> new_points = points;

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

      Point<T, N> offset{};
      // skip self at j=0
      for (size_t j = 1; j < result_set.size(); ++j)
      {
        const auto &q = points[ret_indexes[j]];
        auto        delta = p - q;
        // avoid div by 0
        T dist_sq = length_squared(delta) + T(1e-6);

        offset = offset + delta / dist_sq;
      }

      new_points[i] = p + normalized(offset) * step_size;
    }

    points = std::move(new_points);
  }
}

} // namespace ps