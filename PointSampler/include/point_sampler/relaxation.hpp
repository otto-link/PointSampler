/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <cstddef>
#include <optional>

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N>
void relaxation_ktree(std::vector<Point<T, N>> &points,
                      std::size_t               k_neighbors = 8,
                      T                         step_size = T(0.1),
                      std::size_t               iterations = 10)
{
  for (std::size_t iter = 0; iter < iterations; ++iter)
  {
    PointCloudAdaptor<T, N> adaptor(points);
    KDTree<T, N>            index(N, adaptor);
    index.buildIndex();

    std::vector<Point<T, N>> new_points = points;

    for (std::size_t i = 0; i < points.size(); ++i)
    {
      const auto &p = points[i];

      std::vector<size_t> ret_indexes(k_neighbors + 1);
      std::vector<T>      out_dists_sqr(k_neighbors + 1);

      nanoflann::KNNResultSet<T> result_set(k_neighbors + 1);
      result_set.init(ret_indexes.data(), out_dists_sqr.data());

      std::array<T, N> query;
      for (std::size_t d = 0; d < N; ++d)
        query[d] = p[d];

      index.findNeighbors(result_set, query.data(), nanoflann::SearchParameters());

      Point<T, N> offset{};
      for (std::size_t j = 1; j < result_set.size(); ++j) // skip self at j=0
      {
        const auto &q = points[ret_indexes[j]];
        auto        delta = p - q;
        T           dist_sq = length_squared(delta) + T(1e-6); // avoid div by 0

        offset = offset + delta / dist_sq;
      }

      new_points[i] = p + normalized(offset) * step_size;
    }

    points = std::move(new_points);
  }
}

} // namespace ps