/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <queue>

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
 *
 * Groups points into clusters based on density: a point is a core if it has at
 * least `min_pts` neighbors within distance `eps`. Clusters are formed by
 * expanding from core points. Noise points remain unclustered.
 *
 * @tparam T       Scalar type (float/double).
 * @tparam N       Dimension.
 * @param  points  Input point cloud.
 * @param  eps     Neighborhood radius for density check.
 * @param  min_pts Minimum neighbors (including self) to be a core point.
 * @return         A vector of cluster labels (-1 = noise, 0..k = cluster IDs).
 *
 * @par Example
 * @code {.cpp}
 * auto labels = dbscan<double,2>(points, 0.05, 5);
 * @endcode
 *
 * This assigns each 2D point either to a cluster ID or -1 (noise).
 *
 * @image html metrics_dbscan_clustering.jpg
 */
template <typename T, size_t N>
std::vector<int> dbscan_clustering(const std::vector<Point<T, N>> &points,
                                   T                               eps,
                                   size_t                          min_pts)
{
  if (points.empty())
    return {};

  PointCloudAdaptor<T, N> adaptor(points);
  KDTree<T, N>            index(N, adaptor);
  index.buildIndex();

  std::vector<int> labels(points.size(), -1); // -1 = unvisited/noise
  int              cluster_id = 0;

  for (size_t i = 0; i < points.size(); ++i)
  {
    if (labels[i] != -1)
      continue; // already assigned

    // Query neighbors within radius
    std::vector<nanoflann::ResultItem<unsigned int, T>> ret_matches;
    nanoflann::SearchParameters                         params;
    const T                                             eps_sq = eps * eps;
    std::array<T, N>                                    query;
    for (size_t d = 0; d < N; ++d)
      query[d] = points[i][d];

    index.radiusSearch(query.data(), eps_sq, ret_matches, params);

    if (ret_matches.size() < min_pts)
    {
      labels[i] = -2; // mark as noise
      continue;
    }

    // Start new cluster
    labels[i] = cluster_id;

    std::vector<size_t> seed_set;
    seed_set.reserve(ret_matches.size());
    for (auto &m : ret_matches)
      if (m.first != i)
        seed_set.push_back(m.first);

    for (size_t j = 0; j < seed_set.size(); ++j)
    {
      size_t neighbor_idx = seed_set[j];
      if (labels[neighbor_idx] == -2)
        labels[neighbor_idx] = cluster_id; // was
                                           // noise,
                                           // now
                                           // border
      if (labels[neighbor_idx] != -1)
        continue; // already
                  // assigned

      labels[neighbor_idx] = cluster_id;

      // Expand cluster if neighbor is a core
      std::array<T, N> nq;
      for (size_t d = 0; d < N; ++d)
        nq[d] = points[neighbor_idx][d];
      std::vector<nanoflann::ResultItem<unsigned int, T>> n_matches;
      index.radiusSearch(nq.data(), eps_sq, n_matches, params);

      if (n_matches.size() >= min_pts)
      {
        for (auto &nm : n_matches)
          if (labels[nm.first] == -1)
            seed_set.push_back(nm.first);
      }
    }

    cluster_id++;
  }

  return labels;
}

} // namespace ps