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
 * @brief Analyze percolation clusters from a set of points using a radius-based
 * neighbor graph.
 *
 * Builds a graph where edges exist if points are within `connection_radius`,
 * then finds connected components (clusters).
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension.
 * @param points Input set of points.
 * @param connection_radius Maximum distance for connectivity between points.
 * @return A vector of cluster labels, size = points.size(). Label = -1 if
 * unassigned.
 *
 * @par Example
 * @code
 * std::vector<Point<double,2>> pts = { {0.1,0.2}, {0.15,0.22}, {0.9,0.9}};
 * auto labels = analyze_percolation_clusters<double,2>(pts, 0.1);
 *
 * // labels might be {0,0,1}, meaning the first two form a cluster, third is separate.
 * @endcode
 *
 * @image html metrics_percolation_clustering.jpg
 */
template <typename T, size_t N>
std::vector<int> percolation_clustering(const std::vector<Point<T, N>> &points,
                                        T                               connection_radius)
{
  if (points.empty())
    return {};

  // KD-tree for neighbor search
  PointCloudAdaptor<T, N> adaptor(points);
  KDTree<T, N>            index(N, adaptor);
  index.buildIndex();

  std::vector<int> labels(points.size(), -1);
  int              current_cluster = 0;

  for (size_t i = 0; i < points.size(); ++i)
  {
    if (labels[i] != -1)
      continue; // already assigned

    // Start BFS/DFS for new cluster
    std::queue<size_t> q;
    q.push(i);
    labels[i] = current_cluster;

    while (!q.empty())
    {
      size_t p_idx = q.front();
      q.pop();

      // Radius search around p_idx
      const Point<T, N> &p = points[p_idx];
      std::array<T, N>   query;
      for (size_t d = 0; d < N; ++d)
        query[d] = p[d];

      std::vector<nanoflann::ResultItem<unsigned int, T>> ret_matches;
      nanoflann::SearchParameters                         params;
      index.radiusSearch(query.data(),
                         connection_radius * connection_radius,
                         ret_matches,
                         params);

      for (auto &m : ret_matches)
      {
        size_t neighbor = m.first;
        if (labels[neighbor] == -1)
        {
          labels[neighbor] = current_cluster;
          q.push(neighbor);
        }
      }
    }

    ++current_cluster;
  }

  return labels;
}

} // namespace ps