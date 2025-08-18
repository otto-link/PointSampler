/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include "dkm.hpp" // https://github.com/genbattle/dkm

#include "point_sampler/internal/nanoflann_adaptator.hpp"
#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Perform k-means clustering on a set of points.
 *
 * Uses the `dkm` library to cluster points into `k` groups.
 *
 * @tparam T Floating point type.
 * @tparam N Dimensionality of the points.
 * @param  points         Vector of points to cluster.
 * @param  k_clusters     Number of clusters.
 * @param  max_iterations Maximum number of iterations for k-means.
 * @return                std::pair< std::vector<Point<T, N>>,
 *                        std::vector<size_t> >
 *         - First element: vector of cluster centroids.
 *         - Second element: cluster index assignment for each point.
 *
 * @par Example
 * @code
 * std::vector<Point<float, 2>> pts = {
 *     {0.1f, 0.2f}, {0.15f, 0.22f}, {0.8f, 0.75f}
 * };
 * auto [centroids, labels] = kmeans_clustering(pts, 2);
 * // centroids.size() == 2
 * // labels.size() == pts.size()
 * @endcode
 *
 * @image html metrics_kmeans_clustering.jpg
 */
template <typename T, size_t N>
std::pair<std::vector<Point<T, N>>, std::vector<size_t>> kmeans_clustering(
    const std::vector<Point<T, N>> &points,
    size_t                          k_clusters,
    bool                            normalize_data = true,
    size_t                          max_iterations = 100)
{
  using DKMPoint = std::array<T, N>;
  std::vector<DKMPoint> data;
  data.reserve(points.size());

  if (normalize_data)
  {
    auto points_normalized = points;
    normalize_points(points_normalized);

    for (const auto &p : points_normalized)
      data.push_back(p.coords);
  }
  else
  {
    for (const auto &p : points)
      data.push_back(p.coords);
  }

  auto result = dkm::kmeans_lloyd<T>(data, k_clusters, max_iterations);

  auto &centroids_dkm = std::get<0>(result);
  auto &labels_dkm_u32 = std::get<1>(result); // uint32_t labels from DKM

  // Convert centroids
  std::vector<Point<T, N>> centroids;
  centroids.reserve(centroids_dkm.size());
  for (const auto &c : centroids_dkm)
    centroids.emplace_back(c);

  // Convert labels to size_t
  std::vector<size_t> labels;
  labels.reserve(labels_dkm_u32.size());
  for (auto v : labels_dkm_u32)
    labels.push_back(static_cast<size_t>(v));

  return {std::move(centroids), std::move(labels)};
}

} // namespace ps