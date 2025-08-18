/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"
#include "point_sampler/random.hpp"

namespace ps
{

/**
 * @brief Generates clustered points around provided cluster centers using a
 * Gaussian distribution.
 *
 * For each cluster center, this function generates `points_per_cluster` points
 * where each coordinate is sampled from a normal distribution centered at the
 * coordinate of the cluster center with standard deviation `spread`.
 *
 * @tparam T       Scalar type (e.g., float or double)
 * @tparam N       Dimensionality of the space
 * @param  cluster_centers    A vector of cluster center points
 * @param  points_per_cluster Number of points to generate per cluster
 * @param  spread             Standard deviation of the Gaussian spread
 * @param  seed               Optional random seed
 * @return                    std::vector<Point<T, N>> A vector of clustered
 *                            points
 *
 * @par Example
 * @code
 * std::vector<Point<float, 2>> centers = {
 *     {0.2f, 0.2f},
 *     {0.8f, 0.8f}
 * };
 * auto clustered = ps::gaussian_clusters(centers, 100, 0.05f);
 * @endcode
 *
 * @image html out_gaussian_clusters_wrapped.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> gaussian_clusters(
    std::vector<Point<T, N>>    cluster_centers,
    size_t                      points_per_cluster,
    T                           spread,
    std::optional<unsigned int> seed = std::nullopt)
{
  std::mt19937 gen(seed ? *seed : std::random_device{}());

  std::vector<Point<T, N>> points;
  points.reserve(cluster_centers.size() * points_per_cluster);

  for (size_t i = 0; i < cluster_centers.size(); ++i)
  {
    std::normal_distribution<T> dist(0, spread);
    for (size_t j = 0; j < points_per_cluster; ++j)
    {
      Point<T, N> p;
      for (size_t k = 0; k < N; ++k)
        p[k] = cluster_centers[i][k] + dist(gen);
      points.push_back(p);
    }
  }
  return points;
}

/**
 * @brief Generates clustered points around random centers uniformly sampled in
 * a bounding box.
 *
 * Cluster centers are randomly sampled within the provided `axis_ranges`, and
 * each cluster then has `points_per_cluster` points sampled from a Gaussian
 * distribution centered at the cluster's location, with a specified standard
 * deviation `spread`.
 *
 * @tparam T       Scalar type (e.g., float or double)
 * @tparam N       Dimensionality of the space
 * @param  cluster_count      Number of cluster centers to generate
 * @param  points_per_cluster Number of points per cluster
 * @param  axis_ranges        Axis-aligned bounding box ranges for each
 *                            dimension
 * @param  spread             Standard deviation of the Gaussian spread
 * @param  seed               Optional random seed
 * @return                    std::vector<Point<T, N>> A vector of clustered
 *                            points
 *
 * @par Example
 * @code auto clustered = ps::gaussian_clusters<float, 2>(
 *     5, 100,
 *     {{{0,1}, {0,1}}}, 0.03f
 * );
 * @endcode
 */
template <typename T, size_t N>
std::vector<Point<T, N>> gaussian_clusters(
    size_t                                cluster_count,
    size_t                                points_per_cluster,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    T                                     spread,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::vector<Point<T, N>> cluster_centers = random<T, N>(cluster_count,
                                                          axis_ranges,
                                                          seed);
  std::vector<Point<T, N>> points = gaussian_clusters(cluster_centers,
                                                      points_per_cluster,
                                                      spread,
                                                      seed);
  return points;
}

} // namespace ps
