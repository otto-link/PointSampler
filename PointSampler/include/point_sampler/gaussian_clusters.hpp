/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"
#include "point_sampler/random.hpp"

namespace ps
{

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