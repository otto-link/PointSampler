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
 * @brief Compute the angular distribution function (ADF) using nearest
 * neighbors.
 *
 * The angular distribution function (ADF) measures the distribution of bond
 * angles formed by a point and pairs of its nearest neighbors.
 *
 * - Flat distribution → random uniform points.
 * - Peaks at characteristic angles → local order (e.g. hexagonal lattice peaks
 * at 60°).
 * - Depletions → angular avoidance due to constraints or repulsion.
 *
 * @tparam T Numeric type (float, double, ...)
 * @tparam N Dimension of points (N >= 2)
 * @param  points      Vector of points
 * @param  bin_width   Width of angle bins (in radians)
 * @param  k_neighbors Number of neighbors used for angle calculation (default: 8)
 * @return             std::pair<std::vector<T>, std::vector<T>>
 *         - First: bin centers (angles in radians)
 *         - Second: normalized ADF values
 *
 * @par Example
 * @code
 * std::vector<Point<double, 2>> pts = {
 *     {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
 * };
 *
 * auto [theta, g_theta] = angle_distribution_neighbors(pts, 0.1, 6);
 *
 * for (size_t i = 0; i < theta.size(); ++i)
 *   std::cout << "θ=" << theta[i] << "g(θ)=" << g_theta[i] << std::endl;
 * @endcode
 */
template <typename T, size_t N>
std::pair<std::vector<T>, std::vector<T>> angle_distribution_neighbors(
    const std::vector<Point<T, N>> &points,
    T                               bin_width,
    size_t                          k_neighbors = 8)
{
  if constexpr (N < 2)
    throw std::runtime_error("Angle distribution requires dimension >= 2");

  size_t         num_bins = static_cast<size_t>(std::ceil(M_PI / bin_width));
  std::vector<T> angles(num_bins);
  std::vector<T> g_theta(num_bins, T(0));

  // Precompute bin centers
  for (size_t i = 0; i < num_bins; ++i)
    angles[i] = (i + T(0.5)) * bin_width;

  // Get nearest neighbors once
  auto neighbors = nearest_neighbors_indices(points, k_neighbors);

  // Loop over all points
  for (size_t i = 0; i < points.size(); ++i)
  {
    const auto &nbrs = neighbors[i];
    size_t      n_nbrs = nbrs.size();

    // Compute angles between all neighbor pairs
    for (size_t a = 0; a < n_nbrs; ++a)
    {
      for (size_t b = a + 1; b < n_nbrs; ++b)
      {
        size_t j = nbrs[a];
        size_t k = nbrs[b];

        // Vectors from center i to j and k
        std::array<T, N> v1, v2;
        for (size_t d = 0; d < N; ++d)
        {
          v1[d] = points[j][d] - points[i][d];
          v2[d] = points[k][d] - points[i][d];
        }

        // Compute angle
        T dot = T(0), norm1 = T(0), norm2 = T(0);
        for (size_t d = 0; d < N; ++d)
        {
          dot += v1[d] * v2[d];
          norm1 += v1[d] * v1[d];
          norm2 += v2[d] * v2[d];
        }

        if (norm1 > T(0) && norm2 > T(0))
        {
          T cos_theta = std::clamp(dot / (std::sqrt(norm1) * std::sqrt(norm2)),
                                   T(-1),
                                   T(1));
          T theta = std::acos(cos_theta);

          size_t bin = static_cast<size_t>(theta / bin_width);
          if (bin < num_bins)
            g_theta[bin] += T(1);
        }
      }
    }
  }

  // Normalize histogram
  T total = std::accumulate(g_theta.begin(), g_theta.end(), T(0));
  if (total > T(0))
    for (auto &val : g_theta)
      val /= total;

  return {angles, g_theta};
}

/**
 * @brief Compute the distance of each point to the domain boundary.
 *
 * The domain is defined by axis-aligned ranges in each dimension. For each
 * point, the returned distance is the smallest Euclidean distance to any
 * boundary plane of the domain.
 *
 * @tparam T Floating point type.
 * @tparam N Dimensionality of the points.
 * @param  points      Vector of points to evaluate.
 * @param  axis_ranges Vector of size N, where each element is a std::pair<min,
 *                     max>
 *                    defining the domain limits in that dimension.
 * @return             std::vector<T> Distances of each point to the nearest
 *                     domain boundary.
 *
 * @note This assumes the domain is a rectangular box aligned with the
 * coordinate axes.
 *
 * @par Example
 * @code
 * std::vector<Point<double, 2>> pts = { {0.2, 0.8}, {0.9, 0.1} };
 * std::vector<std::pair<double, double>> ranges = { {0.0, 1.0}, {0.0, 1.0} };
 * auto distances = distance_to_boundary(pts, ranges);
 * // distances[0] -> 0.2
 * // distances[1] -> 0.1
 * @endcode
 */
template <typename T, size_t N>
std::vector<T> distance_to_boundary(const std::vector<Point<T, N>>       &points,
                                    const std::array<std::pair<T, T>, N> &axis_ranges)
{
  std::vector<T> distances;
  distances.reserve(points.size());

  for (const auto &p : points)
  {
    // Find min distance to a boundary plane
    T min_dist = std::numeric_limits<T>::max();

    for (size_t d = 0; d < N; ++d)
    {
      T dist_to_min = std::abs(p[d] - axis_ranges[d].first);
      T dist_to_max = std::abs(axis_ranges[d].second - p[d]);

      min_dist = std::min({min_dist, dist_to_min, dist_to_max});
    }

    distances.push_back(min_dist);
  }

  return distances;
}

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
 * std::vector<float> distances_sq = first_neighbor_distance_squared(points);
 *
 * // distances_sq[i] contains the squared distance to the closest neighbor of points[i].
 * @endcode
 *
 * @image html metrics_first_neighbor_distance.jpg
 */
template <typename T, size_t N>
std::vector<T> first_neighbor_distance_squared(std::vector<Point<T, N>> &points)
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

/**
 * @brief Compute local point density based on k-nearest neighbors in N dimensions.
 *
 * The density at each point is estimated as:
 * \f[
 * \rho_i = \frac{k}{V_N r_i^N}
 * \f]
 * where \(r_i\) is the distance to the k-th nearest neighbor, \(V_N\) is the volume
 * of the unit N-ball:
 * \f[
 * V_N = \frac{\pi^{N/2}}{\Gamma(N/2 + 1)}
 * \f]
 * and \(\Gamma\) is the gamma function. This generalizes to any dimension.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of the points.
 * @param points Vector of points in N-dimensional space.
 * @param k Number of nearest neighbors to use for density estimation (should be >= 1).
 * @return Vector of local densities for each point.
 *
 * @par Example
 * @code{.cpp}
 * std::vector<Point<double,3>> pts = ...; // 3D points
 * size_t k = 8;
 * auto densities = local_density_knn<double,3>(pts, k);
 * @endcode
 *
 * @note
 * - Works in any dimension \(N\).
 * - High k gives smoother density estimates, low k captures local fluctuations.
 * - The resulting density units are “points per unit volume” in N dimensions.
 *
 * @image html metrics_local_density_knn.jpg
 */
template <typename T, size_t N>
std::vector<T> local_density_knn(const std::vector<Point<T, N>> &points, size_t k = 8)
{
  auto neighbors = nearest_neighbors_indices(points, k);

  std::vector<T> densities(points.size());

  // Compute unit N-ball volume using gamma function
  T volume_unit_ball = std::pow(M_PI, T(N) / 2) / std::tgamma(T(N) / 2 + 1);

  for (size_t i = 0; i < points.size(); ++i)
  {
    // Distance to k-th nearest neighbor
    const auto &nk_idx = neighbors[i].back();
    T           dist2 = 0;
    for (size_t d = 0; d < N; ++d)
    {
      T diff = points[i][d] - points[nk_idx][d];
      dist2 += diff * diff;
    }
    T r = std::sqrt(dist2);

    // Density = k / (V_N * r^N)
    T volume = volume_unit_ball * std::pow(r, T(N));
    densities[i] = k / volume;
  }

  return densities;
}

/**
 * @brief Finds the nearest neighbors for each point in a set.
 *
 * This function uses a KD-tree to search for the k nearest neighbors of each
 * point in the input set, returning their indices. The search excludes the
 * query point itself.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of each point.
 *
 * @param  points      Vector of N-dimensional points.
 * @param  k_neighbors Number of nearest neighbors to return for each point.
 *
 * @return             A vector where each element is a vector of indices
 *                     representing the nearest neighbors of the corresponding
 *                     point in `points`.
 *
 * @note The KD-tree is rebuilt internally for the search.
 *
 * @par Example
 * @code
 * std::vector<Point<float, 3>> points = {
 *     {0.0f, 0.0f, 0.0f},
 *     {1.0f, 0.0f, 0.0f},
 *     {0.0f, 1.0f, 0.0f},
 *     {1.0f, 1.0f, 0.0f}
 * };
 *
 * auto neighbors = nearest_neighbors_indices(points, 2);
 * // neighbors[0] might contain {1, 2}
 * // neighbors[1] might contain {0, 3}
 * @endcode
 *
 * @image html metrics_nearest_neighbors_indices.jpg
 */
template <typename T, size_t N>
std::vector<std::vector<size_t>> nearest_neighbors_indices(
    const std::vector<Point<T, N>> &points,
    size_t                          k_neighbors = 8)
{
  PointCloudAdaptor<T, N> adaptor(points);
  KDTree<T, N>            index(N, adaptor);
  index.buildIndex();

  std::vector<std::vector<size_t>> all_neighbors;
  all_neighbors.resize(points.size());

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

    // Skip self at index 0
    all_neighbors[i].assign(ret_indexes.begin() + 1, ret_indexes.end());
  }

  return all_neighbors;
}

/**
 * @brief Compute the normalized radial distribution function g(r).
 *
 * The radial distribution function (RDF) describes how the density of points
 * varies as a function of distance from a reference point.
 *
 * - g(r) ≈ 1 → uniform / random distribution at distance r
 * - g(r) > 1 → clustering / aggregation (excess probability of finding
 * neighbors)
 * - g(r) < 1 → depletion / exclusion (points repel or avoid each other)
 *
 * This function normalizes the observed pair distances against the expected
 * density in the domain (given by axis_ranges).
 *
 * @tparam T Numeric type (float, double, ...)
 * @tparam N Dimension of points
 * @param  points       Vector of points
 * @param  axis_ranges  Axis-aligned domain ranges for each dimension
 * @param  bin_width    Width of distance bins
 * @param  max_distance Maximum distance to consider
 * @return              std::pair<std::vector<T>, std::vector<T>>
 *         - First: radii (bin centers)
 *         - Second: normalized RDF values g(r)
 *
 * @par Example
 * @code
 * std::vector<Point<double, 2>> pts = {
 *     {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
 * };
 *
 * std::array<std::pair<double,double>,2> ranges = {
 *     std::make_pair(0.0, 1.0), std::make_pair(0.0, 1.0)
 * };
 *
 * auto [r, g] = radial_distribution(pts, ranges, 0.1, 2.0);
 *
 * for (size_t i = 0; i < r.size(); ++i)
 *   std::cout << "r=" << r[i] << " g(r)=" << g[i] << std::endl;
 * @endcode
 */
template <typename T, size_t N>
std::pair<std::vector<T>, std::vector<T>> radial_distribution(
    const std::vector<Point<T, N>>       &points,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    T                                     bin_width,
    T                                     max_distance)
{
  size_t         num_bins = static_cast<size_t>(std::ceil(max_distance / bin_width));
  std::vector<T> radii(num_bins);
  std::vector<T> g(num_bins, T(0));

  // Precompute bin centers
  for (size_t i = 0; i < num_bins; ++i)
    radii[i] = (i + T(0.5)) * bin_width;

  // Compute volume of domain
  T volume = T(1);
  for (const auto &range : axis_ranges)
    volume *= (range.second - range.first);

  size_t n_points = points.size();
  T      density = static_cast<T>(n_points) / volume;

  // Histogram of pair distances
  for (size_t i = 0; i < n_points; ++i)
  {
    for (size_t j = i + 1; j < n_points; ++j)
    {
      T dist = distance(points[i], points[j]);
      if (dist < max_distance)
      {
        size_t bin = static_cast<size_t>(dist / bin_width);
        if (bin < num_bins)
          g[bin] += T(2); // count both i→j and
                          // j→i
      }
    }
  }

  // Normalize
  T norm_factor = (T)n_points * density;

  // Volume of spherical shell between r1 and r2 in N dimensions
  auto sphere_volume = [](T radius)
  { return std::pow(M_PI, N / 2.0) / std::tgamma(N / 2.0 + 1.0) * std::pow(radius, N); };

  for (size_t i = 0; i < num_bins; ++i)
  {
    T r_inner = i * bin_width;
    T r_outer = (i + 1) * bin_width;

    // Volume of spherical shell
    T shell_vol = sphere_volume(r_outer) - sphere_volume(r_inner);
    g[i] /= norm_factor * shell_vol;
  }

  return {radii, g};
}

} // namespace ps