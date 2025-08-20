/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <functional>
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, size_t N> struct GridND
{
  std::vector<std::optional<Point<T, N>>> cells;
  std::array<size_t, N>                   grid_size{};
  T                                       cell_size;

  GridND(const std::array<size_t, N> &size, T cell_size_)
      : cells(1), grid_size(size), cell_size(cell_size_)
  {
    size_t total = 1;
    for (auto s : size)
      total *= s;
    cells.resize(total);
  }

  // Convert a point coordinate to grid index in each dimension
  std::array<size_t, N> point_to_grid(const Point<T, N>                    &p,
                                      const std::array<std::pair<T, T>, N> &ranges) const
  {
    std::array<size_t, N> idx{};
    for (size_t i = 0; i < N; ++i)
    {
      T clamped = std::min(std::max(p[i], ranges[i].first), ranges[i].second);
      T pos = (clamped - ranges[i].first) / cell_size;
      idx[i] = static_cast<size_t>(std::floor(pos));
      if (idx[i] >= grid_size[i])
        idx[i] = grid_size[i] - 1;
    }
    return idx;
  }

  // Linear index for grid cell
  size_t linear_index(const std::array<size_t, N> &idx) const
  {
    size_t lin_idx = 0;
    size_t stride = 1;
    for (size_t i = 0; i < N; ++i)
    {
      lin_idx += idx[i] * stride;
      stride *= grid_size[i];
    }
    return lin_idx;
  }

  std::optional<Point<T, N>> operator[](const std::array<size_t, N> &idx) const
  {
    return cells[linear_index(idx)];
  }

  std::optional<Point<T, N>> &operator[](const std::array<size_t, N> &idx)
  {
    return cells[linear_index(idx)];
  }
};

template <typename T, size_t N, typename ScaleFn>
bool in_neighborhood(const GridND<T, N>                   &grid,
                     const Point<T, N>                    &p,
                     T                                     base_min_dist,
                     const std::array<std::pair<T, T>, N> &ranges,
                     ScaleFn                               scale_fn)
{
  T scaled_min_dist_p = scale_fn(p) * base_min_dist;

  // Convert point to grid coordinates
  auto idx = grid.point_to_grid(p, ranges);

  // Calculate the radius in cells to check neighbors
  int radius = static_cast<int>(std::ceil(scaled_min_dist_p / grid.cell_size));

  // Iterate neighbor cells in N dimensions (hypercube)
  // We'll do a recursive iteration over N dims

  std::array<int, N> offsets{};
  for (size_t i = 0; i < N; ++i)
    offsets[i] = -radius;

  bool result = false;

  // Recursive lambda to iterate neighbors
  std::function<void(size_t)> check_neighbors;
  check_neighbors = [&](size_t dim)
  {
    if (dim == N)
    {
      // Compute neighbor cell index
      std::array<size_t, N> neighbor_idx{};
      for (size_t d = 0; d < N; ++d)
      {
        int val = static_cast<int>(idx[d]) + offsets[d];
        if (val < 0 || val >= static_cast<int>(grid.grid_size[d]))
          return; // out of grid
                  // bounds
        neighbor_idx[d] = static_cast<size_t>(val);
      }

      const auto &slot = grid[neighbor_idx];
      if (slot)
      {
        T scaled_min_dist_slot = scale_fn(slot.value()) * base_min_dist;
        T dist_thresh = std::max(scaled_min_dist_p, scaled_min_dist_slot);
        T dist_thresh_sq = dist_thresh * dist_thresh;

        Point<T, N> diff = p - slot.value();
        T           dist_sq = T(0);
        for (size_t i = 0; i < N; ++i)
          dist_sq += diff[i] * diff[i];

        if (dist_sq < dist_thresh_sq)
          result = true;
      }
      return;
    }

    for (offsets[dim] = -radius; offsets[dim] <= radius; ++offsets[dim])
    {
      check_neighbors(dim + 1);
      if (result)
        return;
    }
  };

  check_neighbors(0);
  return result;
}

template <typename T, size_t N>
Point<T, N> generate_random_point_around(const Point<T, N> &center,
                                         T                  base_min_dist,
                                         std::mt19937      &gen,
                                         std::function<T(const Point<T, N> &)> scale_fn)
{
  std::uniform_real_distribution<T> dist_angle(0, 2 * M_PI);
  std::uniform_real_distribution<T> dist_radius(0, 1);

  // Generate random direction in N dimensions using normal distribution
  // method
  std::normal_distribution<T> normal(0, 1);

  Point<T, N> dir;
  T           length_dir = 0;
  do
  {
    length_dir = 0;
    for (size_t i = 0; i < N; ++i)
    {
      dir[i] = normal(gen);
      length_dir += dir[i] * dir[i];
    }
    length_dir = std::sqrt(length_dir);
  } while (length_dir == 0);

  for (size_t i = 0; i < N; ++i)
    dir[i] /= length_dir;

  T                                 scaled_min_dist = scale_fn(center) * base_min_dist;
  std::uniform_real_distribution<T> dist_r(scaled_min_dist, 2 * scaled_min_dist);
  T                                 r = dist_r(gen);

  Point<T, N> p;
  for (size_t i = 0; i < N; ++i)
    p[i] = center[i] + dir[i] * r;

  return p;
}

/**
 * @brief Generate a set of Poisson disk samples in N-dimensional space,
 * possibly with a warped metric.
 *
 * This function uses Bridson's algorithm to generate evenly spaced points
 * according to a minimum base distance, which can be warped using a
 * user-defined scaling function (e.g., density or metric warping).
 *
 * @tparam T Scalar type (e.g., float or double).
 * @tparam N Dimension of the sampling space.
 * @tparam ScaleFn Callable type returning a scaling factor at a given point.
 *
 * @param  count               Desired number of points (will attempt to
 *                             generate up to this many).
 * @param  ranges              Coordinate axis ranges (bounding box) for each of
 *                             the N dimensions.
 * @param  base_min_dist       Base minimum distance between any two points
 *                             (before scaling).
 * @param  scale_fn            Function that returns a distance scaling factor
 *                             at a given point. This enables warped-space or
 *                             non-uniform Poisson sampling.
 * @param  seed                Optional RNG seed for reproducibility.
 * @param  new_points_attempts Number of candidate points to try around each
 *                             active point.
 *
 * @return                     std::vector<Point<T, N>> A vector of sample
 *                             points satisfying the scaled Poisson distance
 *                             constraint.
 *
 * @par Example
 * @code
 *  auto ranges = std::array<std::pair<float, float>, 2>{{ {0.f, 1.f}, {0.f, 1.f} }};
 * auto scale_fn = [](const Point<float, 2> &p) -> float {
 *     return 1.0f + 0.5f * std::sin(p[0] * 6.2831f);  // Vary distance with x
 * };
 * auto points = poisson_disk_sampling<float, 2>(100, ranges, 0.05f, scale_fn, 42);
 * @endcode
 *
 * s@image html out_poisson_disk_sampling.csv.jpg
 */
template <typename T, size_t N, typename ScaleFn>
std::vector<Point<T, N>> poisson_disk_sampling(
    size_t                                count,
    const std::array<std::pair<T, T>, N> &ranges,
    T                                     base_min_dist,
    ScaleFn                               scale_fn,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                new_points_attempts = 30)
{
  if (count == 0)
    return {};

  std::mt19937 gen(seed ? *seed : std::random_device{}());

  T cell_size = base_min_dist / std::sqrt(static_cast<T>(N));

  // Compute grid size per axis
  std::array<size_t, N> grid_size{};
  for (size_t i = 0; i < N; ++i)
  {
    T axis_len = ranges[i].second - ranges[i].first;
    grid_size[i] = static_cast<size_t>(std::ceil(axis_len / cell_size));
  }

  GridND<T, N> grid(grid_size, cell_size);

  std::vector<Point<T, N>> sample_points;
  sample_points.reserve(count);

  std::vector<Point<T, N>> process_list;
  process_list.reserve(count);

  // Generate first point randomly inside ranges
  Point<T, N> first_point;
  for (size_t i = 0; i < N; ++i)
  {
    std::uniform_real_distribution<T> dist_axis(ranges[i].first, ranges[i].second);
    first_point[i] = dist_axis(gen);
  }

  sample_points.push_back(first_point);
  process_list.push_back(first_point);

  grid[grid.point_to_grid(first_point, ranges)] = first_point;

  while (!process_list.empty() && sample_points.size() < count)
  {
    // Pop a random element from process_list
    std::uniform_int_distribution<size_t> dist_idx(0, process_list.size() - 1);
    size_t                                idx = dist_idx(gen);
    Point<T, N>                           point = process_list[idx];

    // Remove from process_list (swap-pop)
    process_list[idx] = process_list.back();
    process_list.pop_back();

    for (size_t i = 0; i < new_points_attempts && sample_points.size() < count; ++i)
    {
      Point<T, N> new_point = generate_random_point_around<T, N>(point,
                                                                 base_min_dist,
                                                                 gen,
                                                                 scale_fn);

      // Check bounds
      bool in_bounds = true;
      for (size_t d = 0; d < N; ++d)
      {
        if (new_point[d] < ranges[d].first || new_point[d] > ranges[d].second)
        {
          in_bounds = false;
          break;
        }
      }

      if (!in_bounds)
        continue;

      // Check neighbors with scaled distance
      if (!in_neighborhood(grid, new_point, base_min_dist, ranges, scale_fn))
      {
        sample_points.push_back(new_point);
        process_list.push_back(new_point);
        grid[grid.point_to_grid(new_point, ranges)] = new_point;
      }
    }
  }

  return sample_points;
}

/**
 * @brief Generate uniformly distributed Poisson disk samples in N-dimensional
 * space.
 *
 * This is a convenience wrapper over `poisson_disk_sampling` using a constant
 * distance scale (i.e., uniform metric).
 *
 * @tparam T Scalar type (e.g., float or double).
 * @tparam N Dimension of the sampling space.
 *
 * @param  count               Desired number of points (will attempt to
 *                             generate up to this many).
 * @param  ranges              Coordinate axis ranges (bounding box) for each of
 *                             the N dimensions.
 * @param  base_min_dist       Minimum distance between any two points.
 * @param  seed                Optional RNG seed for reproducibility.
 * @param  new_points_attempts Number of candidate points to try around each
 *                             active point.
 *
 * @return                     std::vector<Point<T, N>> A vector of uniformly
 *                             spaced sample points.
 *
 * @par Example
 * @code
 * auto ranges = std::array<std::pair<float, float>, 2>{{ {0.f, 1.f}, {0.f, 1.f} }};
 * auto points = poisson_disk_sampling_uniform<float, 2>(200, ranges, 0.03f, 1234);
 * @endcode
 *
 * @image html out_poisson_disk_sampling_uniform.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> poisson_disk_sampling_uniform(
    size_t                                count,
    const std::array<std::pair<T, T>, N> &ranges,
    T                                     base_min_dist,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                new_points_attempts = 30)
{
  auto scale_fn = [](const ps::Point<T, N> & /*p*/) -> float { return 1.f; };

  return poisson_disk_sampling(count,
                               ranges,
                               base_min_dist,
                               scale_fn,
                               seed,
                               new_points_attempts);
}

/**
 * @brief Generate random points with a variable-radius Poisson disk sampling. Radius is
 * defined by an input distribution.
 *
 * This algorithm enforces a minimum separation between points based on
 * radii drawn from a user-specified distribution. Two points \f$p_i\f$,
 * \f$p_j\f$ with radii \f$r_i\f$, \f$r_j\f$ must satisfy:
 * \f[
 *   \| p_i - p_j \| > r_i + r_j
 * \f]
 *
 * This produces point sets where local spacing reflects the size
 * distribution: many small radii yield dense clusters, while large
 * radii produce local depletion zones.
 *
 * @tparam T Scalar type (e.g. float, double).
 * @tparam N Dimension of the points.
 * @tparam RadiusGen Callable returning radii sampled from the target distribution.
 *
 * @param n_points Number of points to generate.
 * @param axis_ranges Ranges for each axis, defining the sampling domain.
 * @param radius_gen Generator functor/lambda returning the next radius.
 * @param  seed Optional RNG seed for reproducibility.
 * @param max_attempts Maximum attempts per point before giving up (controls density).
 * @return Vector of generated points satisfying the variable-radius exclusion rule.
 *
 * @note
 * - Larger `max_attempts` increases the chance of filling the domain but
 *   also increases runtime.
 * - For efficiency, use a spatial grid or tree if generating many points.
 * - Radii are drawn independently per point; correlations can be introduced
 *   by adapting `radius_gen`.
 *
 * @par Example
 * @code {.cpp}
 * #include <random>
 *
 * std::mt19937 rng{std::random_device{}()};
 *
 * // 2D unit square
 * std::array<std::pair<double,double>,2> box = { { {0,1}, {0,1} } };
 *
 * // Log-normal radius distribution
 * std::lognormal_distribution<double> logn(0.0, 0.5);
 *
 * auto points = variable_radius_poisson_disk<double,2>(
 *     200,
 *     box,
 *     [&](){ return logn(rng); }
 * );
 * @endcode
 *
 * @image html out_poisson_disk_sampling_distance_distribution.csv.jpg
 */
template <typename T, size_t N, typename RadiusGen>
std::vector<Point<T, N>> poisson_disk_sampling_distance_distribution(
    size_t                                n_points,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    RadiusGen                           &&radius_gen,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                max_attempts = 30)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uniform01(0.0, 1.0);

  std::vector<Point<T, N>> points;
  std::vector<T>           radii;
  points.reserve(n_points);
  radii.reserve(n_points);

  size_t attempts = 0;
  while (points.size() < n_points && attempts < n_points * max_attempts)
  {
    attempts++;

    // candidate
    Point<T, N> p;
    for (size_t d = 0; d < N; ++d)
    {
      auto [a, b] = axis_ranges[d];
      p[d] = a + uniform01(gen) * (b - a);
    }
    T r = radius_gen(); // draw radius

    // check exclusion
    bool valid = true;
    for (size_t j = 0; j < points.size(); ++j)
    {
      T dist = std::sqrt(distance_squared(p, points[j]));
      if (dist < r + radii[j])
      {
        valid = false;
        break;
      }
    }

    if (valid)
    {
      points.push_back(p);
      radii.push_back(r);
    }
  }

  return points;
}

/**
 * @brief Generate N-dimensional points using Poisson disk sampling with a power-law
 * radius distribution.
 *
 * This function generates `n_points` in N-dimensional space such that each point is
 * separated by a local radius sampled from a power-law distribution: \f[ p(r) \propto
 * r^{-\alpha}, \quad r \in [\text{dist_{min}}, \text{dist_{max}}] \f] Smaller radii are
 * more probable than larger ones, creating denser clusters with occasional larger gaps.
 *
 * The sampling respects the axis ranges specified in `axis_ranges` and can optionally use
 * a fixed random seed.
 *
 * @tparam T Scalar type for coordinates (e.g., float, double).
 * @tparam N Dimension of the space.
 * @param n_points Number of points to generate.
 * @param dist_min Minimum radius for the power-law distribution.
 * @param dist_max Maximum radius for the power-law distribution.
 * @param alpha Power-law exponent (\f$\alpha > 0\f$). Larger \f$\alpha\f$ favors smaller
 * distances.
 * @param axis_ranges Array of N pairs specifying min/max range along each axis.
 * @param seed Optional random seed for reproducibility.
 * @param max_attempts Maximum attempts to place a point before skipping (default 30).
 * @return Vector of N-dimensional points satisfying the Poisson disk criteria with
 * power-law distances.
 *
 * @note
 * - Works in arbitrary dimension N.
 * - Uses `poisson_disk_sampling_distance_distribution` internally with a dynamically
 * sampled radius.
 * - Smaller `alpha` produces more uniform spacing; larger `alpha` produces clustered
 * patterns.
 *
 * @par Example
 * @code {.cpp}
 * std::array<std::pair<double,double>,3> ranges = {{{0,1},{0,1},{0,1}}};
 * auto points = poisson_disk_sampling_power_law<double,3>(200, 0.01, 0.2, 1.2, ranges);
 * @endcode
 *
 * @image html out_poisson_disk_sampling_power_law.csv.jpg
 */

template <typename T, size_t N>
std::vector<Point<T, N>> poisson_disk_sampling_power_law(
    size_t                                n_points,
    T                                     dist_min,
    T                                     dist_max,
    T                                     alpha,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                max_attempts = 30)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uni(0.0, 1.0);

  auto power_law_radius = [&]()
  {
    T u = uni(gen);
    return std::pow(std::pow(dist_min, 1 - alpha) + u * (std::pow(dist_max, 1 - alpha) -
                                                         std::pow(dist_min, 1 - alpha)),
                    1.0 / (1 - alpha));
  };

  return poisson_disk_sampling_distance_distribution(n_points,
                                                     axis_ranges,
                                                     power_law_radius,
                                                     seed,
                                                     max_attempts = 30);
}

/**
 * @brief Generate N-dimensional points using Poisson disk sampling with a
 * Weibull-distributed radius.
 *
 * This function generates `n_points` in N-dimensional space such that each point is
 * separated by a local radius sampled from a Weibull distribution: \f[ p(r; k, \lambda) =
 * \frac{k}{\lambda} \left(\frac{r}{\lambda}\right)^{k-1}
 * \exp\left[-\left(\frac{r}{\lambda}\right)^k\right], \quad r \geq 0
 * \f]
 *
 * The Weibull distribution allows flexible control over radius distribution:
 * - Shape parameter \f$k > 0\f$ controls skewness (e.g. \f$k < 1\f$ heavy-tail, \f$k >
 * 1\f$ peaked).
 * - Scale parameter \f$\lambda > 0\f$ sets the typical radius scale.
 *
 * The sampling respects the axis ranges specified in `axis_ranges` and can optionally use
 * a fixed random seed.
 *
 * @tparam T Scalar type for coordinates (e.g., float, double).
 * @tparam N Dimension of the space.
 * @param n_points Number of points to generate.
 * @param lambda Scale parameter of the Weibull distribution.
 * @param k Shape parameter of the Weibull distribution.
 * @param axis_ranges Array of N pairs specifying min/max range along each axis.
 * @param seed Optional random seed for reproducibility.
 * @param max_attempts Maximum attempts to place a point before skipping (default 30).
 * @return Vector of N-dimensional points satisfying the Poisson disk criteria with
 * Weibull-distributed distances.
 *
 * @note
 * - Works in arbitrary dimension N.
 * - Uses `poisson_disk_sampling_distance_distribution` internally with radii sampled from
 * Weibull distribution.
 * - Low shape (\f$k < 1\f$) produces heavy-tailed spacing with more small radii.
 * - High shape (\f$k > 1\f$) produces more peaked, nearly Gaussian-like spacing.
 *
 * @par Example
 * @code {.cpp}
 * std::array<std::pair<double,double>,2> ranges = {{{0,10},{0,10}}};
 * auto points = poisson_disk_sampling_weibull<double,2>(500, 1.0, 2.0, ranges);
 * @endcode
 *
 * @image html out_poisson_disk_sampling_weibull.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> poisson_disk_sampling_weibull(
    size_t                                n_points,
    T                                     lambda,
    T                                     k,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                max_attempts = 30)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uni(0.0, 1.0);

  // Weibull radius generator via inverse CDF
  auto weibull_radius = [&]()
  {
    T u = uni(gen);
    return lambda * std::pow(-std::log(1.0 - u), T(1) / k);
  };

  return poisson_disk_sampling_distance_distribution(n_points,
                                                     axis_ranges,
                                                     weibull_radius,
                                                     seed,
                                                     max_attempts);
}

/**
 * @brief Poisson disk sampling in N dimensions with radii drawn from a Weibull
 * distribution, enforcing a minimum exclusion distance.
 *
 * Each point has an exclusion radius r = max(r_weibull, min_dist).
 * The Weibull distribution is parameterized by scale λ and shape k.
 *
 * @tparam T Floating-point scalar type.
 * @tparam N Dimension of the space.
 * @param n_points Maximum number of points to attempt to place.
 * @param lambda Weibull scale parameter (>0).
 * @param k Weibull shape parameter (>0).
 * @param min_dist Minimum exclusion distance enforced globally.
 * @param axis_ranges Axis-aligned bounding box for the domain.
 * @param seed Optional random seed.
 * @param max_attempts Max attempts to place each point.
 * @return Vector of sampled points.
 *
 * @note
 * - Each point is at least min_dist away from others.
 * - Radii are Weibull-distributed, but truncated below by min_dist.
 * - The effective distribution is Weibull(k, λ) left-truncated at min_dist.
 *
 * @par example
 * @code {.cpp}
 * std::array<std::pair<double,double>, 2> domain = {{{0.0, 1.0}, {0.0, 1.0}}};
 * auto pts = poisson_disk_sampling_weibull<double,2>(
 *                200, 0.2, 1.5, 0.05, domain, 42);
 * @endcode
 *
 * @image html out_poisson_disk_sampling_weibull_min_dist.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> poisson_disk_sampling_weibull(
    size_t                                n_points,
    T                                     lambda,
    T                                     k,
    T                                     dist_min,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    std::optional<unsigned int>           seed = std::nullopt,
    size_t                                max_attempts = 30)
{
  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uni(0.0, 1.0);

  auto weibull_radius = [&]()
  {
    T u = uni(gen);
    // Inverse CDF sampling: r = λ * (-ln(1 - u))^(1/k)
    T r = lambda * std::pow(-std::log(1 - u), T(1) / k);
    return std::max(r, dist_min);
  };

  return poisson_disk_sampling_distance_distribution(n_points,
                                                     axis_ranges,
                                                     weibull_radius,
                                                     seed,
                                                     max_attempts);
}

} // namespace ps
