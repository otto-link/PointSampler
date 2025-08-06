/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <functional>
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, std::size_t N> struct GridND
{
  std::vector<std::optional<Point<T, N>>> cells;
  std::array<std::size_t, N>              grid_size{};
  T                                       cell_size;

  GridND(const std::array<std::size_t, N> &size, T cell_size_)
      : cells(1), grid_size(size), cell_size(cell_size_)
  {
    std::size_t total = 1;
    for (auto s : size)
      total *= s;
    cells.resize(total);
  }

  // Convert a point coordinate to grid index in each dimension
  std::array<std::size_t, N> point_to_grid(
      const Point<T, N>                    &p,
      const std::array<std::pair<T, T>, N> &ranges) const
  {
    std::array<std::size_t, N> idx{};
    for (std::size_t i = 0; i < N; ++i)
    {
      T clamped = std::min(std::max(p[i], ranges[i].first), ranges[i].second);
      T pos = (clamped - ranges[i].first) / cell_size;
      idx[i] = static_cast<std::size_t>(std::floor(pos));
      if (idx[i] >= grid_size[i])
        idx[i] = grid_size[i] - 1;
    }
    return idx;
  }

  // Linear index for grid cell
  std::size_t linear_index(const std::array<std::size_t, N> &idx) const
  {
    std::size_t lin_idx = 0;
    std::size_t stride = 1;
    for (std::size_t i = 0; i < N; ++i)
    {
      lin_idx += idx[i] * stride;
      stride *= grid_size[i];
    }
    return lin_idx;
  }

  std::optional<Point<T, N>> operator[](const std::array<std::size_t, N> &idx) const
  {
    return cells[linear_index(idx)];
  }

  std::optional<Point<T, N>> &operator[](const std::array<std::size_t, N> &idx)
  {
    return cells[linear_index(idx)];
  }
};

template <typename T, std::size_t N, typename ScaleFn>
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
  std::function<void(std::size_t)> check_neighbors;
  check_neighbors = [&](std::size_t dim)
  {
    if (dim == N)
    {
      // Compute neighbor cell index
      std::array<std::size_t, N> neighbor_idx{};
      for (std::size_t d = 0; d < N; ++d)
      {
        int val = static_cast<int>(idx[d]) + offsets[d];
        if (val < 0 || val >= static_cast<int>(grid.grid_size[d]))
          return; // out of grid bounds
        neighbor_idx[d] = static_cast<std::size_t>(val);
      }

      const auto &slot = grid[neighbor_idx];
      if (slot)
      {
        T scaled_min_dist_slot = scale_fn(slot.value()) * base_min_dist;
        T dist_thresh = std::max(scaled_min_dist_p, scaled_min_dist_slot);
        T dist_thresh_sq = dist_thresh * dist_thresh;

        Point<T, N> diff = p - slot.value();
        T           dist_sq = T(0);
        for (std::size_t i = 0; i < N; ++i)
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

template <typename T, std::size_t N>
Point<T, N> generate_random_point_around(const Point<T, N> &center,
                                         T                  base_min_dist,
                                         std::mt19937      &gen,
                                         std::function<T(const Point<T, N> &)> scale_fn)
{
  std::uniform_real_distribution<T> dist_angle(0, 2 * M_PI);
  std::uniform_real_distribution<T> dist_radius(0, 1);

  // Generate random direction in N dimensions using normal distribution method
  std::normal_distribution<T> normal(0, 1);

  Point<T, N> dir;
  T           length_dir = 0;
  do
  {
    length_dir = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
      dir[i] = normal(gen);
      length_dir += dir[i] * dir[i];
    }
    length_dir = std::sqrt(length_dir);
  } while (length_dir == 0);

  for (std::size_t i = 0; i < N; ++i)
    dir[i] /= length_dir;

  T                                 scaled_min_dist = scale_fn(center) * base_min_dist;
  std::uniform_real_distribution<T> dist_r(scaled_min_dist, 2 * scaled_min_dist);
  T                                 r = dist_r(gen);

  Point<T, N> p;
  for (std::size_t i = 0; i < N; ++i)
    p[i] = center[i] + dir[i] * r;

  return p;
}

template <typename T, std::size_t N, typename ScaleFn>
std::vector<Point<T, N>> poisson_disk_sampling(
    std::size_t                           count,
    const std::array<std::pair<T, T>, N> &ranges,
    T                                     base_min_dist,
    ScaleFn                               scale_fn,
    std::optional<unsigned int>           seed = std::nullopt,
    std::size_t                           new_points_attempts = 30)
{
  if (count == 0)
    return {};

  std::mt19937 gen(seed ? *seed : std::random_device{}());

  T cell_size = base_min_dist / std::sqrt(static_cast<T>(N));

  // Compute grid size per axis
  std::array<std::size_t, N> grid_size{};
  for (std::size_t i = 0; i < N; ++i)
  {
    T axis_len = ranges[i].second - ranges[i].first;
    grid_size[i] = static_cast<std::size_t>(std::ceil(axis_len / cell_size));
  }

  GridND<T, N> grid(grid_size, cell_size);

  std::vector<Point<T, N>> sample_points;
  sample_points.reserve(count);

  std::vector<Point<T, N>> process_list;
  process_list.reserve(count);

  // Generate first point randomly inside ranges
  Point<T, N> first_point;
  for (std::size_t i = 0; i < N; ++i)
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
    std::uniform_int_distribution<std::size_t> dist_idx(0, process_list.size() - 1);
    std::size_t                                idx = dist_idx(gen);
    Point<T, N>                                point = process_list[idx];

    // Remove from process_list (swap-pop)
    process_list[idx] = process_list.back();
    process_list.pop_back();

    for (std::size_t i = 0; i < new_points_attempts && sample_points.size() < count; ++i)
    {
      Point<T, N> new_point = generate_random_point_around<T, N>(point,
                                                                 base_min_dist,
                                                                 gen,
                                                                 scale_fn);

      // Check bounds
      bool in_bounds = true;
      for (std::size_t d = 0; d < N; ++d)
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

// wrapper, uniform density
template <typename T, std::size_t N>
std::vector<Point<T, N>> poisson_disk_sampling_uniform(
    std::size_t                           count,
    const std::array<std::pair<T, T>, N> &ranges,
    T                                     base_min_dist,
    std::optional<unsigned int>           seed = std::nullopt,
    std::size_t                           new_points_attempts = 30)
{
  auto scale_fn = [](const ps::Point<T, N> & /*p*/) -> float { return 1.f; };

  return poisson_disk_sampling(count,
                               ranges,
                               base_min_dist,
                               scale_fn,
                               seed,
                               new_points_attempts);
}

} // namespace ps