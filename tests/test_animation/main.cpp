/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "point_sampler.hpp"

std::string helper_zfill(const std::string &str, size_t n_zeros)
{
  // https://stackoverflow.com/questions/6143824
  return std::string(n_zeros - std::min(n_zeros, str.length()), '0') + str;
}

int main()
{
  std::cout << "testing PointSampler...\n";

  const size_t dim = 2;
  size_t       count = 1000;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {std::make_pair(0.f, 2.f),
                                                     std::make_pair(0.f, 1.f)};

  // relaxation
  if (false)
  {
    auto points = ps::random<float, dim>(count, ranges, seed);

    size_t k_neighbors = 8;
    float  step_size = 0.001f;
    size_t iterations = 100;

    for (size_t it = 0; it < iterations; ++it)
    {
      std::string sit = helper_zfill(std::to_string(it), 4);

      ps::relaxation_ktree<float, dim>(points, k_neighbors, step_size, 1);
      ps::filter_points_in_range(points, ranges);
      ps::save_points_to_csv("anim_relaxation_ktree_" + sit + ".csv", points);
    }
  }

  // filtering
  if (false)
  {
    auto points = ps::random<float, dim>(5 * count, ranges, seed);

    auto scale_function = [](const ps::Point<float, 2> &p) -> float
    {
      float x = p[0], y = p[1];
      return 1.f + 1.f * std::sin(8.0f * x) * std::cos(8.0f * y);
    };

    float  step = 0.001f;
    size_t it = 0;
    for (float min_dist = step; min_dist < 0.1f; min_dist += step)
    {
      std::string sit = helper_zfill(std::to_string(it), 4);

      points = ps::distance_rejection_filter_warped<float, dim>(points,
                                                                min_dist,
                                                                scale_function);
      ps::save_points_to_csv("anim_distance_filter_" + sit + ".csv", points);

      it++;
    }
  }

  // Poisson disc
  if (false)
  {
    float  base_min_dist = 0.02f;
    float  step = 0.02f;
    size_t it = 0;
    for (float amp = 0.f; amp < 2.f; amp += step)
    {
      std::string sit = helper_zfill(std::to_string(it), 4);

      auto scale_function = [amp](const ps::Point<float, 2> &p) -> float
      {
        float x = p[0], y = p[1];
        return 1.f + amp * (1.f + std::sin(8.0f * x) * std::cos(8.0f * y));
      };

      auto points = ps::poisson_disk_sampling<float, dim>(5 * count,
                                                          ranges,
                                                          base_min_dist,
                                                          scale_function,
                                                          seed);
      ps::save_points_to_csv("anim_poisson_" + sit + ".csv", points);

      it++;
    }
  }

  return 0;
}
