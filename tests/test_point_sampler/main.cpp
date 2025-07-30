/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "point_sampler.hpp"
#include "point_sampler/internal/logger.hpp"

int main()
{
  PSLOG->info("testing PointSampler...");

  const size_t dim = 2;
  size_t       count = 100;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {std::make_pair(0.f, 1.f),
                                                     std::make_pair(-1.f, 1.f)};

  {
    PSLOG->info("ps::generate_random_points_white...");
    auto points = ps::generate_random_points_white<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_generate_random_points_white.csv", points);
  }

  {
    PSLOG->info("ps::generate_random_points_jittered_grid...");

    std::array<float, dim> jitter = {0.2f, 0.2f};
    std::array<float, dim> stagger = {0.5f, 0.f};

    auto points = ps::generate_random_points_jittered_grid<float, dim>(count,
                                                                       ranges,
                                                                       jitter,
                                                                       stagger,
                                                                       seed);
    ps::save_points_to_csv("out_generate_random_points_jittered_grid.csv", points);

    // full jitter
    points = ps::generate_random_points_jittered_grid<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_generate_random_points_jittered_grid_full.csv", points);
  }

  return 0;
}
