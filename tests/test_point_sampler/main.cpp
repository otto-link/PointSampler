/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "point_sampler.hpp"
#include "point_sampler/internal/logger.hpp"

int main()
{
  PSLOG->info("testing PointSampler...");

  const size_t dim = 2;
  size_t       count = 1000;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {std::make_pair(-1.f, 1.f),
                                                     std::make_pair(-2.f, 2.f)};

  {
    PSLOG->info("ps::random...");

    auto points = ps::random<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_random.csv", points);
  }

  {
    PSLOG->info("ps::jittered_grid...");

    std::array<float, dim> jitter = {0.2f, 0.2f};
    std::array<float, dim> stagger = {0.5f, 0.f};

    auto points = ps::jittered_grid<float, dim>(count, ranges, jitter, stagger, seed);
    ps::save_points_to_csv("out_jittered_grid.csv", points);

    // full jitter
    points = ps::jittered_grid<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_jittered_grid_full.csv", points);
  }

  {
    PSLOG->info("ps::rejection_sampling...");

    auto density = [](const ps::Point<float, 2> &p) -> float
    {
      // Gaussian centered at origin
      return std::exp(-2.f * (p[0] * p[0] + p[1] * p[1]));
    };

    auto points = ps::rejection_sampling<float, 2>(count, ranges, density, seed);
    ps::save_points_to_csv("out_rejection_sampling.csv", points);
  }

  {
    PSLOG->info("ps::importance_resampling...");

    auto density = [](const ps::Point<float, 2> &p) -> float
    {
      // Gaussian centered at origin
      return std::exp(-2.f * (p[0] * p[0] + p[1] * p[1]));
    };

    std::size_t oversampling_ratio = 1000;

    auto points = ps::importance_resampling<float, 2>(count,
                                                      oversampling_ratio,
                                                      ranges,
                                                      density,
                                                      seed);
    ps::save_points_to_csv("out_importance_resampling.csv", points);
  }

  return 0;
}
