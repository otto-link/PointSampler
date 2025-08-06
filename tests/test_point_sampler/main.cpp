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
    PSLOG->info("ps::hammersley...");

    auto points = ps::hammersley<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_hammersley.csv", points);
  }

  {
    PSLOG->info("ps::halton...");

    auto points = ps::halton<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_halton.csv", points);
  }

  {
    PSLOG->info("ps::jittered_grid...");

    std::array<float, dim> jitter = {0.3f, 0.3f};
    std::array<float, dim> stagger = {0.5f, 0.f};

    auto points = ps::jittered_grid<float, dim>(count, ranges, jitter, stagger, seed);
    ps::save_points_to_csv("out_jittered_grid.csv", points);

    // full jitter
    points = ps::jittered_grid<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_jittered_grid_full.csv", points);
  }

  {
    PSLOG->info("ps::poisson_disk_sampling...");

    // minimum distance scaling function
    auto scale_fn = [](const ps::Point<float, dim> &p) -> float
    {
      // scale = 1 at origin and scale = 4 far away
      return 1.f + 3.f * (1.f - std::exp(-2.f * (p[0] * p[0] + p[1] * p[1])));
    };

    float base_min_dist = 0.05f;

    auto points = ps::poisson_disk_sampling<float, dim>(count,
                                                        ranges,
                                                        base_min_dist,
                                                        scale_fn,
                                                        seed);
    ps::save_points_to_csv("out_poisson_disk_sampling.csv", points);

    // uniform, no scaling
    base_min_dist = 0.1f;

    points = ps::poisson_disk_sampling_uniform<float, dim>(count,
                                                           ranges,
                                                           base_min_dist,
                                                           seed);
    ps::save_points_to_csv("out_poisson_disk_sampling_uniform.csv", points);
  }

  {
    PSLOG->info("ps::rejection_sampling...");

    auto density = [](const ps::Point<float, dim> &p) -> float
    {
      // Gaussian centered at origin
      return std::exp(-2.f * (p[0] * p[0] + p[1] * p[1]));
    };

    auto points = ps::rejection_sampling<float, dim>(count, ranges, density, seed);
    ps::save_points_to_csv("out_rejection_sampling.csv", points);
  }

  {
    PSLOG->info("ps::importance_resampling...");

    auto density = [](const ps::Point<float, dim> &p) -> float
    {
      // Gaussian centered at origin
      return std::exp(-2.f * (p[0] * p[0] + p[1] * p[1]));
    };

    size_t oversampling_ratio = 1000;

    auto points = ps::importance_resampling<float, dim>(count,
                                                        oversampling_ratio,
                                                        ranges,
                                                        density,
                                                        seed);
    ps::save_points_to_csv("out_importance_resampling.csv", points);
  }

  {
    PSLOG->info("ps::random...");

    size_t cluster_count = 10;
    size_t points_per_cluster = 50;
    float  spread = 0.1f;

    auto cluster_centers = ps::random<float, dim>(cluster_count, ranges, seed);
    auto points = ps::gaussian_clusters(cluster_centers,
                                        points_per_cluster,
                                        spread,
                                        seed);
    ps::save_points_to_csv("out_gaussian_clusters.csv", points);

    // wrapper
    points = ps::gaussian_clusters(cluster_count,
                                   points_per_cluster,
                                   ranges,
                                   spread,
                                   seed);
    ps::save_points_to_csv("out_gaussian_clusters_wrapped.csv", points);
  }

  {
    PSLOG->info("ps::relaxation_ktree...");

    auto   points = ps::random<float, dim>(count, ranges, seed);
    size_t k_neighbors = 8;
    float  step_size = 0.01f;
    size_t iterations = 10;

    ps::relaxation_ktree<float, dim>(points, k_neighbors, step_size, iterations);
    ps::save_points_to_csv("out_relaxation_ktree.csv", points);

    // remove pts outside the initial bounding box
    ps::filter_points_in_range(points, ranges);
    ps::save_points_to_csv("out_relaxation_ktree_filtered.csv", points);

    // rescale to fit in initial bounding box
    ps::refit_points_to_range(points, ranges);
    ps::save_points_to_csv("out_relaxation_ktree_refit.csv", points);
  }

  {
    PSLOG->info("ps::distance_rejection_filter...");

    float min_dist = 0.1f;

    auto points = ps::random<float, dim>(count, ranges, seed);
    points = ps::distance_rejection_filter<float, dim>(points, min_dist);

    ps::save_points_to_csv("out_distance_rejection_filter.csv", points);
  }

  return 0;
}
