/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "point_sampler.hpp"

int main()
{
  std::cout << "testing PointSampler...\n";

  const size_t dim = 2;
  size_t       count = 1000;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {std::make_pair(-1.f, 1.f),
                                                     std::make_pair(-2.f, 2.f)};

  {
    std::cout << "ps::random...\n";

    auto points = ps::random<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_random.csv", points);
  }

  {
    std::cout << "ps::hammersley...\n";

    auto points = ps::hammersley<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_hammersley.csv", points);
  }

  {
    std::cout << "ps::halton...\n";

    auto points = ps::halton<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_halton.csv", points);
  }

  {
    std::cout << "ps::latin_hypercube_sampling...\n";

    auto points = ps::latin_hypercube_sampling<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_latin_hypercube_sampling.csv", points);
  }

  {
    std::cout << "ps::jittered_grid...\n";

    std::array<float, dim> jitter = {0.3f, 0.3f};
    std::array<float, dim> stagger = {0.5f, 0.f};

    auto points = ps::jittered_grid<float, dim>(count, ranges, jitter, stagger, seed);
    ps::save_points_to_csv("out_jittered_grid.csv", points);

    // full jitter
    points = ps::jittered_grid<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("out_jittered_grid_full.csv", points);
  }

  {
    std::cout << "ps::poisson_disk_sampling...\n";

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
    std::cout << "ps::poisson_disk_sampling_distance_distribution...\n";

    // Log-normal radius distribution
    std::lognormal_distribution<float> logn(0.f, 2.f);
    std::mt19937                       rng{std::random_device{}()};

    auto points = ps::poisson_disk_sampling_distance_distribution<float, dim>(
        count,
        ranges,
        [&]() { return logn(rng); },
        seed);

    ps::save_points_to_csv("out_poisson_disk_sampling_distance_distribution.csv", points);
  }

  {
    std::cout << "ps::poisson_disk_sampling_power_law...\n";

    float dist_min = 0.01f;
    float dist_max = 0.2f;
    float alpha = 1.2f;

    auto points = ps::poisson_disk_sampling_power_law<float, dim>(count,
                                                                  dist_min,
                                                                  dist_max,
                                                                  alpha,
                                                                  ranges,
                                                                  seed);

    ps::save_points_to_csv("out_poisson_disk_sampling_power_law.csv", points);
  }

  {
    std::cout << "ps::poisson_disk_sampling_weibull...\n";

    float lambda = 0.05f;
    float k = 0.8f;

    auto points = ps::poisson_disk_sampling_weibull<float, dim>(count,
                                                                lambda,
                                                                k,
                                                                ranges,
                                                                seed);

    ps::save_points_to_csv("out_poisson_disk_sampling_weibull.csv", points);
  }

  {
    std::cout << "ps::poisson_disk_sampling_weibull_min_dist...\n";

    float lambda = 0.05f;
    float k = 0.8f;
    float dist_min = 0.025f;

    auto points = ps::poisson_disk_sampling_weibull<float, dim>(count,
                                                                lambda,
                                                                k,
                                                                dist_min,
                                                                ranges,
                                                                seed);

    ps::save_points_to_csv("out_poisson_disk_sampling_weibull_min_dist.csv", points);
  }

  {
    std::cout << "ps::random_walk_filaments...\n";

    size_t n_filaments = 4;
    size_t filament_count = 100;
    float  step_size = 0.05;
    float  persistence = 0.8f;
    float  gaussian_sigma = 0.1f; // filament thickness
    size_t gaussian_samples = 10;

    std::vector<float> distances = {}; // optional, distance to the filament

    auto points = ps::random_walk_filaments(n_filaments,
                                            filament_count,
                                            step_size,
                                            ranges,
                                            seed,
                                            persistence,
                                            gaussian_sigma,
                                            gaussian_samples,
                                            &distances);

    ps::save_points_to_csv("out_random_walk_filaments.csv", points);
    ps::save_vector_to_csv("metrics_random_walk_filaments_dst.csv", distances);
  }

  {
    std::cout << "ps::rejection_sampling...\n";

    auto density = [](const ps::Point<float, dim> &p) -> float
    {
      // Gaussian centered at origin
      return std::exp(-2.f * (p[0] * p[0] + p[1] * p[1]));
    };

    auto points = ps::rejection_sampling<float, dim>(count, ranges, density, seed);
    ps::save_points_to_csv("out_rejection_sampling.csv", points);
  }

  {
    std::cout << "ps::importance_resampling...\n";

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
    std::cout << "ps::random...\n";

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
    std::cout << "ps::relaxation_ktree...\n";

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
    std::cout << "ps::distance_rejection_filter...\n";

    float min_dist = 0.1f;

    auto points = ps::random<float, dim>(count, ranges, seed);
    points = ps::distance_rejection_filter<float, dim>(points, min_dist);

    ps::save_points_to_csv("out_distance_rejection_filter.csv", points);
  }

  {
    std::cout << "ps::distance_rejection_filter_warped...\n";

    float min_dist = 0.05f;

    auto scale_function = [](const ps::Point<float, 2> &p) -> float
    {
      float x = p[0], y = p[1];
      return 1.f + 1.f * std::sin(4.0f * x) * std::cos(4.0f * y);
    };

    auto points = ps::random<float, dim>(5 * count, ranges, seed);
    points = ps::distance_rejection_filter_warped<float, dim>(points,
                                                              min_dist,
                                                              scale_function);

    ps::save_points_to_csv("out_distance_rejection_filter_warped.csv", points);
  }

  {
    std::cout << "ps::random_rejection_filter_warped...\n";

    float keep_fraction = 0.5f;

    auto points = ps::random<float, dim>(count, ranges, seed);
    points = ps::random_rejection_filter<float, dim>(points, keep_fraction);

    ps::save_points_to_csv("out_random_rejection_filter.csv", points);
  }

  {
    std::cout << "ps::first_neighbor_distance...\n";

    auto               points = ps::random<float, dim>(50, ranges, seed);
    std::vector<float> dist_sq = ps::first_neighbor_distance_squared(points);

    ps::save_points_to_csv("metrics_first_neighbor_distance.csv", points);
    ps::save_vector_to_csv("metrics_first_neighbor_distance_dist_sq.csv", dist_sq);
  }

  {
    std::cout << "ps::distance_to_boundary...\n";

    auto               points = ps::random<float, dim>(50, ranges, seed);
    std::vector<float> dist = ps::distance_to_boundary(points, ranges);

    ps::save_points_to_csv("metrics_distance_to_boundary.csv", points);
    ps::save_vector_to_csv("metrics_distance_to_boundary_dist.csv", dist);
  }

  {
    std::cout << "ps::nearest_neighbors_indices...\n";

    auto         points = ps::random<float, dim>(50, ranges, seed);
    const size_t k_neighbors = 5;

    std::vector<std::vector<size_t>> idx = ps::nearest_neighbors_indices(points,
                                                                         k_neighbors);

    ps::save_points_to_csv("metrics_nearest_neighbors_indices.csv", points);

    // trick to save the indices as csv
    std::vector<ps::Point<size_t, k_neighbors>> p_dummy;
    for (auto &idx_vector : idx)
      p_dummy.push_back(ps::Point<size_t, k_neighbors>(idx_vector));
    ps::save_points_to_csv("metrics_nearest_neighbors_indices_idx.csv", p_dummy);
  }

  {
    std::cout << "ps::dbscan_clustering...\n";

    // random points
    auto points = ps::latin_hypercube_sampling<float, dim>(count, ranges, seed);

    float            connection_radius = 0.1f;
    size_t           min_pts = 5;
    std::vector<int> labels = ps::dbscan_clustering<float, dim>(points,
                                                                connection_radius,
                                                                min_pts);

    ps::save_points_to_csv("metrics_dbscan_clustering.csv", points);
    ps::save_vector_to_csv("metrics_dbscan_clustering_labels.csv", labels);
  }

  {
    std::cout << "ps::percolation_clustering...\n";

    // random points
    auto points = ps::latin_hypercube_sampling<float, dim>(count, ranges, seed);

    float            connection_radius = 0.1f;
    std::vector<int> labels = ps::percolation_clustering<float, dim>(points,
                                                                     connection_radius);

    ps::save_points_to_csv("metrics_percolation_clustering.csv", points);
    ps::save_vector_to_csv("metrics_percolation_clustering_labels.csv", labels);
  }

  {
    std::cout << "ps::kmeans_clustering...\n";

    // random points
    auto points = ps::latin_hypercube_sampling<float, dim>(count, ranges, seed);
    ps::save_points_to_csv("metrics_kmeans_clustering.csv", points);

    size_t k_clusters = 3;

    // build clustering on min and max distance to nearest neighbors
    {
      size_t                           k_neighbors = 4;
      std::vector<std::vector<size_t>> idx = ps::nearest_neighbors_indices(points,
                                                                           k_neighbors);

      // build a cluster using the minimum distance and the average
      // distance between the neighbors to have an idea of the
      // compacteness at the point and around the others points
      std::vector<float> dist_min;
      std::vector<float> dist_avg;
      dist_min.reserve(idx.size());
      dist_avg.reserve(idx.size());

      for (size_t k = 0; k < idx.size(); ++k)
      {
        float dmin = 1e9f;
        float davg = 0.f;

        // loop over neighbors
        for (size_t r = 0; r < k_neighbors; ++r)
        {
          float dist = ps::distance_squared(points[k], points[idx[k][r]]);
          dmin = std::min(dist, dmin);
          davg += dist;
        }

        dist_min.push_back(dmin);
        dist_avg.push_back(davg); // will be normalized after in kmeans
      }

      std::vector<ps::Point<float, 2>> data = ps::merge_by_dimension<float, 2>(
          {dist_min, dist_avg});

      // 3 clusters: (1) densely packed points with close neigbors,
      // (2) partially dense packed points with some neighbors further
      // away, lonely points

      auto km = ps::kmeans_clustering(data, k_clusters);

      ps::save_points_to_csv("metrics_kmeans_clustering_centroids.csv", km.first);
      ps::save_vector_to_csv("metrics_kmeans_clustering_labels.csv", km.second);
    }
  }

  {
    std::cout << "ps::radial_distribution...\n";

    auto points = ps::random<float, dim>(10 * count, ranges, seed);

    float bin_width = 0.005f;
    float max_distance = 0.5f;

    auto g = ps::radial_distribution<float, dim>(points, ranges, bin_width, max_distance);

    ps::save_points_to_csv("metrics_radial_distribution.csv", points);
    ps::save_vector_to_csv("metrics_radial_distribution_r.csv", g.first);
    ps::save_vector_to_csv("metrics_radial_distribution_pdf.csv", g.second);
  }

  {
    std::cout << "ps::angle_distribution_neighbors...\n";

    auto points = ps::random<float, dim>(10 * count, ranges, seed);

    float bin_width = 3.1416f / 32.f; // rads

    // /!\ O(N^3)... can be pretty slow
    auto g = ps::angle_distribution_neighbors<float, dim>(points, bin_width);

    ps::save_points_to_csv("metrics_angle_distribution_neighbors.csv", points);
    ps::save_vector_to_csv("metrics_angle_distribution_neighbors_alpha.csv", g.first);
    ps::save_vector_to_csv("metrics_angle_distribution_neighbors_pdf.csv", g.second);
  }

  {
    std::cout << "ps::local_density_knn...\n";

    auto points = ps::random<float, dim>(count, ranges, seed);
    auto d = ps::local_density_knn<float, dim>(points);

    ps::save_points_to_csv("metrics_local_density_knn.csv", points);
    ps::save_vector_to_csv("metrics_local_density_knn_d.csv", d);
  }

  return 0;
}
