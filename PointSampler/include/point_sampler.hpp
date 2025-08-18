/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

/**
 * @mainpage PointSampler
 *
 * @section intro_sec Introduction
 *
 * **PointSampler** is a lightweight, header-only C++ library for point sampling
 * in arbitrary dimensions. It offers generic and reusable algorithms for:
 *
 * - Random point generation
 * - Distance-based filtering
 * - Utility operations on points
 *
 * The library is template-based and dimension-independent, making it suitable
 * for 2D, 3D, or higher-dimensional spaces.
 *
 * @section usage_sec Usage Example
 *
 * @code{.cpp}
 * #include <point_sampler.hpp>
 *
 * std::vector<Point<float, 2>> pts = ps::random<float, 2>(1000, {{0,1},{0,1}});
 * auto filtered = ps::distance_rejection_filter(pts, 0.05f);
 * @endcode
 *
 * @section modules_sec Categories
 *
 * - Random Sampling Functions
 *   - @ref random.hpp
 *   - @ref halton.hpp
 *   - @ref hammersley.hpp
 *   - @ref jittered_grid.hpp
 *   - @ref latin_hypercube_sampling.hpp
 *   - @ref poisson_disk_sampling.hpp
 *   - @ref gaussian_clusters.hpp
 *   - @ref importance_resampling.hpp
 *   - @ref rejection_sampling.hpp
 *   - @ref random_walk_filaments.hpp
 * - Filtering Functions
 *   - @ref distance_rejection_filter.hpp
 *   - @ref function_rejection_filter.hpp
 *   - @ref random_rejection_filter.hpp
 *   - @ref range.hpp
 *   - @ref relaxation.hpp
 * - Clustering
 *   - @ref kmeans_clustering.hpp
 *   - @ref dbscan_clustering.hpp
 *   - @ref percolation_clustering.hpp
 * - Data
 *   - @ref metrics.hpp
 * - Point representation
 *   - @ref point.hpp
 *
 * @section repo_sec Repository
 *
 * GitHub: https://github.com/otto-link/PointSampler
 */

#include "point_sampler/point.hpp"
#include "point_sampler/utils.hpp"

#include "point_sampler/dbscan_clustering.hpp"
#include "point_sampler/distance_rejection_filter.hpp"
#include "point_sampler/function_rejection_filter.hpp"
#include "point_sampler/gaussian_clusters.hpp"
#include "point_sampler/halton.hpp"
#include "point_sampler/hammersley.hpp"
#include "point_sampler/importance_resampling.hpp"
#include "point_sampler/jittered_grid.hpp"
#include "point_sampler/kmeans_clustering.hpp"
#include "point_sampler/latin_hypercube_sampling.hpp"
#include "point_sampler/metrics.hpp"
#include "point_sampler/percolation_clustering.hpp"
#include "point_sampler/poisson_disk_sampling.hpp"
#include "point_sampler/random.hpp"
#include "point_sampler/random_rejection_filter.hpp"
#include "point_sampler/random_walk_filaments.hpp"
#include "point_sampler/range.hpp"
#include "point_sampler/rejection_sampling.hpp"
#include "point_sampler/relaxation.hpp"
