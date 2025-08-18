/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Generates samples using Latin Hypercube Sampling (LHS).
 *
 * This function produces `sample_count` evenly stratified samples across each
 * dimension defined in `axis_ranges`. Each dimension is divided into equal
 * intervals (strata), and one point is randomly selected from each stratum with
 * added jitter. The strata are randomly permuted across dimensions to avoid
 * correlation.
 *
 * This method ensures uniform coverage of the space while maintaining
 * randomness, making it useful in Monte Carlo integration, surrogate modeling,
 * and probabilistic sampling.
 *
 * @tparam T Scalar type (e.g., float or double)
 * @tparam N Dimensionality of the space
 *
 * @param  sample_count Number of points to generate
 * @param  axis_ranges  Array of N (min, max) pairs defining the domain in each
 *                      dimension
 * @param  seed         Optional seed for reproducible randomness
 *
 * @return              std::vector<Point<T, N>> of LHS-sampled points
 *
 * @par Example
 * @code {.cpp}
 * std::array<std::pair<float, float>, 2> range = {{{0.0f, 1.0f}, {0.0f, 1.0f}}};
 * std::vector<Point<float, 2>> points = latin_hypercube_sampling<float, 2>(1000, range);
 * @endcode
 *
 * @image html out_latin_hypercube_sampling.jpg
 */
template <typename T, std::size_t N>
std::vector<Point<T, N>> latin_hypercube_sampling(
    std::size_t                           sample_count,
    const std::array<std::pair<T, T>, N> &axis_ranges,
    std::optional<unsigned int>           seed = std::nullopt)
{
  std::mt19937                      rng(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> jitter(0.0, 1.0);

  std::vector<Point<T, N>> samples(sample_count);

  for (std::size_t dim = 0; dim < N; ++dim)
  {
    std::vector<T> strata(sample_count);
    T              range_min = axis_ranges[dim].first;
    T              range_max = axis_ranges[dim].second;
    T              stride = (range_max - range_min) / sample_count;

    // Generate stratified positions with jitter
    for (std::size_t i = 0; i < sample_count; ++i)
      strata[i] = range_min + (i + jitter(rng)) * stride;

    std::shuffle(strata.begin(), strata.end(), rng);

    // Assign values to the sample vector
    for (std::size_t i = 0; i < sample_count; ++i)
      samples[i][dim] = strata[i];
  }

  return samples;
}

} // namespace ps