/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <optional>

#include "point_sampler/point.hpp"

namespace ps
{

/** \brief Generate random walk filaments in N dimensions with optional Gaussian
   thickness.
 *
 * Each filament starts at a random seed and grows step by step, where each step
 * is a random direction with a persistence factor to avoid sharp turns. Around
 * each step, additional points can be sampled from a Gaussian distribution to
 * form a "thick" filament.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension.
 * @param n_filaments Number of separate filaments.
 * @param filament_count Number of points per filament.
 * @param step_size Average step length.
 * @param ranges Bounding box for clamping.
 * @param seed Optional RNG seed.
 * @param persistence Correlation between steps (0 = totally random, 1 =
 * straight line).
 * @param gaussian_sigma Standard deviation of Gaussian scatter around the
 * filament (0 = no scatter, >0 = thick filament).
 * @param gaussian_samples Number of samples drawn per step for thickness.
 * @param p_distances Optional output vector to store p_distances of each point
 * from the filament center (0 for core filament points).
 * @return A vector of generated filament points.
 *
 * @image html out_random_rejection_filter.csv.jpg
 */
template <typename T, size_t N>
std::vector<Point<T, N>> random_walk_filaments(
    size_t                                n_filaments,
    size_t                                filament_count,
    T                                     step_size,
    const std::array<std::pair<T, T>, N> &ranges,
    std::optional<unsigned int>           seed = std::nullopt,
    T                                     persistence = T(0.8),
    T                                     gaussian_sigma = T(0),
    size_t                                gaussian_samples = 0,
    std::vector<T>                       *p_distances = nullptr) // optional
// output
{
  std::vector<Point<T, N>> points;
  points.reserve(n_filaments * filament_count * (1 + gaussian_samples));
  if (p_distances)
    p_distances->reserve(n_filaments * filament_count * (1 + gaussian_samples));

  std::mt19937                      gen(seed ? *seed : std::random_device{}());
  std::uniform_real_distribution<T> uniform(-1, 1);
  std::normal_distribution<T>       normal(0, gaussian_sigma);

  for (size_t f = 0; f < n_filaments; ++f)
  {
    // Random starting point
    Point<T, N> p;
    for (size_t d = 0; d < N; ++d)
      p[d] = std::uniform_real_distribution<T>(ranges[d].first, ranges[d].second)(gen);

    // Random initial direction (normalized)
    std::array<T, N> dir;
    T                norm = 0;
    for (size_t d = 0; d < N; ++d)
    {
      dir[d] = uniform(gen);
      norm += dir[d] * dir[d];
    }
    norm = std::sqrt(norm);
    for (size_t d = 0; d < N; ++d)
      dir[d] /= norm;

    for (size_t i = 0; i < filament_count; ++i)
    {
      // Always add core filament point
      points.push_back(p);
      if (p_distances)
        p_distances->push_back(T(0));

      // Add Gaussian scatter points (thickness)
      for (size_t g = 0; g < gaussian_samples; ++g)
      {
        Point<T, N> q = p;
        T           dist2 = 0;
        for (size_t d = 0; d < N; ++d)
        {
          T offset = normal(gen);
          q[d] += offset;
          dist2 += offset * offset;
        }

        // Discard points outside the bounding box
        bool inside = true;
        for (size_t d = 0; d < N; ++d)
        {
          if (q[d] < ranges[d].first || q[d] > ranges[d].second)
          {
            inside = false;
            break;
          }
        }

        if (inside)
        {
          points.push_back(q);
          if (p_distances)
            p_distances->push_back(std::sqrt(dist2));
        }
      }

      // Random perturbation
      std::array<T, N> rnd;
      T                rnd_norm = 0;
      for (size_t d = 0; d < N; ++d)
      {
        rnd[d] = uniform(gen);
        rnd_norm += rnd[d] * rnd[d];
      }
      rnd_norm = std::sqrt(rnd_norm);
      for (size_t d = 0; d < N; ++d)
        rnd[d] /= rnd_norm;

      // Blend with persistence
      for (size_t d = 0; d < N; ++d)
        dir[d] = persistence * dir[d] + (1 - persistence) * rnd[d];

      // Normalize direction
      T dnorm = 0;
      for (size_t d = 0; d < N; ++d)
        dnorm += dir[d] * dir[d];
      dnorm = std::sqrt(dnorm);
      for (size_t d = 0; d < N; ++d)
        dir[d] /= dnorm;

      // Step forward
      for (size_t d = 0; d < N; ++d)
        p[d] += step_size * dir[d];
    }
  }

  return points;
}

} // namespace ps