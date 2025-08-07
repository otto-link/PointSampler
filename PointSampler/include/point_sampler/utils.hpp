/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "point_sampler/point.hpp"

namespace ps
{

/**
 * @brief Save a set of N-dimensional points to a CSV file.
 *
 * The output file will contain one point per line, with each coordinate
 * separated by commas. Optionally, a header row ("x0,x1,...,xN") can be written
 * as the first line.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of each point.
 *
 * @param  filename     Path to the output CSV file.
 * @param  points       Vector of points to be saved.
 * @param  write_header If true, writes a header row with column names
 *                      ("x0,x1,...").
 *
 * @return              true if the file was successfully written, false
 *                      otherwise.
 */
template <typename T, size_t N>
bool save_points_to_csv(const std::string              &filename,
                        const std::vector<Point<T, N>> &points,
                        bool                            write_header = true)
{
  std::ofstream out(filename);
  if (!out.is_open())
    return false;

  if (write_header)
  {
    for (size_t i = 0; i < N; ++i)
    {
      out << "x" << i;
      if (i < N - 1)
        out << ",";
    }
    out << "\n";
  }

  for (const auto &point : points)
  {
    for (size_t i = 0; i < N; ++i)
    {
      out << point[i];
      if (i < N - 1)
        out << ",";
    }
    out << "\n";
  }

  return true;
}

/**
 * @brief Rearranges a list of N-dimensional points into N separate coordinate
 * vectors.
 *
 * This function decomposes a vector of N-dimensional points into N vectors,
 * where each vector contains all the values from one coordinate dimension.
 * Useful for plotting or statistical analysis.
 *
 * For example, given 3D points: [(1,2,3), (4,5,6), (7,8,9)], the result will
 * be:
 *   - dimension 0: [1, 4, 7]
 *   - dimension 1: [2, 5, 8]
 *   - dimension 2: [3, 6, 9]
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of each point.
 *
 * @param  points Vector of N-dimensional points.
 *
 * @return        An array of N vectors, each containing the values for one
 *                coordinate axis.
 */
template <typename T, size_t N>
std::array<std::vector<T>, N> split_by_dimension(const std::vector<Point<T, N>> &points)
{
  std::array<std::vector<T>, N> components;

  for (const auto &point : points)
    for (size_t i = 0; i < N; ++i)
    {
      components[i].push_back(point[i]);
    }

  return components;
}

} // namespace ps
