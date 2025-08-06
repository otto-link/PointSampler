/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "point_sampler/point.hpp"

namespace ps
{

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

// Ex.:
//   auto separated = split_by_dimension(pts);
//   separated[0] = {1.0f, 4.0f, 7.0f} // x values
//   separated[1] = {2.0f, 5.0f, 8.0f} // y values
//   separated[2] = {3.0f, 6.0f, 9.0f} // z values
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