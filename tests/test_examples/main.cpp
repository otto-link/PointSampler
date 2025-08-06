/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <iostream>

#include "point_sampler.hpp"

int main()
{
  // generate 5 random pts in 3D in the cube [-1, 1] x [-2, 2] x
  // [0, 1], with float precision
  const size_t dim = 3;
  size_t       count = 5;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {std::make_pair(-1.f, 1.f),
                                                     std::make_pair(-2.f, 2.f),
                                                     std::make_pair(0.f, 1.f)};

  std::vector<ps::Point<float, dim>> points = ps::random<float, dim>(count, ranges, seed);

  // display x, y, z
  for (const auto &point : points)
  {
    std::cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")\n";
  }

  return 0;
}
