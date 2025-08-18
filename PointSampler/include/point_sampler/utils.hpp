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
 * @brief Save a 1D vector of values to a CSV file.
 *
 * This function writes a sequence of values to a CSV file with one value per
 * row. The column name can be customized via the @p header_name parameter.
 *
 * @tparam T Type of the values (must be streamable to std::ostream).
 * @param  filename     Path to the output CSV file.
 * @param  values       The vector of values to write.
 * @param  write_header If true, writes a header line at the top of the file.
 * @param  header_name  Name of the column header (only used if @p write_header
 *                      is true).
 * @return              True if the file was successfully written, false
 *                      otherwise.
 *
 * @note The file will be overwritten if it already exists.
 *
 * @par Example
 * @code
 * std::vector<double> data = {1.0, 2.5, 3.7};
 * save_vector_to_csv("data.csv", data, true, "measurement");
 * // data.csv content:
 * // measurement
 * // 1.0
 * // 2.5
 * // 3.7
 * @endcode
 */
template <typename T>
bool save_vector_to_csv(const std::string    &filename,
                        const std::vector<T> &values,
                        bool                  write_header = true,
                        const std::string    &header_name = "value")
{
  std::ofstream out(filename);
  if (!out.is_open())
    return false;

  if (write_header)
    out << header_name << "\n";

  for (const auto &val : values)
    out << val << "\n";

  return true;
}

/**
 * @brief Add a new dimension to a set of points.
 *
 * This function takes a vector of points of dimension N and appends a new
 * coordinate to each point, producing a new vector of points of dimension N+1.
 *
 * @tparam T Numeric type of the coordinates (e.g., float, double, int).
 * @tparam N Current number of dimensions in the input points.
 * @param  points        Vector of input points of dimension N.
 * @param  new_dimension Vector of values for the new dimension. Must have the
 *                       same size as `points`.
 * @return               A vector of points of dimension N+1, with the new
 *                       dimension appended.
 *
 * @throws std::runtime_errorifthesizeof`points`and`new_dimension`donotmatch.
 *
 * @note This function creates a new vector of points with an increased
 * dimension count, since the type Point<T, N> is distinct from Point<T, N+1>.
 *
 * @par Example
 * @code
 * std::vector<Point<float, 2>> points = { {{1.0f, 2.0f}}, {{3.0f, 4.0f}}};
 * std::vector<float> z_values = { 10.0f, 20.0f };
 * auto points3D = add_dimension(points, z_values);
 * // points3D now contains {{1.0f, 2.0f, 10.0f}}, {{3.0f, 4.0f, 20.0f}}
 * @endcode
 */
template <typename T, size_t N>
std::vector<Point<T, N + 1>> add_dimension(const std::vector<Point<T, N>> &points,
                                           const std::vector<T>           &new_dimension)
{
  if (points.size() != new_dimension.size())
    throw std::runtime_error(
        "add_dimension: size mismatch between points and new dimension data");

  std::vector<Point<T, N + 1>> result;
  result.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i)
  {
    std::array<T, N + 1> coords{};
    std::copy(points[i].coords.begin(), points[i].coords.end(), coords.begin());
    coords[N] = new_dimension[i];
    result.emplace_back(coords);
  }

  return result;
}

/**
 * @brief Extract clusters of points given DBSCAN (or any clustering) labels.
 *
 * @tparam T Scalar type.
 * @tparam N Dimension.
 * @param points Input point cloud.
 * @param labels Cluster labels (-2 = noise, -1 = unvisited, 0..k = cluster
 * IDs).
 * @return A vector of clusters, each cluster is a vector of points.
 *
 * @par Example
 * @code {.cpp}
 * auto labels   = dbscan<double,2>(points, 0.05, 5);
 * auto clusters = extract_clusters(points, labels);
 * @endcode
 */
template <typename T, size_t N>
std::vector<std::vector<Point<T, N>>> extract_clusters(
    const std::vector<Point<T, N>> &points,
    const std::vector<int>         &labels)
{
  if (points.size() != labels.size())
    throw std::runtime_error("extract_clusters: mismatch between points and labels size");

  // find max cluster id
  int max_cluster_id = -1;
  for (int lbl : labels)
    if (lbl >= 0)
      max_cluster_id = std::max(max_cluster_id, lbl);

  std::vector<std::vector<Point<T, N>>> clusters(max_cluster_id + 1);

  for (size_t i = 0; i < points.size(); ++i)
  {
    int lbl = labels[i];
    if (lbl >= 0)
      clusters[lbl].push_back(points[i]);
  }

  return clusters;
}

/**
 * @brief Reconstructs a list of N-dimensional points from N separate coordinate
 * vectors.
 *
 * This function takes N vectors—each representing one coordinate axis—and
 * combines them into a single vector of N-dimensional points. It is the inverse
 * operation of `split_by_dimension`.
 *
 * All coordinate vectors must have the same length.
 *
 * For example, given:
 *   - dimension 0: [1, 4, 7]
 *   - dimension 1: [2, 5, 8]
 *   - dimension 2: [3, 6, 9]
 *
 * The result will be: [(1,2,3), (4,5,6), (7,8,9)]
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam N Dimension of each point.
 *
 * @param  components An array of N vectors, each containing values for one
 *                    coordinate axis.
 *
 * @return            A vector of N-dimensional points reconstructed from the
 *                    coordinate vectors.
 *
 * @throws std::invalid_argumentifthecoordinatevectorsdonotallhavethesame
 *                    length.
 *
 * @par Example
 * @code {.cpp}
 * std::array<std::vector<float>, 3> components = {{
 *     {1.0f, 4.0f, 7.0f},  // x
 *     {2.0f, 5.0f, 8.0f},  // y
 *     {3.0f, 6.0f, 9.0f}   // z
 * }};
 * std::vector<Point<float, 3>> points = merge_by_dimension(components);
 * @endcode
 */
template <typename T, size_t N>
std::vector<Point<T, N>> merge_by_dimension(
    const std::array<std::vector<T>, N> &components)
{
  if constexpr (N > 0)
  {
    std::size_t count = components[0].size();
    for (std::size_t i = 1; i < N; ++i)
    {
      if (components[i].size() != count)
        throw std::invalid_argument("All component vectors must have the same size");
    }

    std::vector<Point<T, N>> points(count);
    for (std::size_t i = 0; i < count; ++i)
    {
      for (std::size_t j = 0; j < N; ++j)
        points[i][j] = components[j][i];
    }

    return points;
  }
  else
  {
    return {};
  }
}

/**
 * @brief Normalize the coordinates of a set of points along each axis to the
 * range [0, 1].
 *
 * This function finds the minimum and maximum value for each axis across all
 * points and rescales each coordinate so that the minimum becomes 0 and the
 * maximum becomes 1.
 *
 * @tparam T Numeric type of the coordinates (e.g., float, double).
 * @tparam N Number of dimensions in each point.
 * @param points Vector of points to normalize. The points are modified in
 *               place.
 *
 * @note If all points have the same value along a given axis, the corresponding
 * normalized coordinate will be set to 0 for that axis.
 *
 * @par Example
 * @code
 * std::vector<Point<float, 3>> points = {
 *     {{1.0f, 5.0f, 10.0f}},
 *     {{3.0f, 15.0f, 20.0f}}
 * };
 * normalize_points(points);
 * // Now points coordinates are scaled in [0, 1] along each axis
 * @endcode
 */
template <typename T, size_t N> void normalize_points(std::vector<Point<T, N>> &points)
{
  if (points.empty())
    return;

  std::array<T, N> min_vals;
  std::array<T, N> max_vals;

  // Initialize min/max arrays
  for (size_t dim = 0; dim < N; ++dim)
  {
    min_vals[dim] = points[0].coords[dim];
    max_vals[dim] = points[0].coords[dim];
  }

  // Find min/max for each axis
  for (const auto &p : points)
  {
    for (size_t dim = 0; dim < N; ++dim)
    {
      min_vals[dim] = std::min(min_vals[dim], p.coords[dim]);
      max_vals[dim] = std::max(max_vals[dim], p.coords[dim]);
    }
  }

  // Normalize each point
  for (auto &p : points)
  {
    for (size_t dim = 0; dim < N; ++dim)
    {
      T range = max_vals[dim] - min_vals[dim];
      if (range != T(0))
        p.coords[dim] = (p.coords[dim] - min_vals[dim]) / range;
      else
        p.coords[dim] = T(0); // Avoid NaN if all values
                              // are equal
    }
  }
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
