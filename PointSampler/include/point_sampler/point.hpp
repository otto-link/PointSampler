/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ps
{

/**
 * @brief A fixed-size N-dimensional point/vector class.
 *
 * Represents a geometric or algebraic point in N-dimensional space with
 * coordinates of type T. Supports element-wise arithmetic operations, scalar
 * math, and basic geometric functions such as dot product, length,
 * normalization, and distance.
 *
 * Designed for use in procedural generation, geometry processing, simulation,
 * and graphics.
 *
 * - Compatible with STL containers and algorithms
 * - Zero-allocation, fast and cache-friendly (uses std::array<T, N>)
 * - Works seamlessly in 2D, 3D, or arbitrary dimensions
 *
 * @par Example:
 * @code {.cpp}
 * Point<float, 2> p{1.0f, 2.0f};
 * auto length = p.x() * p.x() + p.y() * p.y();
 * @endcode
 */

template <typename T, size_t N> struct Point
{
  std::array<T, N> coords;

  // Default constructor
  Point() = default;

  // Constructor from std::vector<T>
  explicit Point(const std::vector<T> &v)
  {
    if (v.size() != N)
      throw std::invalid_argument("Point: vector size mismatch");
    for (size_t i = 0; i < N; ++i)
      coords[i] = v[i];
  }

  // Constructor from initializer list
  Point(std::initializer_list<T> init)
  {
    if (init.size() != N)
      throw std::invalid_argument("Point: initializer list size mismatch");
    std::copy(init.begin(), init.end(), coords.begin());
  }

  Point(const std::array<T, N> &coords_) { coords = coords_; }

  // Accessors
  T       &operator[](size_t i) { return coords[i]; }
  const T &operator[](size_t i) const { return coords[i]; }

  // Optional 2D/3D accessors
  T &x()
  {
    static_assert(N > 0, "No x");
    return coords[0];
  }
  T &y()
  {
    static_assert(N > 1, "No y");
    return coords[1];
  }
  T &z()
  {
    static_assert(N > 2, "No z");
    return coords[2];
  }
  T &w()
  {
    static_assert(N > 3, "No w");
    return coords[3];
  }

  const T &x() const
  {
    static_assert(N > 0, "No x");
    return coords[0];
  }
  const T &y() const
  {
    static_assert(N > 1, "No y");
    return coords[1];
  }
  const T &z() const
  {
    static_assert(N > 2, "No z");
    return coords[2];
  }
  const T &w() const
  {
    static_assert(N > 3, "No w");
    return coords[3];
  }
};

// ------------------------------
// Arithmetic Operators
// ------------------------------
template <typename T, size_t N>
Point<T, N> operator+(const Point<T, N> &a, const Point<T, N> &b)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = a[i] + b[i];
  return result;
}

template <typename T, size_t N>
Point<T, N> operator-(const Point<T, N> &a, const Point<T, N> &b)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = a[i] - b[i];
  return result;
}

template <typename T, size_t N>
Point<T, N> operator*(const Point<T, N> &a, const Point<T, N> &b)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = a[i] * b[i];
  return result;
}

template <typename T, size_t N>
Point<T, N> operator/(const Point<T, N> &a, const Point<T, N> &b)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = a[i] / b[i];
  return result;
}

// Scalar ops
template <typename T, size_t N> Point<T, N> operator+(const Point<T, N> &p, T scalar)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = p[i] + scalar;
  return result;
}

template <typename T, size_t N> Point<T, N> operator-(const Point<T, N> &p, T scalar)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = p[i] - scalar;
  return result;
}

template <typename T, size_t N> Point<T, N> operator*(const Point<T, N> &p, T scalar)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = p[i] * scalar;
  return result;
}

template <typename T, size_t N> Point<T, N> operator/(const Point<T, N> &p, T scalar)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = p[i] / scalar;
  return result;
}

// Reverse scalar ops
template <typename T, size_t N> Point<T, N> operator*(T scalar, const Point<T, N> &p)
{
  return p * scalar;
}

template <typename T, size_t N> Point<T, N> operator+(T scalar, const Point<T, N> &p)
{
  return p + scalar;
}

// ------------------------------
// Geometric Functions
// ------------------------------
template <typename T, size_t N> T dot(const Point<T, N> &a, const Point<T, N> &b)
{
  T result = T(0);
  for (size_t i = 0; i < N; ++i)
    result += a[i] * b[i];
  return result;
}

template <typename T, size_t N> T length_squared(const Point<T, N> &a)
{
  return dot(a, a);
}

template <typename T, size_t N> T length(const Point<T, N> &a)
{
  return std::sqrt(length_squared(a));
}

template <typename T, size_t N> Point<T, N> normalized(const Point<T, N> &a)
{
  T len = length(a);
  if (len == T(0))
    return Point<T, N>();
  else
    return a / len;
}

template <typename T, size_t N>
T distance_squared(const Point<T, N> &a, const Point<T, N> &b)
{
  return length_squared(a - b);
}

template <typename T, size_t N> T distance(const Point<T, N> &a, const Point<T, N> &b)
{
  return length(a - b);
}

template <typename T, size_t N>
Point<T, N> lerp(const Point<T, N> &a, const Point<T, N> &b, T t)
{
  return a + (b - a) * t;
}

template <typename T, size_t N>
Point<T, N> clamp(const Point<T, N> &p, T min_val, T max_val)
{
  Point<T, N> result;
  for (size_t i = 0; i < N; ++i)
    result[i] = std::min(std::max(p[i], min_val), max_val);
  return result;
}

} // namespace ps