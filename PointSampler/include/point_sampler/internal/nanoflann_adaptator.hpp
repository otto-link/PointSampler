/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once
#include <nanoflann.hpp>

#include "point_sampler/point.hpp"

namespace ps
{

template <typename T, size_t N> struct PointCloudAdaptor
{
  const std::vector<Point<T, N>> &pts;

  PointCloudAdaptor(const std::vector<Point<T, N>> &points) : pts(points) {}

  inline size_t kdtree_get_point_count() const { return pts.size(); }

  inline T kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    return pts[idx][dim];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX &) const
  {
    return false;
  } // Not using bounding boxes
};

template <typename T, size_t N>
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<T, PointCloudAdaptor<T, N>>,
    PointCloudAdaptor<T, N>,
    N>;

} // namespace ps