#pragma once

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "superpixels/utils.h"

/*
*   Merging coplanar neighboring superpixels into one PlanarTerrain region (merge_coplanar_regions_).
*
*   Each superpixel is its own small plane (about 10 cm), so eroding its footprint by the foot
*   radius leaves little: CPD erodes whole planes instead. Groups grow greedily from the largest
*   superpixel: a neighbor (hulls within merge_max_gap_ of each other) joins if its normal is within
*   merge_max_angle_ of the group plane's and its centroid within merge_max_offset_ of that plane.
*   Comparing with the group plane, not the neighbor, keeps a gently curving surface from chaining
*   into one region.
*/
struct PlaneRegion
{
    Eigen::Isometry3d planeToWorld;          // plane frame: origin on the plane, z along the normal
    std::vector<Eigen::Vector2d> points;     // the superpixel's points, plane coordinates
    std::vector<Eigen::Vector2d> hull;       // convex hull, plane coordinates
    int count = 0;                           // pixels, the weight in the group plane
};

// Indices of the regions in each group; every region is in exactly one group.
std::vector<std::vector<int>> coplanarGroups(const std::vector<PlaneRegion> & regions, const SuperpixelParams & params);

// The group's plane (count-weighted centroid and normal) and all its points in that plane.
void mergeGroup(const std::vector<PlaneRegion> & regions, const std::vector<int> & group,
                Eigen::Isometry3d & planeToWorld, std::vector<Eigen::Vector2d> & points);
