#pragma once

#include <vector>

#include <Eigen/Dense>

#include <convex_plane_decomposition/PlanarRegion.h>

#include "superpixels/utils.h"

/*
*   A region's footprint and foothold region, from its points in the region plane.
*
*   The points are rasterized at footprint_resolution_ and closed over gaps up to
*   footprint_close_radius_ (the floor image is sparse). The largest connected part, with its
*   holes, becomes the boundary: non-convex, unlike the convex hull, so it does not claim the space
*   between the arms of an L-shaped region. The insets are the footprint eroded by foot_radius_, as
*   in CPD: possibly several non-convex parts with holes, largest first. Consumers grow convex
*   foothold regions inside them (OCS2 does so around each foothold).
*
*   Returns false (and leaves the output alone) if no foot fits: the eroded footprint is empty.
*/
bool regionFootprint(const std::vector<Eigen::Vector2d> & points,
                     const SuperpixelParams & params,
                     convex_plane_decomposition::BoundaryWithInset & boundaryWithInset);
