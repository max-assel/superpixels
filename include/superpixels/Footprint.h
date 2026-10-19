#pragma once

#include <vector>

#include <Eigen/Dense>

#include <convex_plane_decomposition/PlanarRegion.h>

#include "superpixels/utils.h"

/*
*   A region's footprints and foothold regions, from its points in the region plane.
*
*   The points are rasterized at footprint_resolution_ and closed over gaps up to
*   footprint_close_radius_ (the floor image is sparse). Each connected part, with its holes,
*   becomes one footprint boundary: non-convex, unlike the convex hull, so it does not claim the
*   space between the arms of an L-shaped region. Its insets are the part eroded by foot_radius_,
*   as in CPD: possibly several non-convex pieces with holes, largest first. Consumers grow convex
*   foothold regions inside them (OCS2 does so around each foothold).
*
*   Parts where no foot fits (the eroded part is empty) are left out.
*/
std::vector<convex_plane_decomposition::BoundaryWithInset> regionFootprints(const std::vector<Eigen::Vector2d> & points,
                                                                            const SuperpixelParams & params);
