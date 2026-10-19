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
*   between the arms of an L-shaped region. The inset is the convex polygon CPD grows inside the
*   footprint eroded by foot_radius_, seeded at the point farthest from the eroded edge.
*
*   Returns false (and leaves the output alone) if no foot fits: no inset can be grown.
*/
bool regionFootprint(const std::vector<Eigen::Vector2d> & points,
                     const SuperpixelParams & params,
                     convex_plane_decomposition::BoundaryWithInset & boundaryWithInset);
