#include "superpixels/Footprint.h"

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <convex_plane_decomposition/ConvexRegionGrowing.h>

namespace cpd = convex_plane_decomposition;

namespace
{
    // Grid geometry: cell (row, col) has its center at origin + (col + 0.5, row + 0.5) * resolution.
    struct Grid
    {
        Eigen::Vector2d origin;
        double resolution;

        cpd::CgalPoint2d toPlane(const cv::Point & cell) const
        {
            return cpd::CgalPoint2d(origin.x() + (cell.x + 0.5) * resolution, origin.y() + (cell.y + 0.5) * resolution);
        }
    };

    cpd::CgalPolygon2d toPolygon(const std::vector<cv::Point> & contour, const Grid & grid)
    {
        // Simplify to within one cell; keep the raw contour if simplifying breaks it.
        std::vector<cv::Point> simplified;
        cv::approxPolyDP(contour, simplified, 1.0, true);
        cpd::CgalPolygon2d polygon;
        for (const auto & cell : simplified)
            polygon.push_back(grid.toPlane(cell));
        if (polygon.size() < 3 || !polygon.is_simple())
        {
            polygon.clear();
            for (const auto & cell : contour)
                polygon.push_back(grid.toPlane(cell));
        }
        return polygon;
    }

    struct Contours
    {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;  // next, previous, first child (hole), parent
        int largest = -1;                  // outer contour with the largest area, -1 if none

        explicit Contours(const cv::Mat & mask)
        {
            cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
            double largestArea = 0.0;
            for (int i = 0; i < (int) contours.size(); i++)
            {
                const double area = cv::contourArea(contours[i]);
                if (hierarchy[i][3] < 0 && area > largestArea)
                {
                    largest = i;
                    largestArea = area;
                }
            }
        }
    };

    // The largest outer contour and its holes as a polygon with holes (outer counterclockwise,
    // holes clockwise). Empty if the mask has no part with an area.
    cpd::CgalPolygonWithHoles2d largestPart(const Contours & c, const Grid & grid)
    {
        cpd::CgalPolygonWithHoles2d shape;
        if (c.largest < 0)
            return shape;
        cpd::CgalPolygon2d outer = toPolygon(c.contours[c.largest], grid);
        if (outer.size() < 3 || !outer.is_simple())
            return shape;
        if (outer.is_clockwise_oriented())
            outer.reverse_orientation();
        shape = cpd::CgalPolygonWithHoles2d(outer);
        for (int child = c.hierarchy[c.largest][2]; child >= 0; child = c.hierarchy[child][0])
        {
            cpd::CgalPolygon2d hole = toPolygon(c.contours[child], grid);
            if (hole.size() < 3 || !hole.is_simple() || std::abs(hole.area()) < grid.resolution * grid.resolution)
                continue;
            if (hole.is_counterclockwise_oriented())
                hole.reverse_orientation();
            shape.add_hole(hole);
        }
        return shape;
    }

    // The mask restricted to the connected part containing the given cell.
    cv::Mat partContaining(const cv::Mat & mask, const cv::Point & cell)
    {
        cv::Mat labels;
        cv::connectedComponents(mask, labels, 8, CV_32S);
        return labels == labels.at<int>(cell);
    }
}

bool regionFootprint(const std::vector<Eigen::Vector2d> & points,
                     const SuperpixelParams & params,
                     cpd::BoundaryWithInset & boundaryWithInset)
{
    if (points.size() < 3)
        return false;

    const double res = params.footprint_resolution_;
    const int closeCells = std::max(1, (int) std::ceil(params.footprint_close_radius_ / res));
    const int footCells = std::max(1, (int) std::ceil(params.foot_radius_ / res));

    // Grid over the points, padded so closing and erosion never touch its border.
    Eigen::Vector2d lo = points.front(), hi = points.front();
    for (const auto & p : points)
    {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const int pad = closeCells + 2;
    const Grid grid{lo - Eigen::Vector2d::Constant(pad * res), res};
    const int cols = (int) std::ceil((hi.x() - lo.x()) / res) + 2 * pad + 1;
    const int rows = (int) std::ceil((hi.y() - lo.y()) / res) + 2 * pad + 1;

    cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
    for (const auto & p : points)
    {
        const int col = std::clamp((int) std::floor((p.x() - grid.origin.x()) / res), 0, cols - 1);
        const int row = std::clamp((int) std::floor((p.y() - grid.origin.y()) / res), 0, rows - 1);
        mask.at<uint8_t>(row, col) = 255;
    }
    const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * closeCells + 1, 2 * closeCells + 1));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeKernel);

    const Contours parts(mask);
    const cpd::CgalPolygonWithHoles2d boundary = largestPart(parts, grid);
    if (boundary.outer_boundary().is_empty())
        return false;

    // Foot-center region: the boundary's part (closing can leave small separate ones) eroded by the
    // foot radius.
    cv::Mat filled(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::drawContours(filled, parts.contours, parts.largest, cv::Scalar(255), cv::FILLED);
    filled &= mask;  // drawing fills the holes in; cut them out again
    cv::Mat eroded;
    const cv::Mat footKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * footCells + 1, 2 * footCells + 1));
    cv::erode(filled, eroded, footKernel);
    if (cv::countNonZero(eroded) == 0)
        return false;

    // Seed at the point farthest from the eroded edge, and grow inside the part that contains it.
    cv::Mat distance;
    cv::distanceTransform(eroded, distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Point seed;
    cv::minMaxLoc(distance, nullptr, nullptr, nullptr, &seed);
    const cpd::CgalPolygonWithHoles2d footArea = largestPart(Contours(partContaining(eroded, seed)), grid);
    if (footArea.outer_boundary().is_empty())
        return false;
    cpd::CgalPolygon2d inset = cpd::growConvexPolygonInsideShape(footArea, grid.toPlane(seed),
                                                                 params.inset_vertices_, params.inset_growth_factor_);
    if (inset.size() < 3 || std::abs(inset.area()) < res * res)
        return false;
    // Region growing occasionally runs away (vertices at 1e17 m); keep only insets inside the shape.
    for (const auto & vertex : inset.vertices())
    {
        if (footArea.outer_boundary().bounded_side(vertex) == CGAL::ON_UNBOUNDED_SIDE)
            return false;
    }

    boundaryWithInset.boundary = boundary;
    boundaryWithInset.insets = {cpd::CgalPolygonWithHoles2d(inset)};
    return true;
}
