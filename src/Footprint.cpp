#include "superpixels/Footprint.h"

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

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

    // Simplified to within one cell, coarser if that is not a simple polygon (contours of large
    // sparse masks can touch themselves); empty if no simplification gives one.
    cpd::CgalPolygon2d toPolygon(const std::vector<cv::Point> & contour, const Grid & grid)
    {
        for (double epsilon : {1.0, 2.0, 3.0})
        {
            std::vector<cv::Point> simplified;
            cv::approxPolyDP(contour, simplified, epsilon, true);
            cpd::CgalPolygon2d polygon;
            for (const auto & cell : simplified)
                polygon.push_back(grid.toPlane(cell));
            if (polygon.size() >= 3 && polygon.is_simple())
                return polygon;
        }
        return cpd::CgalPolygon2d();
    }

    struct Contours
    {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;  // next, previous, first child (hole), parent

        explicit Contours(const cv::Mat & mask)
        {
            cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        }
    };

    // One outer contour and its holes as a polygon with holes (outer counterclockwise, holes
    // clockwise). Empty if the contour does not make a simple polygon.
    cpd::CgalPolygonWithHoles2d part(const Contours & c, int index, const Grid & grid)
    {
        cpd::CgalPolygonWithHoles2d shape;
        cpd::CgalPolygon2d outer = toPolygon(c.contours[index], grid);
        if (outer.size() < 3 || !outer.is_simple())
            return shape;
        if (outer.is_clockwise_oriented())
            outer.reverse_orientation();
        shape = cpd::CgalPolygonWithHoles2d(outer);
        for (int child = c.hierarchy[index][2]; child >= 0; child = c.hierarchy[child][0])
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

    // Every outer contour with an area, largest first.
    std::vector<cpd::CgalPolygonWithHoles2d> allParts(const Contours & c, const Grid & grid)
    {
        std::vector<std::pair<double, cpd::CgalPolygonWithHoles2d>> parts;
        for (int i = 0; i < (int) c.contours.size(); i++)
        {
            if (c.hierarchy[i][3] >= 0)
                continue;
            cpd::CgalPolygonWithHoles2d shape = part(c, i, grid);
            const double area = shape.outer_boundary().is_empty() ? 0.0 : shape.outer_boundary().area();
            if (area >= grid.resolution * grid.resolution)
                parts.emplace_back(area, std::move(shape));
        }
        std::sort(parts.begin(), parts.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        std::vector<cpd::CgalPolygonWithHoles2d> shapes;
        for (auto & p : parts)
            shapes.push_back(std::move(p.second));
        return shapes;
    }
}

std::vector<cpd::BoundaryWithInset> regionFootprints(const std::vector<Eigen::Vector2d> & points,
                                                     const SuperpixelParams & params)
{
    std::vector<cpd::BoundaryWithInset> footprints;
    if (points.size() < 3)
        return footprints;

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
    // Open with the smallest kernel: removes one-cell necks and diagonal-only links, which make
    // contours touch themselves (not simple polygons).
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // One footprint per connected part (closing can leave several, especially for merged regions).
    // Insets, as in CPD: the part eroded by the foot radius, possibly several pieces with holes.
    const Contours parts(mask);
    const cv::Mat footKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * footCells + 1, 2 * footCells + 1));
    for (int i = 0; i < (int) parts.contours.size(); i++)
    {
        if (parts.hierarchy[i][3] >= 0)
            continue;  // a hole
        cpd::CgalPolygonWithHoles2d boundary = part(parts, i, grid);
        if (boundary.outer_boundary().is_empty())
            continue;
        cv::Mat filled(rows, cols, CV_8UC1, cv::Scalar(0));
        cv::drawContours(filled, parts.contours, i, cv::Scalar(255), cv::FILLED);
        filled &= mask;  // drawing fills the holes in; cut them out again
        cv::Mat eroded;
        cv::erode(filled, eroded, footKernel);
        std::vector<cpd::CgalPolygonWithHoles2d> insets = allParts(Contours(eroded), grid);
        if (insets.empty())
            continue;  // no foot fits
        cpd::BoundaryWithInset footprint;
        footprint.boundary = std::move(boundary);
        footprint.insets = std::move(insets);
        footprints.push_back(std::move(footprint));
    }
    return footprints;
}
