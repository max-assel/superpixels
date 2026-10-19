#include "superpixels/RegionMerging.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

namespace
{
    // Hull corners in the world xy plane, and their bounding box.
    struct WorldHull
    {
        std::vector<Eigen::Vector2d> corners;
        Eigen::Vector2d lo, hi;
    };

    WorldHull worldHull(const PlaneRegion & region)
    {
        WorldHull h;
        for (const auto & p : region.hull)
            h.corners.push_back((region.planeToWorld * Eigen::Vector3d(p.x(), p.y(), 0.0)).head<2>());
        h.lo = h.hi = h.corners.empty() ? Eigen::Vector2d::Zero() : h.corners.front();
        for (const auto & c : h.corners)
        {
            h.lo = h.lo.cwiseMin(c);
            h.hi = h.hi.cwiseMax(c);
        }
        return h;
    }

    double pointSegmentDistance(const Eigen::Vector2d & p, const Eigen::Vector2d & a, const Eigen::Vector2d & b)
    {
        const Eigen::Vector2d ab = b - a;
        const double t = ab.squaredNorm() > 0.0 ? std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.0, 1.0) : 0.0;
        return (a + t * ab - p).norm();
    }

    // Inside a convex polygon of either orientation.
    bool insideConvex(const Eigen::Vector2d & p, const std::vector<Eigen::Vector2d> & polygon)
    {
        int sign = 0;
        for (size_t i = 0; i < polygon.size(); i++)
        {
            const Eigen::Vector2d e = polygon[(i + 1) % polygon.size()] - polygon[i];
            const Eigen::Vector2d d = p - polygon[i];
            const double cross = e.x() * d.y() - e.y() * d.x();
            if (cross != 0.0)
            {
                const int s = cross > 0.0 ? 1 : -1;
                if (sign != 0 && s != sign)
                    return false;
                sign = s;
            }
        }
        return true;
    }

    // Distance between two convex hulls in xy; 0 if they overlap.
    double hullGap(const WorldHull & a, const WorldHull & b)
    {
        for (const auto & c : a.corners)
            if (insideConvex(c, b.corners))
                return 0.0;
        for (const auto & c : b.corners)
            if (insideConvex(c, a.corners))
                return 0.0;
        double gap = std::numeric_limits<double>::max();
        const size_t na = a.corners.size(), nb = b.corners.size();
        for (size_t i = 0; i < na; i++)
        {
            for (size_t j = 0; j < nb; j++)
            {
                gap = std::min(gap, pointSegmentDistance(a.corners[i], b.corners[j], b.corners[(j + 1) % nb]));
                gap = std::min(gap, pointSegmentDistance(b.corners[j], a.corners[i], a.corners[(i + 1) % na]));
            }
        }
        return gap;
    }

    struct Plane
    {
        Eigen::Vector3d sumPoints = Eigen::Vector3d::Zero();
        Eigen::Vector3d sumNormals = Eigen::Vector3d::Zero();
        double weight = 0.0;

        void add(const PlaneRegion & r)
        {
            Eigen::Vector3d n = r.planeToWorld.linear().col(2);
            if (weight > 0.0 && n.dot(sumNormals) < 0.0)
                n = -n;
            sumPoints += r.count * r.planeToWorld.translation();
            sumNormals += r.count * n;
            weight += r.count;
        }
        Eigen::Vector3d centroid() const { return sumPoints / weight; }
        Eigen::Vector3d normal() const { return sumNormals.normalized(); }
    };
}

std::vector<std::vector<int>> coplanarGroups(const std::vector<PlaneRegion> & regions, const SuperpixelParams & params)
{
    const int n = static_cast<int>(regions.size());
    std::vector<WorldHull> hulls;
    for (const auto & r : regions)
        hulls.push_back(worldHull(r));

    // Neighbors: hulls within merge_max_gap_ (bounding boxes first).
    std::vector<std::vector<int>> neighbors(n);
    const double gap = params.merge_max_gap_;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (hulls[i].corners.size() < 3 || hulls[j].corners.size() < 3)
                continue;
            if ((hulls[i].lo.array() > hulls[j].hi.array() + gap).any() || (hulls[j].lo.array() > hulls[i].hi.array() + gap).any())
                continue;
            if (hullGap(hulls[i], hulls[j]) <= gap)
            {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return regions[a].count > regions[b].count; });

    const double cosMaxAngle = std::cos(params.merge_max_angle_);
    std::vector<bool> assigned(n, false);
    std::vector<std::vector<int>> groups;
    for (int seed : order)
    {
        if (assigned[seed])
            continue;
        std::vector<int> group{seed};
        assigned[seed] = true;
        Plane plane;
        plane.add(regions[seed]);
        std::deque<int> frontier{seed};
        while (!frontier.empty())
        {
            const int current = frontier.front();
            frontier.pop_front();
            for (int candidate : neighbors[current])
            {
                if (assigned[candidate])
                    continue;
                const PlaneRegion & r = regions[candidate];
                const Eigen::Vector3d normal = plane.normal();
                const bool aligned = std::abs(r.planeToWorld.linear().col(2).dot(normal)) >= cosMaxAngle;
                const bool onPlane = std::abs((r.planeToWorld.translation() - plane.centroid()).dot(normal)) <= params.merge_max_offset_;
                if (aligned && onPlane)
                {
                    assigned[candidate] = true;
                    group.push_back(candidate);
                    plane.add(r);
                    frontier.push_back(candidate);
                }
            }
        }
        groups.push_back(group);
    }
    return groups;
}

void mergeGroup(const std::vector<PlaneRegion> & regions, const std::vector<int> & group,
                Eigen::Isometry3d & planeToWorld, std::vector<Eigen::Vector2d> & points)
{
    if (group.size() == 1)
    {
        planeToWorld = regions[group.front()].planeToWorld;
        points = regions[group.front()].points;
        return;
    }

    Plane plane;
    for (int i : group)
        plane.add(regions[i]);
    Eigen::Vector3d normal = plane.normal();
    if (normal.z() < 0.0)
        normal = -normal;
    // Plane x axis: world x projected onto the plane (world y if the plane faces along x).
    Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX() - normal.x() * normal;
    if (xAxis.norm() < 1e-6)
        xAxis = Eigen::Vector3d::UnitY() - normal.y() * normal;
    xAxis.normalize();
    Eigen::Matrix3d rotation;
    rotation << xAxis, normal.cross(xAxis), normal;
    planeToWorld = Eigen::Translation3d(plane.centroid()) * Eigen::Quaterniond(rotation);

    // Every member's points, through the world, into the merged plane (projected onto it).
    const Eigen::Isometry3d worldToPlane = planeToWorld.inverse();
    points.clear();
    for (int i : group)
    {
        for (const auto & p : regions[i].points)
            points.push_back((worldToPlane * (regions[i].planeToWorld * Eigen::Vector3d(p.x(), p.y(), 0.0))).head<2>());
    }
}
