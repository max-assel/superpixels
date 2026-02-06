#pragma once

#include <superpixels/utils.h>

#include <cv_bridge/cv_bridge.h>

class ConvexHullifier
{
    public:
        ConvexHullifier(const SuperpixelParams & params);

        void setParams(const SuperpixelParams & params);

        void run(std::vector<std::vector<float>> & centers,
                    std::vector<int> & center_counts,
                    std::vector<std::vector<cv::Point>> & superpixels,
                    std::vector<std::vector<Eigen::Vector2f>> & superpixel_projections,
                    std::vector<std::vector<Eigen::Vector2f>> & superpixel_convex_hulls,
                    std::vector<Eigen::Matrix3f> & superpixel_rotations,
                    const cv::Mat & depth_image);

    private:

        void convexHull(const std::vector<Eigen::Vector2f> & superpixel, std::vector<Eigen::Vector2f> & convex_hull);

        bool polarSort(const Eigen::Vector2f & a, const Eigen::Vector2f & b, const Eigen::Vector2f & lowest);

        bool ccw(const Eigen::Vector2f & a, const Eigen::Vector2f & b, const Eigen::Vector2f & c);

        void grahamScan(const std::vector<Eigen::Vector2f> & superpixel, std::vector<Eigen::Vector2f> & convex_hull);

        Eigen::Vector3f projectPointOntoPlane(const Eigen::Vector3f & regionPt);

        SuperpixelParams params_;    

        rclcpp::Logger logger_ = rclcpp::get_logger("ConvexHullifier");
};
