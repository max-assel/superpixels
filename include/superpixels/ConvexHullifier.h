#pragma once
#include <superpixels/utils.h>

#include <cv_bridge/cv_bridge.h>

class ConvexHullifier
{
    public:
        ConvexHullifier(const SuperpixelParams & params);

        void setParams(const SuperpixelParams & params);

        void run(const std::vector<std::vector<double>> & centers,
                    const std::vector<std::vector<cv::Point>> & superpixels,
                    const cv::Mat & depth_image,
                    const cv_bridge::CvImagePtr & raw_depth_img_ptr);

    private:

        void convexHull(const std::vector<Eigen::Vector2d> & superpixel, std::vector<Eigen::Vector2d> & convex_hull);

        bool polarSort(const Eigen::Vector2d & a, const Eigen::Vector2d & b, const Eigen::Vector2d & lowest);

        bool ccw(const Eigen::Vector2d & a, const Eigen::Vector2d & b, const Eigen::Vector2d & c);

        void grahamScan(const std::vector<Eigen::Vector2d> & superpixel, std::vector<Eigen::Vector2d> & convex_hull);

        Eigen::Vector3d projectPointOntoPlane(const Eigen::Vector3d & regionPt);

        SuperpixelParams params_;

        tf2_ros::TransformListener * tfListener_; /**< transform listener */

        tf2_ros::Buffer tfBuffer_; /**< transform buffer */ // TODO: add buffer?        
};
