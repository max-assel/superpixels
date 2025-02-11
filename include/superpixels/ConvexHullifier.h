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

        Eigen::Vector3d projectPointOntoPlane(const Eigen::Vector3d & regionPt);

        SuperpixelParams params_;

        tf2_ros::TransformListener * tfListener_; /**< transform listener */

        tf2_ros::Buffer tfBuffer_; /**< transform buffer */ // TODO: add buffer?        
};
