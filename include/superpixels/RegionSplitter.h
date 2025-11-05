#pragma once

#include <superpixels/utils.h>

class RegionSplitter
{
    public:
        RegionSplitter(const SuperpixelParams & params);

        void run(const std::vector<std::vector<double>> & centers,
                    const std::vector<std::vector<cv::Point>> & superpixels,
                    std::vector<std::vector<Eigen::Vector2d>> & superpixel_projections,
                    std::vector<Eigen::Matrix3d> & superpixel_rotations,
                    const cv::Mat & depth_image);

    private:

        bool polarSort(const Eigen::Vector2d & a, const Eigen::Vector2d & b);

        SuperpixelParams params_;    
        rclcpp::Logger logger_ = rclcpp::get_logger("RegionSplitter");
            
};
