#pragma once

#include <superpixels/utils.h>

class RegionSplitter
{
    public:
        RegionSplitter(const SuperpixelParams & params);

        void run(const std::vector<std::vector<float>> & centers,
                    const std::vector<std::vector<cv::Point>> & superpixels,
                    std::vector<std::vector<Eigen::Vector2f>> & superpixel_projections,
                    std::vector<Eigen::Matrix3f> & superpixel_rotations,
                    const cv::Mat & depth_image);

    private:

        bool polarSort(const Eigen::Vector2f & a, const Eigen::Vector2f & b);

        SuperpixelParams params_;    
        rclcpp::Logger logger_ = rclcpp::get_logger("RegionSplitter");
            
};
