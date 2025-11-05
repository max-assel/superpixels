#pragma once

#include <superpixels/utils.h>

class RegionSplitter
{
    public:
        RegionSplitter(const SuperpixelParams & params);

        void run(const std::vector<std::vector<double>> & centers,
                    const std::vector<std::vector<cv::Point>> & superpixels,
                    std::vector<std::vector<Eigen::Vector2d>> & superpixel_projections,
                    std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                    std::vector<Eigen::Matrix3d> & superpixel_rotations,
                    const cv::Mat & depth_image);

    private:
        SuperpixelParams params_;    
};
