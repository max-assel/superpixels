#pragma once

#include <superpixels/utils.h>

class Ransac
{
    public:
        Ransac(const SuperpixelParams & params);

        void setParams(const SuperpixelParams & params);

        cv::Vec3f run(const std::vector<cv::Point> & pixels, 
                        const cv::Mat & depth_image, 
                        const cv::Vec3f & og_normal);

    private:
        void sample(const std::vector<cv::Point> & pixels,
                    std::vector<cv::Point> & sample,
                    std::vector<int> & indices);

        void fit(const std::vector<cv::Point> & sample, 
                    const cv::Mat & depth_image,
                    const uint8_t & num_samples,
                    Eigen::VectorXf & x);

        void compute_inliers(const std::vector<cv::Point> & pixels,
                                const cv::Mat & depth_image,
                                std::vector<cv::Point> & inliers,
                                const Eigen::VectorXf & x);

        SuperpixelParams params_;   

        std::vector<cv::Point> samples;
        std::vector<int> indices;  
        std::vector<cv::Point> inliers;
        Eigen::VectorXf x;
        Eigen::VectorXf x_best;
        cv::Vec3f normal_best;         
};
