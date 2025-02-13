#pragma once

#include <superpixels/utils.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <random>

class ImagePreprocessor
{
    public:
        ImagePreprocessor(const SuperpixelParams & params);

        void setParams(const SuperpixelParams & params);

        void cleanImages(const cv::Mat & raw_depth_img,
                            // const cv::Mat & raw_label_img,
                            const cv::Mat & raw_normal_img,
                            cv::Mat & checked_depth_img,
                            // cv::Mat & checked_label_img,
                            cv::Mat & checked_normal_img,
                            cv::Mat & visited);

        void healthCheck(const cv::Mat & depth_img,
                            // const cv::Mat & label_img,
                            const cv::Mat & normal_img);

        void preprocessImages(const cv::Mat & checked_depth_img,
                                // const cv::Mat & checked_label_img,
                                const cv::Mat & checked_normal_img,
                                cv::Mat & preprocessed_depth_img,
                                // cv::Mat & preprocessed_label_img,
                                cv::Mat & preprocessed_normal_img);

        void fillInImage(const cv::Mat & cleaned_depth_img,
                            // const cv::Mat & cleaned_label_img,
                            const cv::Mat & cleaned_normal_img,
                            const cv::Mat & visited,
                            cv::Mat & filled_depth_img,
                            // cv::Mat & filled_label_img,
                            cv::Mat & filled_normal_img,
                            const cv_bridge::CvImagePtr & raw_depth_img_ptr);

    private:
        SuperpixelParams params_;

        tf2_ros::TransformListener * tfListener_; /**< transform listener */

        tf2_ros::Buffer tfBuffer_; /**< transform buffer */ // TODO: add buffer?      

        std::default_random_engine generator;
};