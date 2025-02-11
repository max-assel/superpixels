#pragma once
#include <superpixels/utils.h>

class ConvexHullifier
{
    public:
        ConvexHullifier(const SuperpixelParams & params);

        void setParams(const SuperpixelParams & params);

        void run(const std::vector<std::vector<double>> & centers,
                 const std::vector<std::vector<cv::Point>> & superpixels,
                 const cv::Mat & depth_image);

    private:

        cv::Vec3f projectPointOntoPlane(const cv::Vec3f & worldPt,
                                        const cv::Vec3f & centerWorldPt,
                                        const cv::Vec3f & normal);

        SuperpixelParams params_;
};
