#include <superpixels/ConvexHullifier.h>

ConvexHullifier::ConvexHullifier(const SuperpixelParams & params)
{
    params_ = params;
}

void ConvexHullifier::setParams(const SuperpixelParams & params)
{
    params_ = params;
}

cv::Vec3f ConvexHullifier::projectPointOntoPlane(const cv::Vec3f & worldPt,
                                                  const cv::Vec3f & centerWorldPt,
                                                  const cv::Vec3f & normal)
{
    cv::Vec3f projPt;

    cv::Vec3f diff = worldPt - centerWorldPt;
    float dist = diff.dot(normal);
    projPt = worldPt - dist * normal;

    return projPt;
}

void ConvexHullifier::run(const std::vector<std::vector<double>> & centers,
                            const std::vector<std::vector<cv::Point>> & superpixels,
                            const cv::Mat & depth_image)
{
    std::vector<std::vector<cv::Vec3f>> superpixel_projections(centers.size());
    // std::vector<std::vector<cv::Point>> superpixel_convex_hulls(centers.size());

    // Build convex hulls for each superpixel
    for (int i = 0; i < (int) centers.size(); i++)
    {
        // project points onto plane
        superpixel_projections[i].resize(superpixels[i].size());
        for (int j = 0; j < superpixels[i].size(); j++)
        {
            cv::Point center = cv::Point(centers[i][0], centers[i][1]);
            cv::Point pixel = superpixels[i][j];
            cv::Vec3f normal = cv::Vec3f(centers[i][4], centers[i][5], centers[i][6]);
            float depth = depth_image.at<float>(pixel.y, pixel.x);

            cv::Vec3f worldPt;
            floorPixelToWorld(worldPt, pixel, depth, params_.k_c_, params_.h_);

            cv::Vec3f centerWorldPt;
            floorPixelToWorld(centerWorldPt, center, centers[i][2], params_.k_c_, params_.h_);

            cv::Vec3f projPt = projectPointOntoPlane(worldPt, centerWorldPt, normal);

            superpixel_projections[i][j] = projPt;
        }
        
        // Get 2d plane points

        // Calculate convex hull and store
    }

    return;
}