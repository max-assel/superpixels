#include <superpixels/RegionSplitter.h>

RegionSplitter::RegionSplitter(const SuperpixelParams & params)
{
    params_ = params;
}

void RegionSplitter::run(const std::vector<std::vector<double>> & centers,
                            const std::vector<std::vector<cv::Point>> & superpixels,
                            std::vector<std::vector<Eigen::Vector2d>> & superpixel_projections,
                            std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                            std::vector<Eigen::Matrix3d> & egocan_to_region_rotations,
                            const cv::Mat & depth_img)
{
    RCLCPP_INFO_STREAM(node_->get_logger(), "   [RegionSplitter::run]");

    std::vector<std::vector<double>>centers_lc = centers;
    std::vector<std::vector<cv::Point>> superpixels_lc = superpixels;

    // Loop through each superpixel
    for (int i = 0; i < (int) centers_lc.size(); i++)
    {
        // Grab superpixel
        cv::Point center_pixel = cv::Point(centers_lc[i][0], centers_lc[i][1]);
        float center_depth = centers_lc[i][2];
        cv::Vec3f centerEgocanPt;
        pixelToEgocanFrame(centerEgocanPt, center_pixel, center_depth, params_.k_c_, params_.h_);

        Eigen::Vector3d center(centerEgocanPt.val[0], centerEgocanPt.val[1], centerEgocanPt.val[2]);

        // Get points
        std::vector<cv::Point> superpixel_points = superpixels_lc[i];

        // Sort points by bearing from center
        

        // Split points into regions based on angle change



    }
    return;
}