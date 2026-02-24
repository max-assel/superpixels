#include <superpixels/RegionSplitter.h>

RegionSplitter::RegionSplitter(const SuperpixelParams & params)
{
    params_ = params;
}

bool RegionSplitter::polarSort(const Eigen::Vector2f & a, const Eigen::Vector2f & b)
{
    float angle_a = std::atan2(a[1], a[0]);
    float angle_b = std::atan2(b[1], b[0]);

    if (angle_a < angle_b)
    {
        return true;
    }
    else if (angle_a == angle_b)
    {
        return a.norm() < b.norm();
    }
    else
    {
        return false;
    }
}


void RegionSplitter::run(const std::vector<std::vector<float>> & centers,
                            const std::vector<std::vector<cv::Point>> & superpixels,
                            std::vector<std::vector<Eigen::Vector2f>> & superpixel_projections,
                            std::vector<Eigen::Matrix3f> & egocan_to_region_rotations,
                            const cv::Mat & depth_img)
{
    RCLCPP_INFO_STREAM(logger_, "   [RegionSplitter::run]");

    std::vector<std::vector<float>> centers_lc = centers;
    std::vector<std::vector<cv::Point>> superpixels_lc = superpixels;

    float angle_threshold = M_PI / 6.0; // 30 degrees

    // Loop through each superpixel
    for (int i = 0; i < (int) centers_lc.size(); i++)
    {
        // Grab superpixel
        cv::Point center_pixel = cv::Point(centers_lc[i][0], centers_lc[i][1]);
        float center_depth = centers_lc[i][2];
        cv::Vec3f centerEgocanPt;
        pixelToEgocanFrame(centerEgocanPt, center_pixel, center_depth, params_.k_c_, params_.h_);

        Eigen::Vector3f center(centerEgocanPt.val[0], centerEgocanPt.val[1], centerEgocanPt.val[2]);

        // Get projected 2D points
        std::vector<Eigen::Vector2f> superpixel_projection = superpixel_projections[i];

        // Sort points by bearing from center
        std::vector<Eigen::Vector2f> superpixel_projection_sorted = superpixel_projection;
        std::sort(superpixel_projection_sorted.begin(), superpixel_projection_sorted.end(), std::bind(&RegionSplitter::polarSort, 
                                                    this, 
                                                    std::placeholders::_1, 
                                                    std::placeholders::_2)); 

        // Split points into regions based on angle change
        for (int j = 1; j < (int) superpixel_projection_sorted.size(); j++)
        {
            Eigen::Vector2f prev_point = superpixel_projection_sorted[j - 1];
            Eigen::Vector2f current_point = superpixel_projection_sorted[j];

            RCLCPP_INFO_STREAM(logger_, "       superpixel " << i << ", point " << j << ":");
            RCLCPP_INFO_STREAM(logger_, "           previous point: " << prev_point[0] << ", " << prev_point[1]);
            RCLCPP_INFO_STREAM(logger_, "           current point: " << current_point[0] << ", " << current_point[1]);

            float angle_prev = std::atan2(prev_point[1], prev_point[0]);
            float angle_current = std::atan2(current_point[1], current_point[0]);

            float angle_diff = angle_current - angle_prev;
            if (angle_diff < 0)
            {
                angle_diff += 2.0 * M_PI;
            }

            RCLCPP_INFO_STREAM(logger_, "           angle difference: " << angle_diff << " radians.");

            if (angle_diff > angle_threshold)
            {
                RCLCPP_INFO_STREAM(logger_, "       Splitting region " << i << " at point " << j << " due to angle change of " << angle_diff << " radians.");

                // Split region here
                // For simplicity, we will just log the split. Actual splitting logic would go here.
            }
        }


    }
    return;
}