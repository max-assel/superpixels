#include <superpixels/Ransac.h>

Ransac::Ransac(const SuperpixelParams & params)
{
    params_ = params;
}

void Ransac::setParams(const SuperpixelParams & params)
{
    params_ = params;
}

cv::Vec3f Ransac::run(const std::vector<cv::Point> & pixels, 
                        const cv::Mat & depth_image, 
                        const cv::Vec3f & og_normal)
{
// RCLCPP_INFO_STREAM(node_->get_logger(), "   [SuperpixelDepthSegmenter::ransac]");

    // RCLCPP_INFO_STREAM(node_->get_logger(), "       number of points: " << pixels.size());

    cv::Vec3f normal = og_normal;

    if (pixels.size() < params_.ransac_K)
        return normal;

    int max_inliers = 0;

    for (int n = 0; n < params_.ransac_N; n++)
    {
        samples = std::vector<cv::Point>(params_.ransac_K);
        indices = std::vector<int>(params_.ransac_K);
        inliers.clear();

        // RCLCPP_INFO_STREAM(node_->get_logger(), "       iteration: " << n);

        // RCLCPP_INFO_STREAM(node_->get_logger(), "           sampling ...");
        sample(pixels, samples, indices);

        // RCLCPP_INFO_STREAM(node_->get_logger(), "           fitting ...");
        // fit
        fit(samples, depth_image, params_.ransac_K, x);

        // RCLCPP_INFO_STREAM(node_->get_logger(), "           computing inliers ...");
        // compute inliers
        compute_inliers(pixels, depth_image, inliers, x);

        // RCLCPP_INFO_STREAM(node_->get_logger(), "           number of inliers: " << inliers.size());

        // RCLCPP_INFO_STREAM(node_->get_logger(), "           updating ...");
        // update
        if (inliers.size() > max_inliers)
        {
            max_inliers = inliers.size();

            // update normal
            fit(inliers, depth_image, max_inliers, x_best);

        }
    }

    normal_best = cv::Vec3f(x_best(0), -1.0, x_best(2));

    normal = normal_best / cv::norm(normal_best);

    return normal;    
}

void Ransac::sample(const std::vector<cv::Point> & pixels,
                    std::vector<cv::Point> & samples,
                    std::vector<int> & indices)
{
    int idx = -1;

    for (int i = 0; i < params_.ransac_K; i++)
    {
        idx = -1;
        while (idx == -1 || std::find(indices.begin(), indices.end(), idx) != indices.end())
        {
            int rand_idx = rand();
            idx = rand_idx % pixels.size();
            // RCLCPP_INFO_STREAM(node_->get_logger(), "           rand_idx: " << rand_idx << ", idx: " << idx);
        }
        samples[i] = pixels[idx];
        indices[i] = idx;
    }
}

void Ransac::fit(const std::vector<cv::Point> & samples,
                    const cv::Mat & depth_image,
                    const int & num_samples,
                    Eigen::VectorXf & x)
{
    Eigen::MatrixXf A(num_samples, 3);
    Eigen::VectorXf b(num_samples);

    cv::Point pixel;
    cv::Vec3f egocanPt;

    for (int i = 0; i < num_samples; i++)
    {
        pixel = samples[i];
        
        pixelToEgocanFrame(egocanPt, pixel, depth_image.at<float>(pixel.y, pixel.x), params_.half_k_c_, params_.two_over_h_times_k_c_);

        A(i, 0) = egocanPt.val[0];
        A(i, 1) = 1.0;
        A(i, 2) = egocanPt.val[2];
        b(i) = egocanPt.val[1];
    }

    x = A.colPivHouseholderQr().solve(b);
}

void Ransac::compute_inliers(const std::vector<cv::Point> & pixels,
                                const cv::Mat & depth_image,
                                std::vector<cv::Point> & inliers,
                                const Eigen::VectorXf & x)
{
    cv::Point pixel;
    cv::Vec3f egocanPt;
    float y_hat;
    float error;

    for (int i = 0; i < pixels.size(); i++)
    {
        pixel = pixels[i];
        pixelToEgocanFrame(egocanPt, pixel, depth_image.at<float>(pixel.y, pixel.x), params_.half_k_c_, params_.two_over_h_times_k_c_);

        y_hat = x(0) * egocanPt.val[0] + x(1) + x(2) * egocanPt.val[2];
        error = std::abs(egocanPt.val[1] - y_hat);

        // do not know size beforehand
        if (error < params_.ransac_T)
            inliers.push_back(pixel);
    }    
}