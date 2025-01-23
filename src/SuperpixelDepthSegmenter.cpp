#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    nh_ = nh;

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();
    params_.n_c_ = configYamlNode["superpixels"]["n_c"].as<int>();
    params_.n_s_ = configYamlNode["superpixels"]["n_s"].as<int>();
    params_.num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();
    params_.warm_start_ = configYamlNode["superpixels"]["warm_start"].as<bool>();

    ROS_INFO_STREAM("   params_:");
    ROS_INFO_STREAM("       num_superpixels_: " << params_.num_superpixels_);
    ROS_INFO_STREAM("       n_c_: " << params_.n_c_);
    ROS_INFO_STREAM("       n_s_: " << params_.n_s_);
    ROS_INFO_STREAM("       num_iterations_: " << params_.num_iterations_);
    ROS_INFO_STREAM("       warm_start_: " << params_.num_iterations_);

    // Set up subscribers and publishers
    image_transport::ImageTransport it(nh);

    std::string depth_image_topic =  "/egocylinder/floor_image";
    std::string label_image_topic =  "/egocylinder/floor_labels";
    std::string normal_image_topic = "/egocylinder/floor_normals";

    nh_.getParam("depth_image_topic", depth_image_topic);
    nh_.getParam("label_image_topic", label_image_topic);
    nh_.getParam("normal_image_topic", normal_image_topic);

    depth_image_sub_.subscribe(it, depth_image_topic, 3);
    label_image_sub_.subscribe(it, label_image_topic, 3);
    normal_image_sub_.subscribe(it, normal_image_topic, 3);

    msg_sync_ = boost::make_shared<MsgSynchronizer>(depth_image_sub_, label_image_sub_, normal_image_sub_, 10);
    msg_sync_->registerCallback(boost::bind(&SuperpixelDepthSegmenter::allImageCallback, this, _1, _2, _3));
}

void SuperpixelDepthSegmenter::allImageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg, 
                                                const sensor_msgs::ImageConstPtr& label_image_msg, 
                                                const sensor_msgs::ImageConstPtr& normal_image_msg)
{   
    std::lock_guard<std::mutex> lock(img_mutex_);

    // ROS_INFO_STREAM("[SuperpixelDepthSegmenter::allImageCallback]");
    // ROS_INFO_STREAM("       time stamp: " << depth_image->header.stamp);

    depth_image_msg_ = depth_image_msg;
    label_image_msg_ = label_image_msg;
    normal_image_msg_ = normal_image_msg;

    return;
}

bool SuperpixelDepthSegmenter::notReceivedDepthImage()
{
    return (depth_image_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedLabelImage()
{
    return (label_image_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedNormalImage()
{
    return (normal_image_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedImage()
{
    bool notReceivedDepth = notReceivedDepthImage();
    bool notReceivedLabel = notReceivedLabelImage();
    bool notReceivedNormal = notReceivedNormalImage();

    if (notReceivedDepth)
        ROS_WARN_STREAM("Not received depth image.");

    if (notReceivedLabel)
        ROS_WARN_STREAM("Not received label image.");

    if (notReceivedNormal)
        ROS_WARN_STREAM("Not received normal image.");

    return (notReceivedDepth || notReceivedLabel || notReceivedNormal);
}

void SuperpixelDepthSegmenter::run()
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if (notReceivedImage())
    {
        ROS_WARN("Not ready to segment, no images received yet.");
        return;
    }
 
    try
    {
        depth_image_ptr_ = cv_bridge::toCvCopy(depth_image_msg_, sensor_msgs::image_encodings::TYPE_32FC1);
        label_image_ptr_ = cv_bridge::toCvCopy(label_image_msg_, sensor_msgs::image_encodings::TYPE_8UC1);
        normal_image_ptr_ = cv_bridge::toCvCopy(normal_image_msg_, sensor_msgs::image_encodings::TYPE_32FC3);

    } catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (depth_image_ptr_->image.size() != label_image_ptr_->image.size() || 
        depth_image_ptr_->image.size() != normal_image_ptr_->image.size() ||
        label_image_ptr_->image.size() != normal_image_ptr_->image.size())
    {
        ROS_ERROR("Image sizes do not match.");
        return;
    }

    // checkSparsity();

    cv::Mat depth_image = depth_image_ptr_->image;
    cv::Mat label_image = label_image_ptr_->image;
    cv::Mat normal_image = normal_image_ptr_->image;

    // if (!initialized_)
    // {

    // Pre-processing
    preprocessing(depth_image);

    // Initialize data
    init_data(depth_image);

    //     initialized_ = true;
    // }

    return;
}

void SuperpixelDepthSegmenter::visualize()
{
    return;
}

void SuperpixelDepthSegmenter::preprocessing(const cv::Mat & depth_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::preprocessing]");

    int width = depth_image.cols;
    int height = depth_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval
}

void SuperpixelDepthSegmenter::init_data(const cv::Mat & depth_image)
{
    /* Initialize the cluster and distance matrices. */
    clusters_ = cv::Mat(depth_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer
    distances_ = cv::Mat(depth_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point

    /* Initialize the centers and counters. */
    // int rough_center_count = 0;
    centers_.clear();
    center_counts_.clear();
    for (int c = params_.step_; c < depth_image.cols - (params_.step_ / 2); c += params_.step_)
    {
        for (int r = params_.step_; r < depth_image.rows - (params_.step_ / 2); r += params_.step_)
        {
            // ROS_INFO_STREAM("       (r, c): (" << r << ", " << c << ")");

            // float depth = depth_image.at<float>(r, c);

            // if (std::isnan(depth) || std::fabs(depth) < 1e-6)
            // {
            //     continue;
            // }

            std::vector<double> center;

            /* Find the local minimum (gradient-wise). */
            cv::Point localMinimum(c, r);
            bool success = findLocalMinimum(depth_image, localMinimum);
            // cv::Vec3b color = lab_image.at<cv::Vec3b>(localMinimum.y, localMinimum.x);

            if (!success)
            {
                continue;
            }


            /* Generate the center vector. */
            // center.push_back(depth);
            center.push_back(localMinimum.x);
            center.push_back(localMinimum.y);

            /* Append to vector of centers. */
            centers_.push_back(center);
            center_counts_.push_back(0);
        }
    }

    ROS_INFO_STREAM("       centers_.size(): " << centers_.size());
    ROS_INFO_STREAM("       center_counts_.size(): " << center_counts_.size());

}

bool SuperpixelDepthSegmenter::findLocalMinimum(const cv::Mat & depth_image, cv::Point & loc_min)
{
    // double min_grad = std::numeric_limits<double>::max();
    // cv::Point loc_min = center;
    const cv::Point center = loc_min; 

    int deltaX = 1;
    int deltaY = 1;

    float center_color = depth_image.at<float>(center.y, center.x);

    for (int c = center.x - deltaX; c <= center.x + deltaX; c++)
    {
        for (int r = center.y - deltaY; r <= center.y + deltaY; r++)
        {
            float depth = depth_image.at<float>(r, c);

            // double grad = sqrt(pow(color.val[0] - center_color.val[0], 2) +
            //                    pow(color.val[1] - center_color.val[1], 2) +
            //                    pow(color.val[2] - center_color.val[2], 2));

            if (std::isnan(depth) || std::fabs(depth) < 1e-6)
            {
                continue;
            } else
            {
                loc_min = cv::Point(c, r);
                return true;
            }
        }
    }

    return false;

    // return loc_min;
}

void SuperpixelDepthSegmenter::checkSparsity()
{

    int rows = depth_image_ptr_->image.rows;
    int cols = depth_image_ptr_->image.cols;

    // Sanity check middle pixel
    ROS_INFO_STREAM("Depth image size --- rows: " << depth_image_ptr_->image.rows << ", cols: " << depth_image_ptr_->image.cols);
    ROS_INFO_STREAM("Label image size --- rows: " << label_image_ptr_->image.rows << ", cols: " << label_image_ptr_->image.cols);
    ROS_INFO_STREAM("Normal image size --- rows: " << normal_image_ptr_->image.rows << ", cols: " << normal_image_ptr_->image.cols);

    int nan_depth_count = 0;
    int nan_label_count = 0;
    int nan_normal_count = 0;

    int finite_depth_count = 0;
    int finite_label_count = 0;
    int finite_normal_count = 0;

    int zero_depth_count = 0;
    int zero_label_count = 0;
    int zero_normal_count = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (depth_image_ptr_->image.at<float>(r, c) != depth_image_ptr_->image.at<float>(r, c))
            {
                // ROS_ERROR_STREAM("Depth image has NaN value at row: " << r << ", col: " << c);
                nan_depth_count++;
            } else if (depth_image_ptr_->image.at<float>(r, c) == 0)
            {
                zero_depth_count++;
            } else
            {
                // ROS_INFO_STREAM("Depth image value at row: " << r << ", col: " << c << " is: " << depth_image_ptr_->image.at<float>(r, c));
                finite_depth_count++;
            }

            if (label_image_ptr_->image.at<uint8_t>(r, c) != label_image_ptr_->image.at<uint8_t>(r, c))
            {
                // ROS_ERROR_STREAM("Label image has NaN value at row: " << r << ", col: " << c);
                nan_label_count++;
            } else if (label_image_ptr_->image.at<uint8_t>(r, c) == 0)
            {
                zero_label_count++;
            } else
            {
                finite_label_count++;
            }

            if (normal_image_ptr_->image.at<cv::Vec3f>(r, c)[0] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[0] ||
                normal_image_ptr_->image.at<cv::Vec3f>(r, c)[1] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[1] ||
                normal_image_ptr_->image.at<cv::Vec3f>(r, c)[2] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[2])
            {
                // ROS_ERROR_STREAM("Normal image has NaN value at row: " << r << ", col: " << c);
                nan_normal_count++;
            } else if (normal_image_ptr_->image.at<cv::Vec3f>(r, c)[0] == 0 &&
                       normal_image_ptr_->image.at<cv::Vec3f>(r, c)[1] == 0 &&
                       normal_image_ptr_->image.at<cv::Vec3f>(r, c)[2] == 0)
            {
                zero_normal_count++;
            } else
            {
                // ROS_INFO_STREAM("Normal image value at row: " << r << ", col: " << c << " is: " << normal_image_ptr_->image.at<cv::Vec3f>(r, c));
                finite_normal_count++;
            }
        }
    }

    int total_pixels = rows * cols;

    ROS_INFO_STREAM("Depth image NaN count: " << nan_depth_count);
    ROS_INFO_STREAM("Depth image zero count: " << zero_depth_count);
    ROS_INFO_STREAM("Depth image finite count: " << finite_depth_count);

    ROS_INFO_STREAM("Label image NaN count: " << nan_label_count);
    ROS_INFO_STREAM("Label image zero count: " << zero_label_count);
    ROS_INFO_STREAM("Label image finite count: " << finite_label_count);

    ROS_INFO_STREAM("Normal image NaN count: " << nan_normal_count);
    ROS_INFO_STREAM("Normal image zero count: " << zero_normal_count);
    ROS_INFO_STREAM("Normal image finite count: " << finite_normal_count);

    // ROS_INFO_STREAM("Depth image sparsity ratio: " << float(finite_depth_count) / total_pixels);
    // ROS_INFO_STREAM("Label image sparsity ratio: " << float(finite_label_count) / total_pixels);
    // ROS_INFO_STREAM("Normal image sparsity ratio: " << float(finite_normal_count) / total_pixels);

    return;    
}