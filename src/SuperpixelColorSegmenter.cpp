#include <superpixels/SuperpixelColorSegmenter.h>

SuperpixelColorSegmenter::SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    ROS_INFO_STREAM("[SuperpixelColorSegmenter::SuperpixelColorSegmenter]");

    nh_ = nh;

    image_transport::ImageTransport it(nh);

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();
    params_.n_c_ = configYamlNode["superpixels"]["n_c"].as<int>();
    params_.n_s_ = configYamlNode["superpixels"]["n_s"].as<int>();

    color_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    // read in example image
    const std::string superixels_path = ros::package::getPath("superpixels");
    const std::string image_name = "flower.jpg";

    color_image_ptr_->header.stamp = ros::Time::now();
    color_image_ptr_->header.frame_id = "N/A";
    color_image_ptr_->header.seq = 0;
    color_image_ptr_->encoding = sensor_msgs::image_encodings::BGR8;
    color_image_ptr_->image = cv::imread(superixels_path + "/images/" + image_name, cv::IMREAD_COLOR);

    if (color_image_ptr_->image.empty())
    {
        ROS_ERROR_STREAM("Could not read the image: " << image_name);
        return;
    } else
    {
        ROS_INFO_STREAM("Read the image: " << image_name);
    }

    center_grid_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    center_grid_image_ptr_->header = color_image_ptr_->header;
    center_grid_image_ptr_->encoding = color_image_ptr_->encoding;

    color_image_pub_ = it.advertise("/superpixels/color", 1);
    center_grid_image_pub_ = it.advertise("/superpixels/center_grid", 1);

    return;
}

void SuperpixelColorSegmenter::run()
{
    ROS_INFO_STREAM("[SuperpixelColorSegmenter::run]");

    // Pre-processing
    cv::Mat lab_image;
    preprocessing(lab_image);

    // Superpixels algorithm
    generateSuperpixels(lab_image);

    // Create connectivity
    // createConnectivity(lab_image);

    // color_image_ptr_->header.stamp = ros::Time::now();

    // color_image_pub_.publish(color_image_ptr_->toImageMsg());

    return;
}

void SuperpixelColorSegmenter::preprocessing(cv::Mat & lab_image)
{
    ROS_INFO_STREAM("   [SuperpixelColorSegmenter::preprocessing]");

    cv::cvtColor(color_image_ptr_->image, lab_image, cv::COLOR_BGR2Lab);
    int width = lab_image.cols;
    int height = lab_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double)params_.num_superpixels_); // superpixel grid interval
}

void SuperpixelColorSegmenter::generateSuperpixels(const cv::Mat & image)
{
    ROS_INFO_STREAM("   [SuperpixelColorSegmenter::generateSuperpixels]");

    // Clear data
    clear_data(); // TODO: add flag for warm-starting

    // Initialize data
    init_data(image);
    
    // Generate superpixels

    // Display center grid
    cv::Mat center_grid_image = color_image_ptr_->image.clone();
    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(center_grid_image, color);
    center_grid_image_ptr_->image = center_grid_image;
    center_grid_image_ptr_->header.stamp = ros::Time::now();
    center_grid_image_pub_.publish(center_grid_image_ptr_->toImageMsg());
    return;
}

void SuperpixelColorSegmenter::createConnectivity(const cv::Mat & image)
{
    // Create connectivity

    // Calculate distance between superpixels

    // Create graph

    return;
}

void SuperpixelColorSegmenter::displayCenterGrid(cv::Mat & image, const cv::Vec3b & color)
{
    ROS_INFO_STREAM("   [SuperpixelColorSegmenter::displayCenterGrid]");
    
    // Display center grid
    for (int i = 0; i < (int) centers_.size(); i++) 
    {
        cv::circle(image, cv::Point(centers_[i][3], centers_[i][4]), 2, color, -1);
    }

    return;
}

void SuperpixelColorSegmenter::displayContours(cv::Mat & image, const cv::Vec3b & color)
{
    // Display contours

    return;
}

void SuperpixelColorSegmenter::displaySuperpixelsWithClusterMeans(cv::Mat & image)
{
    // Display superpixels with cluster means

    return;
}

double SuperpixelColorSegmenter::computeDistance(const int & center_idx, const cv::Vec3b & color, const cv::Point & pixel)
{
    return 0.0;
}

cv::Point SuperpixelColorSegmenter::findLocalMinimum(const cv::Mat & image, const cv::Point & center)
{
    double min_grad = std::numeric_limits<double>::max();
    cv::Point loc_min = center;

    for (int i = center.x - 1; i <= center.x + 1; i++)
    {
        for (int j = center.y - 1; j <= center.y + 1; j++)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(j, i);
            double grad = sqrt(pow(color.val[0] - image.at<cv::Vec3b>(center.y, center.x).val[0], 2) +
                               pow(color.val[1] - image.at<cv::Vec3b>(center.y, center.x).val[1], 2) +
                               pow(color.val[2] - image.at<cv::Vec3b>(center.y, center.x).val[2], 2));

            if (grad < min_grad)
            {
                min_grad = grad;
                loc_min = cv::Point(i, j);
            }
        }
    }

    return loc_min;
}

void SuperpixelColorSegmenter::clear_data()
{
    ROS_INFO_STREAM("   [SuperpixelColorSegmenter::clear_data]");
    clusters_.release();
    distances_.release();
    centers_.clear();
    center_counts_.clear();

    return;
}

void SuperpixelColorSegmenter::init_data(const cv::Mat & image)
{
    ROS_INFO_STREAM("   [SuperpixelColorSegmenter::init_data]");

    /* Initialize the cluster and distance matrices. */
    clusters_ = cv::Mat(image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer
    distances_ = cv::Mat(image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point

    /* Initialize the centers and counters. */
    centers_.clear();
    center_counts_.clear();
    for (int i = params_.step_; i < image.cols - params_.step_ / 2; i += params_.step_)
    {
        for (int j = params_.step_; j < image.rows - params_.step_ / 2; j += params_.step_)
        {
            std::vector<double> center;

            /* Find the local minimum (gradient-wise). */
            cv::Point localMinimum = findLocalMinimum(image, cv::Point(i, j));
            cv::Vec3b color = image.at<cv::Vec3b>(localMinimum.y, localMinimum.x);

            /* Generate the center vector. */
            center.push_back(color.val[0]);
            center.push_back(color.val[1]);
            center.push_back(color.val[2]);
            center.push_back(localMinimum.x);
            center.push_back(localMinimum.y);

            /* Append to vector of centers. */
            centers_.push_back(center);
            center_counts_.push_back(0);
        }
    }

    return;
}