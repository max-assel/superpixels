#include <superpixels/SuperpixelColorSegmenter.h>

SuperpixelColorSegmenter::SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
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
    }

    color_image_pub_ = it.advertise("/superpixels/color", 1);

    return;
}

void SuperpixelColorSegmenter::run()
{
    // Pre-processing
    cv::Mat lab_image;
    preprocessing(lab_image);

    // Superpixels algorithm
    generateSuperpixels(lab_image);

    // Create connectivity
    createConnectivity(lab_image);

    // color_image_ptr_->header.stamp = ros::Time::now();

    // color_image_pub_.publish(color_image_ptr_->toImageMsg());

    return;
}

void SuperpixelColorSegmenter::preprocessing(cv::Mat & lab_image)
{
    cv::cvtColor(color_image_ptr_->image, lab_image, cv::COLOR_BGR2Lab);
    int width = lab_image.cols;
    int height = lab_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double)params_.num_superpixels_); // superpixel grid interval
}

void SuperpixelColorSegmenter::generateSuperpixels(const cv::Mat & image)
{
    // Generate superpixels

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
    // Display center grid

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

cv::Point SuperpixelColorSegmenter::findLocalMinimum(cv::Mat & image, const cv::Point & center)
{
    return cv::Point(0, 0);
}

void SuperpixelColorSegmenter::clear_data()
{
    return;
}

void SuperpixelColorSegmenter::init_data(cv::Mat & image)
{
    return;
}