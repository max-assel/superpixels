#include <superpixels/SuperpixelColorOpenCVSegmenter.h>

SuperpixelColorOpenCVSegmenter::SuperpixelColorOpenCVSegmenter(ros::NodeHandle nh, const std::string & config_path) : SuperpixelSegmenter(nh, config_path), nh_(nh)
{

    image_transport::ImageTransport it(nh);

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

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

    // Load algorithm
    const std::string algorithm_name = configYamlNode["superpixels"]["algorithm"].as<std::string>();
    num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();

    int algorithm;
    if (algorithm_name == "SLIC")
    {
        algorithm = cv::ximgproc::SLIC;
    } else if (algorithm_name == "SLICO")
    {
        algorithm = cv::ximgproc::SLICO;
    } else if (algorithm_name == "MSLIC")
    {
        algorithm = cv::ximgproc::MSLIC;
    } else
    {
        ROS_ERROR_STREAM("Invalid superpixels algorithm: " << algorithm_name);
        return;
    }

    const int region_size = configYamlNode["superpixels"]["region_size"].as<int>();
    const double ruler = configYamlNode["superpixels"]["ruler"].as<double>();

    slic_ = cv::ximgproc::createSuperpixelSLIC(color_image_ptr_->image, algorithm, region_size, ruler);

    color_image_pub_ = it.advertise("/superpixels/color", 1);
    mask_image_pub_ = it.advertise("/superpixels/mask", 1);

    return;
}

void SuperpixelColorOpenCVSegmenter::runSegmentation()
{
    color_image_ptr_->header.stamp = ros::Time::now();

    // Superpixels algorithm
    cv::Mat contourMask = runSuperpixels();

    // mask_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    // mask_image_ptr_->header = color_image_ptr_->header;
    // mask_image_ptr_->encoding = color_image_ptr_->encoding;
    // mask_image_ptr_->image = cv::Mat::zeros(color_image_ptr_->image.size(), CV_8UC3);
    
    // ROS_INFO_STREAM("contourMask middle pixel: " << contourMask.at<uint8_t>(contourMask.rows/2, contourMask.cols/2));

    color_image_pub_.publish(color_image_ptr_->toImageMsg());
    // mask_image_pub_.publish(mask_image_ptr_->toImageMsg());

    return;
}

cv::Mat SuperpixelColorOpenCVSegmenter::runSuperpixels()
{
    // Run superpixels algorithm
    slic_->iterate(num_iterations_);

    cv::Mat contourMask;
    slic_->getLabelContourMask(contourMask, true);
    
    return contourMask;
}

double SuperpixelColorOpenCVSegmenter::calculateDistance()
{
    return 0.0;
}