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
    overlay_image_pub_ = it.advertise("/superpixels/overlaid_color", 1);

    return;
}

void SuperpixelColorOpenCVSegmenter::runSegmentation()
{
    // Superpixels algorithm
    cv::Mat contourMask;
    cv::Mat labels;
    int num_superpixels;
    std::chrono::steady_clock::time_point timeBegin, timeEnd;
    timeBegin = std::chrono::steady_clock::now();
    runSuperpixels(contourMask, labels, num_superpixels);
    timeEnd = std::chrono::steady_clock::now();
    int64_t tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count();
    double tracking_time_sec = tracking_time / 1.0e6; 
    ROS_INFO_STREAM("Superpixels took: " << tracking_time_sec << " seconds");

    cv::Mat overlaidContours;
    overlayContoursWithMeans(overlaidContours, contourMask, labels, num_superpixels);

    // Overlay contours on top of image
    // cv::Mat overlay;
    // color_image_ptr_->image.copyTo(overlay);
    overlaidContours.setTo(cv::Scalar(0, 0, 0), contourMask);


    color_image_ptr_->header.stamp = ros::Time::now();

    overlay_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    overlay_image_ptr_->header = color_image_ptr_->header;
    overlay_image_ptr_->encoding = color_image_ptr_->encoding;
    overlay_image_ptr_->image = overlaidContours;

    // ROS_INFO_STREAM("contourMask middle pixel: " << contourMask.at<uint8_t>(contourMask.rows/2, contourMask.cols/2));

    color_image_pub_.publish(color_image_ptr_->toImageMsg());
    overlay_image_pub_.publish(overlay_image_ptr_->toImageMsg());

    return;
}

void SuperpixelColorOpenCVSegmenter::overlayContoursWithMeans(cv::Mat & overlaidContours, const cv::Mat & contourMask, const cv::Mat & labels, const int & num_superpixels)
{
    std::vector<cv::Vec3i> means_int(num_superpixels, cv::Vec3i(0, 0, 0));
    std::vector<cv::Vec3b> means_byte(num_superpixels, cv::Vec3b(0, 0, 0));
    std::vector<int> counts(num_superpixels, 0);

    for (int r = 0; r < labels.rows; r++)
    {
        for (int c = 0; c < labels.cols; c++)
        {
            // ROS_INFO_STREAM("[" << r << ", " << c << "]");
            cv::Vec3b color = color_image_ptr_->image.at<cv::Vec3b>(r, c);
            cv::Vec3i color_int = color;
            // ROS_INFO_STREAM("       color: " << color);
            const int label = labels.at<int>(r, c);
            // ROS_INFO_STREAM("       label: " << label);

            // ROS_INFO_STREAM("       pre-counts[" << label << "]: " << counts[label]);
            // ROS_INFO_STREAM("       pre-means_int[" << label << "]: " << means_int[label]);            

            counts[label] = counts[label] + 1;
            // ROS_INFO_STREAM("       counts[" << label << "]: " << counts[label]);
            cv::Vec3i mult = means_int[label] * (counts[label] - 1);
            // ROS_INFO_STREAM("       mult: " << mult);
            cv::Vec3i num = mult + color_int;
            // ROS_INFO_STREAM("       num: " << num);
            means_int[label] = (num) / counts[label];
            // ROS_INFO_STREAM("       means_int[" << label << "]: " << means_int[label]);            
        }
    }

    // ROS_INFO_STREAM("means:");
    for (int i = 0; i < num_superpixels; i++)
    {
        means_byte[i] = means_int[i];
        // ROS_INFO_STREAM("   [" << i << "]: " << means[i]);
    }

    overlaidContours = cv::Mat::zeros(labels.size(), CV_8UC3);
    for (int r = 0; r < labels.rows; r++)
    {
        for (int c = 0; c < labels.cols; c++)
        {
            const int label = labels.at<int>(r, c);
            overlaidContours.at<cv::Vec3b>(r, c) = means_byte[label];
        }
    }
}

void SuperpixelColorOpenCVSegmenter::runSuperpixels(cv::Mat & contourMask, cv::Mat & labels, int & num_superpixels)
{
    // Run superpixels algorithm
    slic_->iterate(num_iterations_);

    slic_->getLabelContourMask(contourMask, true);

    slic_->getLabels(labels);
    
    num_superpixels = slic_->getNumberOfSuperpixels();

    return;
}

double SuperpixelColorOpenCVSegmenter::calculateDistance()
{
    return 0.0;
}