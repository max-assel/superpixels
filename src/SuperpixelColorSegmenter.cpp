#include <superpixels/SuperpixelColorSegmenter.h>

SuperpixelColorSegmenter::SuperpixelColorSegmenter(ros::NodeHandle nh) : SuperpixelSegmenter(nh), nh_(nh) 
{
    image_transport::ImageTransport it(nh);

    color_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    // read in example image
    const std::string superixels_path = ros::package::getPath("superpixels");
    const std::string image_name = "flower.jpg";

    color_image_ptr_->header.stamp = ros::Time::now();
    color_image_ptr_->header.frame_id = "camera_link";
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

void SuperpixelColorSegmenter::runSegmentation()
{
    color_image_ptr_->header.stamp = ros::Time::now();

    color_image_pub_.publish(color_image_ptr_->toImageMsg());

    return;
}

double SuperpixelColorSegmenter::calculateDistance()
{
    return 0.0;
}