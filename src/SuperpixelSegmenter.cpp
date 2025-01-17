#include <superpixels/SuperpixelSegmenter.h>

SuperpixelSegmenter::SuperpixelSegmenter(ros::NodeHandle nh) : nh_(nh), it_(nh)
{
    std::string depth_image_topic = "/camera/depth/image_raw";
    std::string label_image_topic = "/camera/steppability/labels";
    std::string normal_image_topic = "/camera/normals";

    nh_.getParam("depth_image_topic", depth_image_topic);
    nh_.getParam("label_image_topic", label_image_topic);
    nh_.getParam("normal_image_topic", normal_image_topic);

    depth_image_sub_.subscribe(it_, depth_image_topic, 3);
    label_image_sub_.subscribe(it_, label_image_topic, 3);
    normal_image_sub_.subscribe(it_, normal_image_topic, 3);

    msg_sync_ = boost::make_shared<MsgSynchronizer>(depth_image_sub_, label_image_sub_, normal_image_sub_, 10);
    msg_sync_->registerCallback(boost::bind(&SuperpixelSegmenter::allImageCallback, this, _1, _2, _3));
}

void SuperpixelSegmenter::allImageCallback(const sensor_msgs::ImageConstPtr& depth_image, 
                                                const sensor_msgs::ImageConstPtr& label_image, 
                                                const sensor_msgs::ImageConstPtr& normal_image)
{
    // Do something with the images
    ROS_INFO_STREAM("[SuperpixelSegmenter::allImageCallback]");

    return;
}

void SuperpixelSegmenter::runSegmentation()
{
    ros::spin();
}