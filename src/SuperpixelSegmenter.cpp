#include <superpixels/SuperpixelSegmenter.h>

SuperpixelSegmenter::SuperpixelSegmenter(ros::NodeHandle nh) : nh_(nh), it_(nh)
{
    std::string depth_image_topic =  "/egocylinder/floor_image";
    std::string label_image_topic =  "/egocylinder/floor_labels";
    std::string normal_image_topic = "/egocylinder/floor_normals";

    nh_.getParam("depth_image_topic", depth_image_topic);
    nh_.getParam("label_image_topic", label_image_topic);
    nh_.getParam("normal_image_topic", normal_image_topic);

    depth_image_sub_.subscribe(it_, depth_image_topic, 3);
    label_image_sub_.subscribe(it_, label_image_topic, 3);
    normal_image_sub_.subscribe(it_, normal_image_topic, 3);

    msg_sync_ = boost::make_shared<MsgSynchronizer>(depth_image_sub_, label_image_sub_, normal_image_sub_, 10);
    msg_sync_->registerCallback(boost::bind(&SuperpixelSegmenter::allImageCallback, this, _1, _2, _3));
}

void SuperpixelSegmenter::allImageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg, 
                                            const sensor_msgs::ImageConstPtr& label_image_msg, 
                                            const sensor_msgs::ImageConstPtr& normal_image_msg)
{   
    std::lock_guard<std::mutex> lock(img_mutex_);

    // ROS_INFO_STREAM("[SuperpixelSegmenter::allImageCallback]");
    // ROS_INFO_STREAM("       time stamp: " << depth_image->header.stamp);

    depth_image_msg_ = depth_image_msg;
    label_image_msg_ = label_image_msg;
    normal_image_msg_ = normal_image_msg;

    return;
}

void SuperpixelSegmenter::runSegmentation()
{
    std::lock_guard<std::mutex> lock(img_mutex_);

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

    // Sanity check middle pixel
    


    return;
}