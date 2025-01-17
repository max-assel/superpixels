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

bool SuperpixelSegmenter::notReceivedDepthImage()
{
    return (depth_image_msg_ == nullptr);
}

bool SuperpixelSegmenter::notReceivedLabelImage()
{
    return (label_image_msg_ == nullptr);
}

bool SuperpixelSegmenter::notReceivedNormalImage()
{
    return (normal_image_msg_ == nullptr);
}

bool SuperpixelSegmenter::notReceivedImage()
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


void SuperpixelSegmenter::runSegmentation()
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

    // Sanity check middle pixel
    ROS_INFO_STREAM("Depth image size --- rows: " << depth_image_ptr_->image.rows << ", cols: " << depth_image_ptr_->image.cols);
    ROS_INFO_STREAM("Label image size --- rows: " << label_image_ptr_->image.rows << ", cols: " << label_image_ptr_->image.cols);
    ROS_INFO_STREAM("Normal image size --- rows: " << normal_image_ptr_->image.rows << ", cols: " << normal_image_ptr_->image.cols);

    return;
}