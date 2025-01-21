#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path) : SuperpixelSegmenter(nh, config_path), nh_(nh)
{
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

double SuperpixelDepthSegmenter::calculateDistance()
{
    return 0.0;
}

void SuperpixelDepthSegmenter::runSegmentation()
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

    int rows = depth_image_ptr_->image.rows;
    int cols = depth_image_ptr_->image.cols;

    // Sanity check middle pixel
    ROS_INFO_STREAM("Depth image size --- rows: " << depth_image_ptr_->image.rows << ", cols: " << depth_image_ptr_->image.cols);
    ROS_INFO_STREAM("Label image size --- rows: " << label_image_ptr_->image.rows << ", cols: " << label_image_ptr_->image.cols);
    ROS_INFO_STREAM("Normal image size --- rows: " << normal_image_ptr_->image.rows << ", cols: " << normal_image_ptr_->image.cols);

    int non_nan_depth_count = 0;
    int non_nan_label_count = 0;
    int non_nan_normal_count = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (depth_image_ptr_->image.at<float>(r, c) != depth_image_ptr_->image.at<float>(r, c))
            {
                // ROS_ERROR_STREAM("Depth image has NaN value at row: " << r << ", col: " << c);
                // return;
            } else
            {
                // ROS_INFO_STREAM("Depth image value at row: " << r << ", col: " << c << " is: " << depth_image_ptr_->image.at<float>(r, c));
                non_nan_depth_count++;
            }

            if (label_image_ptr_->image.at<uint8_t>(r, c) != label_image_ptr_->image.at<uint8_t>(r, c))
            {
                // ROS_ERROR_STREAM("Label image has NaN value at row: " << r << ", col: " << c);
                // return;
            } else
            {
                non_nan_label_count++;
            }

            if (normal_image_ptr_->image.at<cv::Vec3f>(r, c)[0] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[0] ||
                normal_image_ptr_->image.at<cv::Vec3f>(r, c)[1] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[1] ||
                normal_image_ptr_->image.at<cv::Vec3f>(r, c)[2] != normal_image_ptr_->image.at<cv::Vec3f>(r, c)[2])
            {
                // ROS_ERROR_STREAM("Normal image has NaN value at row: " << r << ", col: " << c);
                // return;
            } else
            {
                // ROS_INFO_STREAM("Normal image value at row: " << r << ", col: " << c << " is: " << normal_image_ptr_->image.at<cv::Vec3f>(r, c));
                non_nan_normal_count++;
            }
        }
    }

    int total_pixels = M_PI * (rows / 2) * (cols / 2);

    ROS_INFO_STREAM("Depth image sparsity ratio: " << float(non_nan_depth_count) / total_pixels);
    ROS_INFO_STREAM("Label image sparsity ratio: " << float(non_nan_label_count) / total_pixels);
    ROS_INFO_STREAM("Normal image sparsity ratio: " << float(non_nan_normal_count) / total_pixels);

    return;
}