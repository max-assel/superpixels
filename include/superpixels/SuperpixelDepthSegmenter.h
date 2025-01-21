#pragma once

#include <superpixels/SuperpixelSegmenter.h>

#include <image_transport/subscriber_filter.h>
#include <tf2_ros/message_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <math.h>

class SuperpixelDepthSegmenter : public SuperpixelSegmenter
{
    public:
        SuperpixelDepthSegmenter(ros::NodeHandle nh);

        void runSegmentation();

    private:
        double calculateDistance();

        bool notReceivedImage();

        bool notReceivedDepthImage();

        bool notReceivedLabelImage();

        bool notReceivedNormalImage();

        void allImageCallback(const sensor_msgs::ImageConstPtr& depth_image, 
                                const sensor_msgs::ImageConstPtr& label_image, 
                                const sensor_msgs::ImageConstPtr& normal_image);

        ros::NodeHandle nh_;

        // image_transport::ImageTransport it_;
        
        image_transport::SubscriberFilter depth_image_sub_;
        image_transport::SubscriberFilter label_image_sub_;
        image_transport::SubscriberFilter normal_image_sub_;

        sensor_msgs::ImageConstPtr depth_image_msg_ = nullptr;
        sensor_msgs::ImageConstPtr label_image_msg_ = nullptr;
        sensor_msgs::ImageConstPtr normal_image_msg_ = nullptr;

        cv_bridge::CvImagePtr depth_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr label_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr normal_image_ptr_ = nullptr;

        using MsgSynchronizer = message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>; //  
        boost::shared_ptr<MsgSynchronizer> msg_sync_;

        std::mutex img_mutex_;

};