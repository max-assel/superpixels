#pragma once

#include <image_transport/subscriber_filter.h>
#include <tf2_ros/message_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class SuperpixelSegmenter
{
    public:
        SuperpixelSegmenter(ros::NodeHandle nh);

        void runSegmentation();

    private:
        void allImageCallback(const sensor_msgs::ImageConstPtr& depth_image, 
                                const sensor_msgs::ImageConstPtr& label_image, 
                                const sensor_msgs::ImageConstPtr& normal_image);

        ros::NodeHandle nh_;

        image_transport::ImageTransport it_;
        
        image_transport::SubscriberFilter depth_image_sub_;
        image_transport::SubscriberFilter label_image_sub_;
        image_transport::SubscriberFilter normal_image_sub_;

        using MsgSynchronizer = message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>; //  
        boost::shared_ptr<MsgSynchronizer> msg_sync_;


};