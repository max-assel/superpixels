#pragma once

#include <image_transport/subscriber_filter.h>
#include <tf2_ros/message_filter.h>

#include <geometry_msgs/PoseStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <math.h>

#include <ocs2_ros_interfaces/visualization/VisualizationHelpers.h>

#include <dynamic_reconfigure/server.h>
#include <superpixels/ParametersConfig.h>

// #include <superpixels/utils.h>
#include <superpixels/ConvexHullifier.h>
#include <superpixels/ImagePreprocessor.h>
#include <superpixels/Visualizer.h>
#include <superpixels/Ransac.h>

class SuperpixelDepthSegmenter 
{
    public:
        SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path);
        ~SuperpixelDepthSegmenter();

        /////////////
        // SEGMENT //
        /////////////
        void run();

        void reconfigureCallback(superpixels::ParametersConfig &config, uint32_t level);

    private:
        void reset_data(const cv::Mat & depth_image,
                        // const cv::Mat & label_image,
                        const cv::Mat & normal_image);

        void init_data(const cv::Mat & depth_image,
                        // const cv::Mat & label_image,
                        const cv::Mat & normal_image);

        cv::Point findLocalMinimum(const cv::Mat & depth_image, 
                                    // const cv::Mat & label_image,
                                    const cv::Mat & normal_image,
                                    const cv::Point & og_center);

        void generateSuperpixels(const cv::Mat & depth_image,
                                    // const cv::Mat & label_image,
                                    const cv::Mat & normal_image);

        bool checkConstraints(const int & center_idx, 
                                const float & depth,
                                // const uint8_t & label,
                                const cv::Vec3f & normal,
                                const cv::Point & pixel);

        double computeDistance(const int & center_idx, 
                                const float & depth,
                                // const uint8_t & label,
                                const cv::Vec3f & normal,
                                const cv::Point & pixel);

        cv::Vec3f ransac(const std::vector<cv::Point> & pixels, const cv::Mat & depth_image, const cv::Vec3f & og_normal);

        // void dilate_img(const cv::Mat & image, cv::Mat & dilated_image);

        // void checkSparsity();

        bool notReceivedImage();

        bool notReceivedDepthImage();

        // bool notReceivedLabelImage();

        bool notReceivedNormalImage();

        void allImageCallback(const sensor_msgs::ImageConstPtr& depth_image, 
                                // const sensor_msgs::ImageConstPtr& label_image, 
                                const sensor_msgs::ImageConstPtr& normal_image);

        cv::Point findClosestPixel(const int & center_idx,
                                    const cv::Point & center, 
                                    const cv::Mat & depth_image,
                                    // const cv::Mat & label_image,
                                    const cv::Mat & normal_image);

        ros::NodeHandle nh_;
        
        // Subscribers
        image_transport::SubscriberFilter raw_depth_img_sub_;
        // image_transport::SubscriberFilter raw_label_img_sub_;
        image_transport::SubscriberFilter raw_normal_img_sub_;

        // Image message pointers
        sensor_msgs::ImageConstPtr raw_depth_img_msg_ = nullptr;
        // sensor_msgs::ImageConstPtr raw_label_img_msg_ = nullptr;
        sensor_msgs::ImageConstPtr raw_normal_img_msg_ = nullptr;

        // Image pointers
        cv_bridge::CvImagePtr raw_depth_img_ptr_ = nullptr;
        // cv_bridge::CvImagePtr raw_label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr raw_normal_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr prop_depth_img_ptr_ = nullptr;
        // cv_bridge::CvImagePtr prop_label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr prop_normal_img_ptr_ = nullptr;

        // Synchronizer
        using MsgSynchronizer = message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image>; // sensor_msgs::Image,   
        boost::shared_ptr<MsgSynchronizer> msg_sync_;

        // Mutex
        std::mutex img_mutex_;

        // Flags
        bool initialized_ = false;

        cv::Mat visited_; // Visited pixels

        // Superpixel matrices and vector
        cv::Mat clusters_; // per-pixel cluster assignments
        cv::Mat distances_; // per-pixel distances to cluster center
        std::vector<std::vector<double>> centers_; // LAB/xy cluster centers
        std::vector<int> center_counts_; // Number of occurrences of each center
        std::vector<std::vector<cv::Point>> superpixels_; // Superpixel pixel locations
        std::vector<std::vector<Eigen::Vector2d>> superpixel_convex_hulls_; // Superpixel convex hulls
        std::vector<Eigen::Matrix3d> egocan_to_region_rotations_; // Superpixel rotations

        SuperpixelParams params_;

        std::chrono::steady_clock::time_point cleanBegin, cleanEnd;
        std::chrono::steady_clock::time_point fillBegin, fillEnd;
        std::chrono::steady_clock::time_point preprocessBegin, preprocessEnd;
        std::chrono::steady_clock::time_point superpixelBegin, superpixelEnd;
        std::chrono::steady_clock::time_point convexHullBegin, convexHullEnd;


        ImagePreprocessor * imagePreprocessor_;
        Visualizer * visualizer_;
        Ransac * ransac_;
        ConvexHullifier * convexHullifier_;
};