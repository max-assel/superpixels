#pragma once

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

#include <yaml-cpp/yaml.h>

class SuperpixelDepthSegmenter 
{
    public:
        SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path);

        /////////////
        // SEGMENT //
        /////////////
        void run();

        ///////////////
        // VISUALIZE //
        ///////////////
        void visualize();

    private:
        /////////////
        // SEGMENT //
        /////////////
        void preprocessImages();

        void calculateStep(const cv::Mat & depth_image);

        void init_data(const cv::Mat & depth_image,
                        const cv::Mat & label_image,
                        const cv::Mat & normal_image);

        cv::Point findLocalMinimum(const cv::Mat & depth_image, 
                                    const cv::Mat & label_image,
                                    const cv::Mat & normal_image,
                                    const cv::Point & og_center);

        void generateSuperpixels(const cv::Mat & depth_image,
                                    const cv::Mat & label_image,
                                    const cv::Mat & normal_image);

        double computeDistance(const int & center_idx, 
                                const float & depth,
                                const uint8_t & label,
                                const cv::Vec3b & normal,
                                const cv::Point & pixel);

        void dilate_depth_image(const cv::Mat & image, cv::Mat & dilated_image);

        void checkSparsity();

        bool notReceivedImage();

        bool notReceivedDepthImage();

        bool notReceivedLabelImage();

        bool notReceivedNormalImage();

        bool isPixelInBounds(const cv::Mat & image, const cv::Point & pixel);

        bool isPixelValid(const cv::Mat & depth_image, 
                            const cv::Mat & label_image,
                            const cv::Mat & normal_image,
                            const cv::Point & pixel);

        void allImageCallback(const sensor_msgs::ImageConstPtr& depth_image, 
                                const sensor_msgs::ImageConstPtr& label_image, 
                                const sensor_msgs::ImageConstPtr& normal_image);

        ///////////////
        // VISUALIZE //
        ///////////////
        void displayCenterGrid(cv::Mat & image, const cv::Vec3b & color);

        void convertDepthImageToColor();

        void overlayCenters();

        ros::NodeHandle nh_;
        
        // Subscribers
        image_transport::SubscriberFilter raw_depth_img_sub_;
        image_transport::SubscriberFilter raw_label_img_sub_;
        image_transport::SubscriberFilter raw_normal_img_sub_;

        // Publishers
        image_transport::Publisher fin_depth_img_pub_;
        image_transport::Publisher center_grid_img_pub_;

        // Image message pointers
        sensor_msgs::ImageConstPtr raw_depth_img_msg_ = nullptr;
        sensor_msgs::ImageConstPtr raw_label_img_msg_ = nullptr;
        sensor_msgs::ImageConstPtr raw_normal_img_msg_ = nullptr;

        // Image pointers
        cv_bridge::CvImagePtr depth_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr normal_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_depth_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_normal_img_ptr_ = nullptr;

        // cv_bridge::CvImagePtr cluster_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr center_grid_img_ptr_ = nullptr;

        // Synchronizer
        using MsgSynchronizer = message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>; //  
        boost::shared_ptr<MsgSynchronizer> msg_sync_;

        // Mutex
        std::mutex img_mutex_;

        // Flags
        bool initialized_ = false;

        // Superpixel matrices and vector
        cv::Mat clusters_; // per-pixel cluster assignments
        cv::Mat distances_; // per-pixel distances to cluster center
        std::vector<std::vector<double>> centers_; // LAB/xy cluster centers
        std::vector<int> center_counts_; // Number of occurrences of each center

        cv::Mat color_depth_image_;

        // Parameters
        struct SuperpixelParams
        {
            int num_superpixels_ = 0; // Desired number of approximately equally-sized superpixels
            int step_ = 0; // superpixel grid interval
            int n_c_ = 0; // Color parameter
            int n_s_ = 0; // Spatial parameter
            int num_iterations_ = 0; // Number of iterations
            bool warm_start_ = false; // Warm start
        };

        SuperpixelParams params_;

        float DELTA = std::numeric_limits<float>::epsilon();

};