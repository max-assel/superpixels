// #include <ros/node_handle.h>
// #include <ros/package.h>
#include "rclcpp/rclcpp.hpp"

// #include <superpixels/SuperpixelSegmenter.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

// #include <yaml-cpp/yaml.h>

/**
* \brief Base class for superpixel segmenter
*/
class SuperpixelColorSegmenter
{
    public:
        SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path);

        void run();

        void visualize();

    private:

        void reset_data();

        void init_data();

        void preprocessing(); // cv::Mat & lab_image

        void generateSuperpixels();

        void createConnectivity();

        double computeDistance(const int & center_idx, const cv::Vec3b & color, const cv::Point & pixel);

        cv::Point findLocalMinimum(const cv::Mat & image, const cv::Point & center);

        void displayCenterGrid(cv::Mat & image, const cv::Vec3b & color);

        void displayContours(cv::Mat & contours);

        void displaySuperpixelsWithClusterMeans(cv::Mat & overlaid_image);

        ros::NodeHandle nh_;

        cv::Mat lab_image;

        cv_bridge::CvImagePtr color_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr center_grid_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr overlay_image_ptr_ = nullptr;

        image_transport::Publisher color_image_pub_;
        image_transport::Publisher center_grid_image_pub_;
        image_transport::Publisher overlay_image_pub_;

        cv::Mat clusters_; // per-pixel cluster assignments
        cv::Mat distances_; // per-pixel distances to cluster center
        std::vector<std::vector<double>> centers_; // LAB/xy cluster centers
        std::vector<int> center_counts_; // Number of occurrences of each center

        struct SuperpixelParams
        {
            int num_superpixels_ = 0; // Desired number of approximately equally-sized superpixels
            int step_ = 0; // superpixel grid interval
            double n_c_ = 0; // Color parameter
            double n_s_ = 0; // Spatial parameter
            int num_iterations_ = 0; // Number of iterations
            bool warm_start_ = false; // Warm start
        };

        SuperpixelParams params_;

};