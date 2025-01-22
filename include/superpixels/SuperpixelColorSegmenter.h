#include <ros/node_handle.h>
#include <ros/package.h>

// #include <superpixels/SuperpixelSegmenter.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <yaml-cpp/yaml.h>

/**
* \brief Base class for superpixel segmenter
*/
class SuperpixelColorSegmenter
{
    public:
        SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path);

        void run();

        void displayCenterGrid(cv::Mat & image, const cv::Vec3b & color);

        void displayContours(cv::Mat & image, const cv::Vec3b & color);

        void displaySuperpixelsWithClusterMeans(cv::Mat & image);

    private:
        void generateSuperpixels(const cv::Mat & image);

        void createConnectivity(const cv::Mat & image);

        void preprocessing(cv::Mat & lab_image);

        double computeDistance(const int & center_idx, const cv::Vec3b & color, const cv::Point & pixel);

        cv::Point findLocalMinimum(const cv::Mat & image, const cv::Point & center);

        void clear_data();

        void init_data(const cv::Mat & image);

        ros::NodeHandle nh_;

        cv_bridge::CvImagePtr color_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr center_grid_image_ptr_ = nullptr;
        // cv_bridge::CvImagePtr superpixel_label_image_ptr_ = nullptr;
        // cv_bridge::CvImagePtr superpixel_distance_image_ptr_ = nullptr;

        image_transport::Publisher color_image_pub_;
        image_transport::Publisher center_grid_image_pub_;

        // Adapted from: https://github.com/PSMM/SLIC-Superpixels/tree/master
        cv::Mat clusters_; // per-pixel cluster assignments
        cv::Mat distances_; // per-pixel distances to cluster center
        std::vector<std::vector<double>> centers_; // LAB/xy cluster centers
        std::vector<int> center_counts_; // Number of occurrences of each center

        struct SuperpixelParams
        {
            int num_superpixels_ = 0; // Desired number of approximately equally-sized superpixels
            int step_ = 0; // superpixel grid interval
            int n_c_ = 0; // Color parameter
            int n_s_ = 0; // Spatial parameter
        };

        SuperpixelParams params_;

};