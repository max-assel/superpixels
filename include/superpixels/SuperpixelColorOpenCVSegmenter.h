// #include <ros/node_handle.h>
// #include <ros/package.h>
#include "rclcpp/rclcpp.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <yaml-cpp/yaml.h>

#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc/seeds.hpp>

/**
* \brief Base class for superpixel segmenter
*/
class SuperpixelColorOpenCVSegmenter
{
    public:
        SuperpixelColorOpenCVSegmenter(ros::NodeHandle nh, const std::string & config_path);

        void runSegmentation();

    private:
        void runSuperpixels(); // cv::Mat & contourMask, cv::Mat & labels, int & num_superpixels

        double calculateDistance();

        void overlayContoursWithMeans(); // cv::Mat & overlaidContours

        ros::NodeHandle nh_;

        cv_bridge::CvImagePtr color_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr overlay_image_ptr_ = nullptr;

        image_transport::Publisher color_image_pub_;
        image_transport::Publisher overlay_image_pub_;

        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic_;

        int num_iterations_ = 0;

        cv::Mat contourMask;
        cv::Mat labels;
        cv::Mat overlaidContours;
        int num_superpixels = 0;        

};