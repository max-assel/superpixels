#include <ros/node_handle.h>
#include <ros/package.h>

#include <superpixels/SuperpixelSegmenter.h>

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
class SuperpixelColorSegmenter : public SuperpixelSegmenter
{
    public:
        SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path);

        void runSegmentation();

    private:
        void runSuperpixels();

        double calculateDistance();

        ros::NodeHandle nh_;

        cv_bridge::CvImagePtr color_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr superpixel_label_image_ptr_ = nullptr;
        cv_bridge::CvImagePtr superpixel_distance_image_ptr_ = nullptr;

        image_transport::Publisher color_image_pub_;

        struct SuperpixelParams
        {
            int k_ = 0; // Desired number of approximately equally-sized superpixels
        };

        SuperpixelParams params_;

};