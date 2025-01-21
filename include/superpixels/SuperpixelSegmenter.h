#include <ros/node_handle.h>

/**
* \brief Base class for superpixel segmenter
*/
class SuperpixelSegmenter 
{
    public:
        SuperpixelSegmenter(ros::NodeHandle nh) {}

        virtual void runSegmentation() = 0;

    private:
        virtual double calculateDistance() = 0;
};