#include <ros/node_handle.h>

#include <superpixels/SuperpixelSegmenter.h>

/**
* \brief Base class for superpixel segmenter
*/
class SuperpixelColorSegmenter : public SuperpixelSegmenter
{
    public:
        SuperpixelColorSegmenter(ros::NodeHandle nh);

        void runSegmentation();

    private:
        double calculateDistance();

        ros::NodeHandle nh_;
};