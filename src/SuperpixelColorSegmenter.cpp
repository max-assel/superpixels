#include <superpixels/SuperpixelColorSegmenter.h>

SuperpixelColorSegmenter::SuperpixelColorSegmenter(ros::NodeHandle nh) : SuperpixelSegmenter(nh), nh_(nh) {}

void SuperpixelColorSegmenter::runSegmentation()
{
    return;
}

double SuperpixelColorSegmenter::calculateDistance()
{
    return 0.0;
}