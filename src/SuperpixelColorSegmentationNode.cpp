#include <ros/init.h>
#include <ros/node_handle.h>

#include <superpixels/SuperpixelColorSegmenter.h>

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "superpixel_color_node");
    ros::NodeHandle nh;

    SuperpixelSegmenter * superpixel_segmenter = new SuperpixelColorSegmenter(nh);

    ros::Rate loop_rate(30.0); // 30 Hz

    while (ros::ok())
    {
        // Do something
        superpixel_segmenter->runSegmentation();
    
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}