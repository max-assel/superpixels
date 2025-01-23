#include <ros/init.h>
#include <ros/node_handle.h>

#include <superpixels/SuperpixelDepthSegmenter.h>

void ros_throw_if(const bool & condition, const std::string & message)
{
    if (condition)
    {
        ROS_ERROR_STREAM(message);
        throw std::runtime_error(message);
    }
}

void ros_throw_param_load(const ros::NodeHandle & nh, const std::string & param_name, std::string & param)
{
    return ros_throw_if( !nh.getParam(param_name, param), "Couldn't find parameter: " + param_name);
}

void ros_throw_param_load(const ros::NodeHandle & nh, const std::string & param_name, bool & param)
{
    return ros_throw_if( !nh.getParam(param_name, param), "Couldn't find parameter: " + param_name);
}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "superpixel_depth_node");
    ros::NodeHandle nh;

    ros::Rate loop_rate(30.0); // 30 Hz

    std::string config_path;
    ros_throw_param_load(nh, "/config_path", config_path);

    SuperpixelDepthSegmenter * superpixel_segmenter = new SuperpixelDepthSegmenter(nh, config_path);

    while (ros::ok())
    {
        // Do something
        superpixel_segmenter->run();

        // Visualize outputs
        superpixel_segmenter->visualize();
    
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}