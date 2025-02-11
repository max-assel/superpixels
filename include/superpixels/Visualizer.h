#pragma once

#include <superpixels/utils.h>

#include <opencv2/opencv.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <convex_plane_decomposition_msgs/PlanarTerrain.h>
#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition_ros/MessageConversion.h>

#include <grid_map_ros/GridMapRosConverter.hpp>

class Visualizer
{
    public:
        Visualizer(const SuperpixelParams & params, ros::NodeHandle nh);

        ///////////////
        // VISUALIZE //
        ///////////////
        void visualize(const cv::Mat & depth_image,
                        const cv::Mat & label_image,
                        const cv::Mat & normal_image,
                        const cv_bridge::CvImagePtr & raw_depth_img_ptr,
                        const cv_bridge::CvImagePtr & raw_label_img_ptr,
                        const cv_bridge::CvImagePtr & raw_normal_img_ptr,
                        const std::vector<std::vector<double>> & centers,
                        const cv::Mat & clusters);

        void setParams(const SuperpixelParams & params);

    private:
        void colorCentroids(const std::vector<std::vector<double>> & centers);

        void publishPlanarRegions(const cv::Mat & depth_img, 
                                    const std::vector<std::vector<double>> & centers);  

        void colorClusters(const cv::Mat & color_depth_image,
                            const cv::Mat & clusters);

        void colorClusterPointCloud(const cv::Mat & depth_image, 
                                    const cv::Mat & clusters);    

        void displayCenterGrid(cv::Mat & image, 
                                const cv::Vec3b & color, 
                                const std::vector<std::vector<double>> & centers);

        void convertDepthImageToColor(cv::Mat & color_depth_image, 
                                        const cv::Mat & depth_image);

        void overlayCenters(const cv::Mat & color_depth_image, 
                            const std::vector<std::vector<double>> & centers);

        SuperpixelParams params_;

        // Publishers
        image_transport::Publisher fin_depth_img_pub_;
        image_transport::Publisher fin_label_img_pub_;
        image_transport::Publisher fin_normal_img_pub_;
        image_transport::Publisher center_grid_img_pub_;
        image_transport::Publisher colored_cluster_img_pub_;        

        ros::Publisher colored_point_cloud_pub_;
        ros::Publisher colored_centroids_pub_;
        ros::Publisher terrainPub_;

        // cv_bridge::CvImagePtr cluster_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr center_grid_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr colored_cluster_img_ptr_ = nullptr;

        cv_bridge::CvImagePtr fin_depth_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_normal_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_normal_img_colored_ptr_ = nullptr;

        std::vector<cv::Scalar> colors_;
};