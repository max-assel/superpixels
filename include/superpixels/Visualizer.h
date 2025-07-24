#pragma once

#include <superpixels/utils.h>

// #include <ros/package.h>

#include <opencv2/opencv.hpp>

// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <convex_plane_decomposition_msgs/msg/planar_terrain.hpp>
#include <convex_plane_decomposition/PlanarRegion.h>
#include <convex_plane_decomposition_ros/MessageConversion.h>

#include <grid_map_ros/GridMapRosConverter.hpp>

class Visualizer
{
    public:
        Visualizer(const SuperpixelParams & params, const rclcpp::Node::SharedPtr & node);

        ///////////////
        // VISUALIZE //
        ///////////////
        void visualize(const cv::Mat & depth_image,
                        const cv::Mat & normal_image,
                        const cv_bridge::CvImagePtr & raw_depth_img_ptr,
                        const cv_bridge::CvImagePtr & raw_normal_img_ptr,
                        const std::vector<std::vector<double>> & centers,
                        const cv::Mat & clusters,
                        const std::vector<int> & center_counts,
                        const std::vector<std::vector<Eigen::Vector2d>> & superpixel_projections,
                        const std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                        const std::vector<Eigen::Matrix3d> & superpixel_rotations);

        void setParams(const SuperpixelParams & params);

        void setColors();

        void outputToDatFile(const cv_bridge::CvImagePtr & raw_depth_img_ptr,
                                const std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls);

    private:
        void publishPlanarRegions(const cv::Mat & depth_img, 
                                    const std::vector<std::vector<double>> & centers,
                                    const std::vector<int> & center_counts,
                                    const std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                                    const std::vector<Eigen::Matrix3d> & superpixel_rotations);  


        void colorCentroids(const std::vector<std::vector<double>> & centers,
                            const std::vector<int> & center_counts);

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
        rclcpp::Node::SharedPtr node_;

        // Publishers
        image_transport::Publisher fin_depth_img_pub_;
        // image_transport::Publisher fin_label_img_pub_;
        image_transport::Publisher fin_normal_img_pub_;
        image_transport::Publisher center_grid_img_pub_;
        image_transport::Publisher colored_cluster_img_pub_;        

        // ros::Publisher colored_point_cloud_pub_;
        // ros::Publisher colored_centroids_pub_;
        // ros::Publisher terrainPub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_point_cloud_pub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr colored_centroids_pub_;
        rclcpp::Publisher<convex_plane_decomposition_msgs::msg::PlanarTerrain>::SharedPtr terrainPub_;

        // cv_bridge::CvImagePtr cluster_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr center_grid_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr colored_cluster_img_ptr_ = nullptr;

        cv_bridge::CvImagePtr fin_depth_img_ptr_ = nullptr;
        // cv_bridge::CvImagePtr fin_label_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_normal_img_ptr_ = nullptr;
        cv_bridge::CvImagePtr fin_normal_img_colored_ptr_ = nullptr;

        std::vector<cv::Scalar> colors_;

        std::shared_ptr<tf2_ros::TransformListener> tfListener_{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tfBuffer_;

};