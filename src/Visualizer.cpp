#include <superpixels/Visualizer.h>

Visualizer::Visualizer(const SuperpixelParams & params, ros::NodeHandle nh)
{
    params_ = params;

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);

    // Set up subscribers and publishers
    image_transport::ImageTransport it(nh);

    fin_depth_img_pub_ = it.advertise("/superpixels/process_depth", 1);
    fin_depth_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    // fin_label_img_pub_ = it.advertise("/superpixels/process_labels", 1);
    // fin_label_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    fin_normal_img_pub_ = it.advertise("/superpixels/process_normals", 1);
    fin_normal_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    fin_normal_img_colored_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    center_grid_img_pub_ = it.advertise("/superpixels/center_grid", 1);
    center_grid_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    colored_cluster_img_pub_ = it.advertise("/superpixels/colored_clusters", 1);
    colored_cluster_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    colored_point_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/superpixels/colored_point_cloud", 1);
    colored_centroids_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/superpixels/colored_centroids", 1);

    setColors();

    // set up terrain publisher
    terrainPub_ = nh.advertise<convex_plane_decomposition_msgs::PlanarTerrain>
                                    ("/convex_plane_decomposition_ros/planar_terrain", 1);
}

void Visualizer::setParams(const SuperpixelParams & params)
{
    params_ = params;
    setColors();
}

void Visualizer::setColors()
{
    colors_.clear();
    colors_.resize(params_.num_superpixels_);
    for (int i = 0; i < (int) colors_.size(); i++)
    {
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);

        // for (int j = 0; j < i; j++)
        // {
        //     if (cv::norm(color - colors_[j]) < 50)
        //     {
        //         color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        //         j = -1;
        //     }
        // }

        colors_[i] = color;
    }
}

void Visualizer::visualize(const cv::Mat & depth_image,
                            const cv::Mat & normal_image,
                            const cv_bridge::CvImagePtr & raw_depth_img_ptr,
                            const cv_bridge::CvImagePtr & raw_normal_img_ptr,
                            const std::vector<std::vector<double>> & centers,
                            const cv::Mat & clusters,
                            const std::vector<int> & center_counts,
                            const std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                            const std::vector<Eigen::Matrix3d> & egocan_to_region_rotations)
{
    // ROS_INFO_STREAM("   [Visualizer::visualize]");

    // std::lock_guard<std::mutex> lock(img_mutex_);

    // if (notReceivedImage())
    // {
    //     ROS_WARN("Not ready to visualize, no images received yet.");
    //     return;
    // }

    // ROS_INFO_STREAM("       Publishing final depth image");
    fin_depth_img_ptr_->header = raw_depth_img_ptr->header;
    fin_depth_img_ptr_->encoding = raw_depth_img_ptr->encoding;
    fin_depth_img_ptr_->image = depth_image;
    fin_depth_img_pub_.publish(fin_depth_img_ptr_->toImageMsg());

    // ROS_INFO_STREAM("       Preparing final label image");
    // fin_label_img_ptr_->header = raw_label_img_ptr->header;
    // fin_label_img_ptr_->encoding = raw_label_img_ptr->encoding;
    // fin_label_img_ptr_->image = label_image;
    // fin_label_img_pub_.publish(fin_label_img_ptr_->toImageMsg());

    // ROS_INFO_STREAM("       Preparing final normal image");
    fin_normal_img_ptr_->header = raw_normal_img_ptr->header;
    fin_normal_img_ptr_->encoding = raw_normal_img_ptr->encoding;
    fin_normal_img_ptr_->image = normal_image;
    // No publishing normal image

    // ROS_INFO_STREAM("       Publishing final colored normal image");
    fin_normal_img_colored_ptr_->header = fin_normal_img_ptr_->header;
    fin_normal_img_colored_ptr_->encoding = "rgb8";
    fin_normal_img_colored_ptr_->image = fin_normal_img_ptr_->image;
    fin_normal_img_colored_ptr_->image = cv::abs(fin_normal_img_colored_ptr_->image);
    fin_normal_img_colored_ptr_->image.convertTo(fin_normal_img_colored_ptr_->image, CV_8UC3, 255.0);
    fin_normal_img_pub_.publish(fin_normal_img_colored_ptr_->toImageMsg());

    cv::Mat color_depth_image = cv::Mat(depth_image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    convertDepthImageToColor(color_depth_image, depth_image);

    overlayCenters(color_depth_image, centers);

    colorClusters(color_depth_image, clusters);

    colorClusterPointCloud(depth_image, clusters);

    colorCentroids(centers, center_counts);

    publishPlanarRegions(depth_image, centers, center_counts, superpixel_convex_hulls, egocan_to_region_rotations);

    return;
}


void Visualizer::publishPlanarRegions(const cv::Mat & depth_img, 
                                        const std::vector<std::vector<double>> & centers,
                                        const std::vector<int> & center_counts,
                                        const std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                                        const std::vector<Eigen::Matrix3d> & egocan_to_region_rotations)
{
    ROS_INFO_STREAM("   [Visualizer::publishPlanarRegions]");

    convex_plane_decomposition_msgs::PlanarTerrain terrain_msg;

    ros::Time lookupTime = fin_depth_img_ptr_->header.stamp;
    std::string egocan_frame = fin_depth_img_ptr_->header.frame_id;

    geometry_msgs::TransformStamped egocanFrameToWorldFrame = 
        tfBuffer_.lookupTransform("world", egocan_frame, lookupTime);

    double foot_radius = 0.02;

    convex_plane_decomposition::PlanarRegion region;
    convex_plane_decomposition::BoundaryWithInset boundaryWithInset;
    convex_plane_decomposition::CgalPolygonWithHoles2d polygonWithHoles;
    convex_plane_decomposition::CgalPolygon2d polygon;
    convex_plane_decomposition::CgalPolygon2d inflated_polygon;
    convex_plane_decomposition::CgalPolygonWithHoles2d inflated_polygon_with_holes;
    convex_plane_decomposition_msgs::PlanarRegion region_msg;

    std::vector<convex_plane_decomposition::CgalPolygonWithHoles2d> insets(1);

    cv::Point center_pixel;
    float center_depth;
    cv::Vec3f centerEgocanCvPt;
    Eigen::Matrix3d regionToEgocanRotMat;
    Eigen::Quaterniond regionToEgocanQuat;
    Eigen::Matrix3d egocanToWorldRotMat;

    Eigen::VectorXd centerWorldPose;

    Eigen::Vector2d convexHullPt, convexHullDir, inflatedConvexHullPt;

    // ROS_INFO_STREAM("       planar regions:");
    for (int i = 0; i < centers.size(); i++)
    {        
        // ROS_INFO_STREAM("           i: " << i);

        if (center_counts[i] == 0)
        {
            ROS_WARN_STREAM("           Region " << i << " has no points.");
            continue;
        }

        if (superpixel_convex_hulls[i].size() <= 3)
        {
            ROS_WARN_STREAM("           Region " << i << " has less than or equal to 3 points.");
        }

        center_pixel = cv::Point(centers[i][0], centers[i][1]);
        center_depth = centers[i][2];

        pixelToEgocanFrame(centerEgocanCvPt, center_pixel, center_depth, params_.k_c_, params_.h_);

        Eigen::Vector3d centerEgocanPt(centerEgocanCvPt.val[0], centerEgocanCvPt.val[1], centerEgocanCvPt.val[2]);

        // ROS_INFO_STREAM("               center (pixel): " << centers[i][0] << ", " << centers[i][1]);
        // ROS_INFO_STREAM("               center (depth): " << centers[i][2]);
        // ROS_INFO_STREAM("               center (egocan frame): " << centerEgocanPt.transpose());
        // ROS_INFO_STREAM("               normal (egocan frame): " << centers[i][3] << ", " << centers[i][4] << ", " << centers[i][5]);

        regionToEgocanRotMat = egocan_to_region_rotations[i].transpose();
        regionToEgocanQuat = Eigen::Quaterniond(regionToEgocanRotMat);

        centerWorldPose = transformHelperPoseStamped(centerEgocanPt, regionToEgocanQuat, egocanFrameToWorldFrame);

        // get rotation matrix
        egocanToWorldRotMat = calculateRotationMatrix(centerWorldPose[3], centerWorldPose[4], centerWorldPose[5]);

        region.transformPlaneToWorld.translation() = centerWorldPose.head(3);
        region.transformPlaneToWorld.linear() = egocanToWorldRotMat;

        // ROS_INFO_STREAM("               translation: " << region.transformPlaneToWorld.translation().transpose());
        // ROS_INFO_STREAM("               rotation: " << region.transformPlaneToWorld.linear().row(0));
        // ROS_INFO_STREAM("                         " << region.transformPlaneToWorld.linear().row(1));
        // ROS_INFO_STREAM("                         " << region.transformPlaneToWorld.linear().row(2));


        // ROS_INFO_STREAM("               convex hull:");
        polygon.container().clear();
        inflated_polygon.container().clear();
        for (int j = 0; j < superpixel_convex_hulls[i].size(); j++)
        {
            convexHullPt = superpixel_convex_hulls[i][j];

            // normal polygon
            polygon.container().emplace_back(convexHullPt[0], convexHullPt[1]);
            // ROS_INFO_STREAM("           point " << j << ": " << polygon.container()[j].x() << ", " << polygon.container()[j].y());

            // inflated polygon
            double norm = convexHullPt.norm();
            convexHullDir = convexHullPt / norm;

            if (norm > foot_radius)
            {
                inflatedConvexHullPt = convexHullPt - foot_radius * convexHullDir;
                // ROS_INFO_STREAM("           inflated point " << j << ": " << foot[0] << ", " << foot[1]);
            } else
            {
                inflatedConvexHullPt = 0.5 * convexHullPt;
                // ROS_INFO_STREAM("           inflated point " << j << ": " << foot[0] << ", " << foot[1]);
            }
            inflated_polygon.container().emplace_back(inflatedConvexHullPt[0], inflatedConvexHullPt[1]);
            // ROS_INFO_STREAM("           inflated point " << j << ": " << inflated_polygon.container()[j].x() << ", " << inflated_polygon.container()[j].y());
        }

        polygonWithHoles.outer_boundary() = polygon;
        boundaryWithInset.boundary = polygonWithHoles;

        inflated_polygon_with_holes.outer_boundary() = inflated_polygon;

        insets[0] = inflated_polygon_with_holes;

        boundaryWithInset.insets = insets;

        region.boundaryWithInset = boundaryWithInset;
        region.bbox2d = boundaryWithInset.boundary.outer_boundary().bbox();

        region_msg = convex_plane_decomposition::toMessage(region);
        cv::Scalar color = colors_[i];
        std_msgs::ColorRGBA region_color;
        region_color.r = color[2] / 255.0;
        region_color.g = color[1] / 255.0;
        region_color.b = color[0] / 255.0;
        region_color.a = 1.0;
        region_msg.color = region_color;

        terrain_msg.planarRegions.push_back(region_msg);

        // ROS_INFO_STREAM("   e0: " << e0.transpose());
        // ROS_INFO_STREAM("   e1: " << e1.transpose());
        // ROS_INFO_STREAM("   normal: " << normal.transpose());
    }    

    // placeholder gridMap
    grid_map::GridMap grid_map;
    grid_map::Length grid_map_dimensions(1.0, 1.0); // lengths in x,y directions [m]
    double grid_map_resolution = 0.1; // resolution [m]
    grid_map::Position grid_map_origin(0.0, 0.0); // origin [m]
    grid_map.setGeometry(grid_map_dimensions, 
                            grid_map_resolution, 
                            grid_map_origin);
    grid_map.add("elevation", 0.0); // add layer with value to initialize to everywhere'
    grid_map.setFrameId("odom");

    grid_map_msgs::GridMap grid_map_msg;
    grid_map::GridMapRosConverter::toMessage(grid_map, grid_map_msg);
    terrain_msg.gridmap = grid_map_msg; 

    terrainPub_.publish(terrain_msg);
}


void Visualizer::overlayCenters(const cv::Mat & color_depth_image, const std::vector<std::vector<double>> & centers)
{
    // overlay center grid on color version of depth image
    cv::Mat overlaid_image = color_depth_image.clone();

    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(overlaid_image, color, centers);

    center_grid_img_ptr_->header = fin_depth_img_ptr_->header;
    // center_grid_img_ptr_->header.stamp = ros::Time::now();
    center_grid_img_ptr_->encoding = sensor_msgs::image_encodings::BGR8;

    center_grid_img_ptr_->image = overlaid_image;
    center_grid_img_pub_.publish(center_grid_img_ptr_->toImageMsg());
}

void Visualizer::convertDepthImageToColor(cv::Mat & color_depth_image, const cv::Mat & depth_image)
{
    double min_depth = 0.0, max_depth = 0.0;
    cv::minMaxLoc(depth_image, &min_depth, &max_depth);

    // ROS_INFO_STREAM("Converting to 8UC3...");

    for (int r = 0; r < color_depth_image.rows; r++)
    {
        for (int c = 0; c < color_depth_image.cols; c++)
        {
            float depth = depth_image.at<float>(r, c);

            if (std::isnan(depth) || std::abs(depth) < 1e-6)
            {
                continue;
            }

            // ROS_INFO_STREAM("   (r, c): (" << r << ", " << c << ")");
            // ROS_INFO_STREAM("       depth: " << depth);

            int quantized_depth = (int) (depth * 255.0 / max_depth); // just scaling by max depth in image. If we do full max depth than image is really hard to see.

            cv::Vec3b color = cv::Vec3b(quantized_depth, quantized_depth, quantized_depth);
            color_depth_image.at<cv::Vec3b>(r, c) = color;
        }
    }    
}

void Visualizer::displayCenterGrid(cv::Mat & image, const cv::Vec3b & color, const std::vector<std::vector<double>> & centers)
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::displayCenterGrid]");
    
    // Display center grid
    for (int i = 0; i < (int) centers.size(); i++) 
    {
        // ROS_INFO_STREAM("       center[" << i << "]: (" << centers[i][0] << ", " << centers[i][1] << ")");
        cv::circle(image, cv::Point(centers[i][0], centers[i][1]), 2, color, -1);
    }

    return;
}

void Visualizer::colorClusters(const cv::Mat & color_depth_image,
                                const cv::Mat & clusters)
{
    // ROS_INFO_STREAM("   [Visualizer::colorClusters]");
    // overlay center grid on color version of depth image
    cv::Mat color_cluster_image = color_depth_image.clone();

    // build ector of random colors for clusters
    // std::vector<cv::Scalar> colors(centers.size());
    // for (int i = 0; i < (int) colors.size(); i++)
    // {
    //     colors[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    // }

    // iterate through valid pixels and color
    for (int r = 0; r < color_depth_image.rows; r++)
    {
        for (int c = 0; c < color_depth_image.cols; c++)
        {    
            // if (isPixelValid(depth_image, label_image, normal_image, current)) 

            int cluster_id = clusters.at<int>(r, c);
            if (cluster_id != -1)
            {
                // ROS_INFO_STREAM("   (r, c): (" << r << ", " << c << ")");
                // ROS_INFO_STREAM("       cluster_id: " << cluster_id);
                cv::Scalar color = colors_[cluster_id];
                // ROS_INFO_STREAM("       color: " << color);
                color_cluster_image.at<cv::Vec3b>(r, c) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
    
    colored_cluster_img_ptr_->header = fin_depth_img_ptr_->header;
    // colored_cluster_img_ptr_->header.stamp = ros::Time::now();
    colored_cluster_img_ptr_->encoding = sensor_msgs::image_encodings::BGR8;

    colored_cluster_img_ptr_->image = color_cluster_image;
    colored_cluster_img_pub_.publish(colored_cluster_img_ptr_->toImageMsg());

}

void Visualizer::colorClusterPointCloud(const cv::Mat & depth_image, const cv::Mat & clusters)
{
    // ROS_INFO_STREAM("   [Visualizer::colorClusterPointCloud]");

    // std::lock_guard<std::mutex> lock(cloud_mutex_);

    // ROS_INFO_STREAM("       Coloring cluster point cloud ...");

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

    // colored_cloud->header = fin_depth_img_ptr_->header;
    // colored_cloud->header.stamp = ros::Time::now();
    colored_cloud->width = depth_image.cols;
    colored_cloud->height = depth_image.rows;
    // colored_cloud->is_dense = cloud_ptr_->is_dense;
    colored_cloud->points.resize(colored_cloud->width * colored_cloud->height);

    // iterate through valid pixels and color
    for (int r = 0; r < depth_image.rows; r++)
    {
        for (int c = 0; c < depth_image.cols; c++)
        {    
            pcl::PointXYZRGBA point;
            
            cv::Point pixel(c, r);
            cv::Vec3f egocanPt;
            pixelToEgocanFrame(egocanPt, pixel, depth_image.at<float>(r, c), params_.k_c_, params_.h_);

            point.x = egocanPt[0];
            point.y = egocanPt[1];
            point.z = egocanPt[2];

            int cluster_id = clusters.at<int>(r, c);
            if (cluster_id != -1)
            {
                cv::Scalar color = colors_[cluster_id];
                point.r = color[2];
                point.g = color[1];
                point.b = color[0];
                point.a = 255;
            } else
            {
                point.a = 0;
            }

            int i = r * colored_cloud->width + c;
            colored_cloud->points[i] = point;
        }
    }

    sensor_msgs::PointCloud2 colored_cloud_msg;
    pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
    colored_cloud_msg.header = fin_depth_img_ptr_->header;

    colored_point_cloud_pub_.publish(colored_cloud_msg);

    return;
}

void Visualizer::colorCentroids(const std::vector<std::vector<double>> & centers,
                                const std::vector<int> & center_counts)
{
    // ROS_INFO_STREAM("   [Visualizer::colorCentroids]");

    // clear prior markers
    visualization_msgs::MarkerArray clear_marker_array;
    visualization_msgs::Marker clearMarker;
    clearMarker.id = 0;
    clearMarker.ns =  "clear";
    clearMarker.action = visualization_msgs::Marker::DELETEALL;
    clear_marker_array.markers.push_back(clearMarker);    
    colored_centroids_pub_.publish(clear_marker_array);

    visualization_msgs::MarkerArray marker_array;

    for (int i = 0; i < centers.size(); i++)
    {
        if (center_counts[i] == 0)
        {
            continue;
        }

        visualization_msgs::Marker marker;
        marker.header = fin_depth_img_ptr_->header;
        marker.ns = "superpixel_centroids";
        marker.id = i;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        cv::Point center_pixel = cv::Point(centers[i][0], centers[i][1]);
        float center_depth = centers[i][2];
        cv::Vec3f center_normal = cv::Vec3f(centers[i][3], centers[i][4], centers[i][5]);

        cv::Vec3f centerEgocanPt;
        pixelToEgocanFrame(centerEgocanPt, center_pixel, center_depth, params_.k_c_, params_.h_);

        marker.points.resize(2);
        double scale = 0.1;
        geometry_msgs::Point p1, p2;
        p1.x = centerEgocanPt[0];
        p1.y = centerEgocanPt[1];
        p1.z = centerEgocanPt[2];
        p2.x = p1.x + scale * center_normal[0];
        p2.y = p1.y + scale * center_normal[1];
        p2.z = p1.z + scale * center_normal[2];

        // ROS_INFO_STREAM("       Centroid " << i << ": (" << p1.x << ", " << p1.y << ", " << p1.z << ")");
        // ROS_INFO_STREAM("       Normal " << i << ": (" << centers[i][4] << ", " << centers[i][5] << ", " << centers[i][6] << ")");

        marker.points[0] = p1;
        marker.points[1] = p2;

        marker.scale.x = 0.01;
        marker.scale.y = 0.02;
        marker.scale.z = 0.0;

        marker.color.a = 1.0;
        cv::Scalar color = colors_[i];
        marker.color.r = color[2] / 255.0;
        marker.color.g = color[1] / 255.0;
        marker.color.b = color[0] / 255.0;

        marker_array.markers.push_back(marker);
    }

    colored_centroids_pub_.publish(marker_array);
}