#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    nh_ = nh;

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.kernel_radius_ = configYamlNode["superpixels"]["kernel_radius"].as<int>();
    params_.num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();
    params_.num_dilation_iterations_ = configYamlNode["superpixels"]["num_dilation_iterations"].as<int>();
    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();    
    params_.w_normal_ = configYamlNode["superpixels"]["w_normal"].as<double>();
    params_.w_pos_ = configYamlNode["superpixels"]["w_pos"].as<double>();
    params_.w_compact_ = configYamlNode["superpixels"]["w_compact"].as<double>();
    params_.warm_start_ = configYamlNode["superpixels"]["warm_start"].as<bool>();

    ROS_INFO_STREAM("   params_:");
    ROS_INFO_STREAM("       num_superpixels_: " << params_.num_superpixels_);
    ROS_INFO_STREAM("       num_iterations_: " << params_.num_iterations_);
    ROS_INFO_STREAM("       num_dilation_iterations_: " << params_.num_dilation_iterations_);
    ROS_INFO_STREAM("       w_normal_: " << params_.w_normal_);
    ROS_INFO_STREAM("       w_pos_: " << params_.w_pos_);
    ROS_INFO_STREAM("       w_compact_: " << params_.w_compact_);
    ROS_INFO_STREAM("       kernel_radius_: " << params_.kernel_radius_);
    ROS_INFO_STREAM("       k_c_: " << params_.k_c_);
    ROS_INFO_STREAM("       v_fov_: " << params_.v_fov_);
    ROS_INFO_STREAM("       v_offset_: " << params_.v_offset_);
    ROS_INFO_STREAM("       h_: " << params_.h_);
    ROS_INFO_STREAM("       warm_start_: " << params_.warm_start_);

    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::calculateStep]");

    int num_pixels = params_.k_c_ * params_.k_c_;
    params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval

    // Set up subscribers and publishers
    image_transport::ImageTransport it(nh);

    std::string depth_img_topic =  "/egocylinder/floor_image";
    std::string label_img_topic =  "/egocylinder/floor_labels";
    std::string normal_img_topic = "/egocylinder/floor_normals";

    nh_.getParam("depth_img_topic", depth_img_topic);
    nh_.getParam("label_img_topic", label_img_topic);
    nh_.getParam("normal_img_topic", normal_img_topic);

    raw_depth_img_sub_.subscribe(it, depth_img_topic, 3);
    raw_label_img_sub_.subscribe(it, label_img_topic, 3);
    raw_normal_img_sub_.subscribe(it, normal_img_topic, 3);

    msg_sync_ = boost::make_shared<MsgSynchronizer>(raw_depth_img_sub_, raw_label_img_sub_, raw_normal_img_sub_, 10);
    msg_sync_->registerCallback(boost::bind(&SuperpixelDepthSegmenter::allImageCallback, this, _1, _2, _3));

    fin_depth_img_pub_ = it.advertise("/superpixels/process_depth", 1);
    fin_depth_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    fin_label_img_pub_ = it.advertise("/superpixels/process_labels", 1);
    fin_label_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    fin_normal_img_pub_ = it.advertise("/superpixels/process_normals", 1);
    fin_normal_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    fin_normal_img_colored_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    center_grid_img_pub_ = it.advertise("/superpixels/center_grid", 1);
    center_grid_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    colored_cluster_img_pub_ = it.advertise("/superpixels/colored_clusters", 1);
    colored_cluster_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    colors_.resize(params_.num_superpixels_);
    for (int i = 0; i < (int) colors_.size(); i++)
    {
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);

        for (int j = 0; j < i; j++)
        {
            if (cv::norm(color - colors_[j]) < 50)
            {
                color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
                j = -1;
            }
        }

        colors_[i] = color;
    }

    prop_depth_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    prop_label_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    prop_normal_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    visited_ = cv::Mat(params_.k_c_, params_.k_c_, CV_8UC1, cv::Scalar(0));

    colored_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/superpixels/colored_point_cloud", 1);
    colored_centroids_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/superpixels/colored_centroids", 1);

    // set up terrain publisher
    terrainPub_ = nh_.advertise<convex_plane_decomposition_msgs::PlanarTerrain>
                                    ("/convex_plane_decomposition_ros/planar_terrain", 1);

}

void SuperpixelDepthSegmenter::reconfigureCallback(superpixels::ParametersConfig &config, uint32_t level) 
{
    params_.kernel_radius_ = config.kernel_radius;
    params_.num_iterations_ = config.num_iterations;
    params_.num_dilation_iterations_ = config.num_dilation_iterations;
    params_.num_superpixels_ = config.num_superpixels;
    params_.w_normal_ = config.w_normal;
    params_.w_pos_ = config.w_pos;
    params_.w_compact_ = config.w_compact;

    int num_pixels = params_.k_c_ * params_.k_c_;
    params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval
}

void SuperpixelDepthSegmenter::allImageCallback(const sensor_msgs::ImageConstPtr& depth_image_msg, 
                                                const sensor_msgs::ImageConstPtr& label_image_msg, 
                                                const sensor_msgs::ImageConstPtr& normal_image_msg)
{   
    std::lock_guard<std::mutex> lock(img_mutex_);

    // ROS_INFO_STREAM("[SuperpixelDepthSegmenter::allImageCallback]");
    // ROS_INFO_STREAM("       time stamp: " << depth_image->header.stamp);

    raw_depth_img_msg_ = depth_image_msg;
    raw_label_img_msg_ = label_image_msg;
    raw_normal_img_msg_ = normal_image_msg;

    return;
}

bool SuperpixelDepthSegmenter::notReceivedDepthImage()
{
    return (raw_depth_img_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedLabelImage()
{
    return (raw_label_img_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedNormalImage()
{
    return (raw_normal_img_msg_ == nullptr);
}

bool SuperpixelDepthSegmenter::notReceivedImage()
{
    bool notReceivedDepth = notReceivedDepthImage();
    bool notReceivedLabel = notReceivedLabelImage();
    bool notReceivedNormal = notReceivedNormalImage();

    if (notReceivedDepth)
        ROS_WARN_STREAM("Not received depth image.");

    if (notReceivedLabel)
        ROS_WARN_STREAM("Not received label image.");

    if (notReceivedNormal)
        ROS_WARN_STREAM("Not received normal image.");

    return (notReceivedDepth || notReceivedLabel || notReceivedNormal);
}

void SuperpixelDepthSegmenter::run()
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if (notReceivedImage())
    {
        ROS_WARN("Not ready to segment, no images received yet.");
        return;
    }
 
    try
    {
        raw_depth_img_ptr_ = cv_bridge::toCvCopy(raw_depth_img_msg_, sensor_msgs::image_encodings::TYPE_32FC1);
        raw_label_img_ptr_ = cv_bridge::toCvCopy(raw_label_img_msg_, sensor_msgs::image_encodings::TYPE_8UC1);
        raw_normal_img_ptr_ = cv_bridge::toCvCopy(raw_normal_img_msg_, sensor_msgs::image_encodings::TYPE_32FC3);

    } catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (raw_depth_img_ptr_->image.size() != raw_label_img_ptr_->image.size() || 
        raw_depth_img_ptr_->image.size() != raw_normal_img_ptr_->image.size() ||
        raw_label_img_ptr_->image.size() != raw_normal_img_ptr_->image.size())
    {
        ROS_ERROR("Image sizes do not match.");
        return;
    }

    cleanBegin = std::chrono::steady_clock::now();

    cv::Mat raw_depth_img = raw_depth_img_ptr_->image;
    cv::Mat raw_label_img = raw_label_img_ptr_->image;
    cv::Mat raw_normal_img = raw_normal_img_ptr_->image;

    // Pre-processing
    // calculateStep(raw_depth_img);

    // Health check
    // healthCheck(raw_depth_img, raw_label_img, raw_normal_img);

    // Clean images
    cv::Mat cleaned_depth_img, cleaned_label_img, cleaned_normal_img;
    cleanImages(raw_depth_img, raw_label_img, raw_normal_img, 
                cleaned_depth_img, cleaned_label_img, cleaned_normal_img);

    // ROS_INFO_STREAM(" after cleaning, (480, 480) is: " << cleaned_depth_img.at<float>(480, 480));

    cleanEnd = std::chrono::steady_clock::now();
    int64_t clean_total_time = std::chrono::duration_cast<std::chrono::microseconds>(cleanEnd - cleanBegin).count();
    double clean_total_time_sec = clean_total_time / 1.0e6; 

    fillBegin = std::chrono::steady_clock::now();

    cv::Mat filled_depth_img, filled_label_img, filled_normal_img;
    fillInImage(cleaned_depth_img, cleaned_label_img, cleaned_normal_img,
                filled_depth_img, filled_label_img, filled_normal_img);

    // ROS_INFO_STREAM(" after filling, (480, 480) is: " << filled_depth_img.at<float>(480, 480));

    fillEnd = std::chrono::steady_clock::now();
    int64_t fill_total_time = std::chrono::duration_cast<std::chrono::microseconds>(fillEnd - fillBegin).count();
    double fill_total_time_sec = fill_total_time / 1.0e6; 

    // Health check
    // healthCheck(cleaned_depth_img, cleaned_label_img, cleaned_normal_img);

    preprocessBegin = std::chrono::steady_clock::now();

    // Pre-processing
    cv::Mat preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img;
    preprocessImages(filled_depth_img, filled_label_img, filled_normal_img,
                        preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    // ROS_INFO_STREAM(" after preprocessing, (480, 480) is: " << preprocessed_depth_img.at<float>(480, 480));

    preprocessEnd = std::chrono::steady_clock::now();
    int64_t preprocess_total_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessBegin).count();
    double preprocess_total_time_sec = preprocess_total_time / 1.0e6; 

    // Health check
    // healthCheck(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    // checkSparsity();

    // if (!initialized_)
    // {

    superpixelBegin = std::chrono::steady_clock::now();

    // Clear data
    reset_data(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    // Initialize data
    // init_data(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    //     initialized_ = true;
    // }

    // Generate superpixels
    generateSuperpixels(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    superpixelEnd = std::chrono::steady_clock::now();
    int64_t superpixel_total_time = std::chrono::duration_cast<std::chrono::microseconds>(superpixelEnd - superpixelBegin).count();
    double superpixel_total_time_sec = superpixel_total_time / 1.0e6; 

    ROS_INFO_STREAM_THROTTLE(3, "Timing ---- \n" << 
                                "   Image cleaning took: " << clean_total_time_sec << " seconds, \n" <<
                                "   Image filling took: " << fill_total_time_sec << " seconds, \n" <<
                                "   Image preprocessing took " << preprocess_total_time_sec << " seconds, \n" <<
                                "   Superpixels took " << superpixel_total_time_sec << " seconds, \n" << 
                                "   Number of superpixels: " << centers_.size() << "\n" <<
                                "   Total: " << clean_total_time_sec + fill_total_time_sec + preprocess_total_time_sec + superpixel_total_time_sec << " seconds");

    // Visualize
    visualize(preprocessed_depth_img, 
                preprocessed_label_img, 
                preprocessed_normal_img);

    // Publish planar regions
    publishPlanarRegions(preprocessed_depth_img);

    return;
}

void SuperpixelDepthSegmenter::publishPlanarRegions(const cv::Mat & depth_img)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::publishPlanarRegions]");

    ros::Time lookupTime = fin_depth_img_ptr_->header.stamp;
    std::string camera_frame = fin_depth_img_ptr_->header.frame_id;

    geometry_msgs::TransformStamped cameraFrameToWorldFrameTransform = 
        tfBuffer_.lookupTransform("world", camera_frame, lookupTime);

    convex_plane_decomposition_msgs::PlanarTerrain terrain_msg;

    // ROS_INFO_STREAM("       regions:");
    for (int i = 0; i < centers_.size(); i++)
    {
        // ROS_INFO_STREAM("           i: " << i);
        convex_plane_decomposition::PlanarRegion region;

        cv::Point pixel = cv::Point(centers_[i][0], centers_[i][1]);
        cv::Vec3f worldPt;
        floorPixelToWorld(worldPt, pixel, depth_img.at<float>(pixel.y, pixel.x), params_.k_c_, params_.h_);

        Eigen::Vector3d center(worldPt.val[0], worldPt.val[1], worldPt.val[2]);
        Eigen::Vector3d normal(centers_[i][4], centers_[i][5], centers_[i][6]);

        // ROS_INFO_STREAM("               center: " << center.transpose());
        // ROS_INFO_STREAM("               normal: " << normal.transpose());

        Eigen::Vector3d arbitraryVec(1, 0, 0);

        // check here to make sure arbitraryVec is not parallel to normal
        if (std::abs(arbitraryVec.dot(normal)) > 0.99)
        {
            arbitraryVec = Eigen::Vector3d(0, 1, 0);
        }

        Eigen::Vector3d e0 = normal.cross(arbitraryVec);
        e0.normalize();
        Eigen::Vector3d e1 = normal.cross(e0);
        e1.normalize();

        // ROS_INFO_STREAM("               e0: " << e0.transpose());
        // ROS_INFO_STREAM("               e1: " << e1.transpose());

        Eigen::Matrix3d regionRotMat;
        regionRotMat << e0, e1, normal;

        Eigen::Quaterniond regionQuat(regionRotMat);

        // Eigen::VectorXd pose_camera_frame(7);
        // pose_camera_frame << center, regionQuat;

        // ROS_INFO_STREAM("               center: " << center.transpose());
        // ROS_INFO_STREAM("               regionQuat: " << regionQuat.x() << ", " << regionQuat.y() << ", " << regionQuat.z() << ", " << regionQuat.w());

        Eigen::VectorXd pose_world_frame = transformHelperPoseStamped(center, regionQuat, cameraFrameToWorldFrameTransform);

        // get rotation matrix
        Eigen::Matrix3d rotMat = calculateRotationMatrix(pose_world_frame[3], pose_world_frame[4], pose_world_frame[5]);

        region.transformPlaneToWorld.translation() = pose_world_frame.head(3);
        region.transformPlaneToWorld.linear() = rotMat;      

        // ROS_INFO_STREAM("               translation: " << region.transformPlaneToWorld.translation().transpose());
        // ROS_INFO_STREAM("               rotation: " << region.transformPlaneToWorld.linear().row(0));
        // ROS_INFO_STREAM("                         " << region.transformPlaneToWorld.linear().row(1));
        // ROS_INFO_STREAM("                         " << region.transformPlaneToWorld.linear().row(2));

        convex_plane_decomposition::BoundaryWithInset boundaryWithInset;

        convex_plane_decomposition::CgalPolygonWithHoles2d polygonWithHoles;
        convex_plane_decomposition::CgalPolygon2d polygon;
        double foot_radius = 0.02;
        double inner_radius = foot_radius;
        double outer_radius = 2*foot_radius;
        polygon.container().emplace_back(-outer_radius, -outer_radius); // bottom left
        polygon.container().emplace_back(outer_radius, -outer_radius); // bottom right
        polygon.container().emplace_back(outer_radius, outer_radius); // top right
        polygon.container().emplace_back(-outer_radius, outer_radius); // top left

        polygonWithHoles.outer_boundary() = polygon;

        boundaryWithInset.boundary = polygonWithHoles;

        std::vector<convex_plane_decomposition::CgalPolygonWithHoles2d> insets;

        convex_plane_decomposition::CgalPolygon2d inflated_polygon;
        inflated_polygon.container().emplace_back(-inner_radius, -inner_radius); // bottom left
        inflated_polygon.container().emplace_back(inner_radius, -inner_radius); // bottom right
        inflated_polygon.container().emplace_back(inner_radius, inner_radius); // top right
        inflated_polygon.container().emplace_back(-inner_radius, inner_radius); // top left

        convex_plane_decomposition::CgalPolygonWithHoles2d inflated_polygon_with_holes;
        inflated_polygon_with_holes.outer_boundary() = inflated_polygon;

        insets.push_back(inflated_polygon_with_holes);

        boundaryWithInset.insets = insets;

        region.boundaryWithInset = boundaryWithInset;
        region.bbox2d = boundaryWithInset.boundary.outer_boundary().bbox();

        convex_plane_decomposition_msgs::PlanarRegion region_msg = convex_plane_decomposition::toMessage(region);

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

void SuperpixelDepthSegmenter::fillInImage(const cv::Mat & cleaned_depth_img,
                                            const cv::Mat & cleaned_label_img,
                                            const cv::Mat & cleaned_normal_img,
                                            cv::Mat & filled_depth_img,
                                            cv::Mat & filled_label_img,
                                            cv::Mat & filled_normal_img)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::fillInImage]");

    // Depth
    filled_depth_img = cleaned_depth_img.clone();

    // Labels
    filled_label_img = cleaned_label_img.clone();

    // Normals
    filled_normal_img = cleaned_normal_img.clone();

    int delta = params_.step_ / 4;

    for (int r = delta; r < (cleaned_depth_img.rows - delta); r += delta)
    {
        for (int c = delta; c < (cleaned_depth_img.cols - delta); c += delta)
        {
            // ROS_INFO_STREAM("    Checking pixel: (" << r << ", " << c << ")");

            cv::Point curr_pixel(c, r);
            if (isPixelInBounds(params_.k_c_, curr_pixel) )
            {
                bool found_visited_pixel = false;

                for (int i = -delta; i <= delta; i++)
                {
                    for (int j = -delta; j <= delta; j++)
                    {
                        cv::Point pixel(c + j, r + i);

                        if (isPixelInBounds(params_.k_c_, pixel) && visited_.at<uint8_t>(pixel.y, pixel.x) == 1)
                        {
                            found_visited_pixel = true;
                            break;
                        }
                    }

                    if (found_visited_pixel)
                    {
                        break;
                    }
                }

                if (!found_visited_pixel)
                {
                    // ROS_INFO_STREAM("       did not find a pixel, filling in at (" << r << ", " << c << ") ...");

                    filled_depth_img.at<float>(r, c) = 0.385;
                    filled_label_img.at<uint8_t>(r, c) = 3;
                    filled_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, -1.0, 0);
                } else
                {
                    // ROS_INFO_STREAM("       invalid");
                }
            }
        }
    }
}

void SuperpixelDepthSegmenter::cleanImages(const cv::Mat & raw_depth_img,
                                            const cv::Mat & raw_label_img,
                                            const cv::Mat & raw_normal_img,
                                            cv::Mat & cleaned_depth_img,
                                            cv::Mat & cleaned_label_img,
                                            cv::Mat & cleaned_normal_img)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::cleanImages]");


    // Depth
    cleaned_depth_img = cv::Mat(raw_depth_img.size(), CV_32F, cv::Scalar(0));

    // Labels
    cleaned_label_img = cv::Mat(raw_label_img.size(), CV_8UC1, cv::Scalar(0));

    // Normals
    cleaned_normal_img = cv::Mat(raw_normal_img.size(), CV_32FC3, cv::Scalar(0));

    // zero invalid depth pixels
    for (int r = 0; r < cleaned_depth_img.rows; r++)
    {
        for (int c = 0; c < cleaned_depth_img.cols; c++)
        {
            if (isPixelValid(raw_depth_img, raw_label_img, raw_normal_img, cv::Point(c, r), params_.k_c_))
            {

                float depth = raw_depth_img.at<float>(r, c);

                // bool valid_depth = (!std::isnan(depth) && std::abs(depth) > 1e-6 && depth > 0);

                uint8_t label = raw_label_img.at<uint8_t>(r, c);

                // bool valid_label = (!std::isnan(label) && label >= 0);

                cv::Vec3f normal = raw_normal_img.at<cv::Vec3f>(r, c);

                // bool valid_normal = (!std::isnan(normal.val[0]) && !std::isnan(normal.val[1]) && !std::isnan(normal.val[2]) && 
                                        // cv::norm(normal) > DELTA);

                // if (valid_depth && valid_label && valid_normal)
                // {
                cleaned_depth_img.at<float>(r, c) = depth;
                cleaned_label_img.at<uint8_t>(r, c) = label;
                cleaned_normal_img.at<cv::Vec3f>(r, c) = normal;
                visited_.at<uint8_t>(r, c) = 1;
                // } 
            } else
            {
                cleaned_depth_img.at<float>(r, c) = 0;
                cleaned_label_img.at<uint8_t>(r, c) = 0;
                cleaned_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
            }

            // if (std::isnan(depth))
            // {
            //     cleaned_depth_img.at<float>(r, c) = 0;
            // } else if (std::fabs(depth) < 1e-6)
            // {
            //     cleaned_depth_img.at<float>(r, c) = 0;
            // } else if (depth < 0)
            // {
            //     cleaned_depth_img.at<float>(r, c) = 0;
            // } else
            // {
            //     cleaned_depth_img.at<float>(r, c) = depth;
            //     visited_.at<uint8_t>(r, c) = 1;
            // }
    //     }
    // }    

    // ROS_INFO_STREAM("       Checking label image ...");

    // for (int r = 0; r < cleaned_label_img.rows; r++)
    // {
    //     for (int c = 0; c < cleaned_label_img.cols; c++)
    //     {

            // if (std::isnan(label))
            //     cleaned_label_img.at<uint8_t>(r, c) = 0;
            // else if (label < 0)
            //     cleaned_label_img.at<uint8_t>(r, c) = 0;
            // else
            //     cleaned_label_img.at<uint8_t>(r, c) = label;
    //     }
    // }

    // ROS_INFO_STREAM("       Checking normal image ...");

    // for (int r = 0; r < cleaned_normal_img.rows; r++)
    // {
    //     for (int c = 0; c < cleaned_normal_img.cols; c++)
    //     {

            // if (std::isnan(normal.val[0]) || std::isnan(normal.val[1]) || std::isnan(normal.val[2]))
            //     cleaned_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
            // else if (cv::norm(normal) < DELTA)
            //     cleaned_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
            // else
            //     cleaned_normal_img.at<cv::Vec3f>(r, c) = normal;
        }
    }

    return;
}

void SuperpixelDepthSegmenter::healthCheck(const cv::Mat & depth_img,
                                            const cv::Mat & label_img,
                                            const cv::Mat & normal_img)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::healthCheck]");
    // Detecting invalid pixels

    int nan_depth_count = 0;
    int negative_depth_count = 0;
    int finite_depth_count = 0;
    int zero_depth_count = 0;

    int nan_label_count = 0;
    int negative_label_count = 0;
    int finite_label_count = 0;

    int nan_normal_count = 0;
    int finite_normal_count = 0;
    int zero_normal_count = 0;

    // ROS_INFO_STREAM("        Checking images ...");

    // Depth
    for (int r = 0; r < depth_img.rows; r++)
    {
        for (int c = 0; c < depth_img.cols; c++)
        {
            float depth = depth_img.at<float>(r, c);

            if (std::isnan(depth))
            {
                ROS_WARN_STREAM_COND(nan_depth_count == 0, "            Detected NaN depth pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                nan_depth_count++;
            } else if (std::abs(depth) < 1e-6)
            {
                ROS_WARN_STREAM_COND(zero_depth_count == 0, "            Detected zero depth pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                zero_depth_count++;
            } else if (depth < 0)
            {
                ROS_WARN_STREAM_COND(negative_depth_count == 0, "            Detected negative depth pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                negative_depth_count++;
            } else
            {
                finite_depth_count++;
            }

            uint8_t label = label_img.at<uint8_t>(r, c);

            if (std::isnan(label))
            {
                ROS_WARN_STREAM_COND(nan_label_count == 0, "            Detected NaN label pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                nan_label_count++;
            } else if (label < 0)
            {
                ROS_WARN_STREAM_COND(negative_label_count == 0, "            Detected negative label pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                negative_label_count++;
            } else
            {
                finite_label_count++;
            }

            cv::Vec3f normal = normal_img.at<cv::Vec3f>(r, c);

            if (std::isnan(normal.val[0]) || std::isnan(normal.val[1]) || std::isnan(normal.val[2]))
            {
                ROS_WARN_STREAM_COND(nan_normal_count == 0, "            Detected NaN normal pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                nan_normal_count++;
            } else if (cv::norm(normal) < DELTA)
            {
                ROS_WARN_STREAM_COND(zero_normal_count == 0, "            Detected zero normal pixel"); //  at: (" << r << ", " << c << "), zeroing ...");
                zero_normal_count++;
            } else
            {
                finite_normal_count++;
            }
        }
    }

    int total_pixels = depth_img.rows * depth_img.cols;

    ROS_INFO_STREAM("        Health check:");
    ROS_INFO_STREAM("           Depth:");
    ROS_INFO_STREAM("               NaN count: " << nan_depth_count);
    ROS_INFO_STREAM("               Negative count: " << negative_depth_count);
    ROS_INFO_STREAM("               Zero count: " << zero_depth_count);
    ROS_INFO_STREAM("               Finite count: " << finite_depth_count);
    ROS_INFO_STREAM("               Unaccounted for: " << total_pixels - (nan_depth_count + negative_depth_count + zero_depth_count + finite_depth_count));
    ROS_INFO_STREAM("        Label:");
    ROS_INFO_STREAM("               NaN count: " << nan_label_count);
    ROS_INFO_STREAM("               Negative count: " << negative_label_count);
    ROS_INFO_STREAM("               Finite count: " << finite_label_count);
    ROS_INFO_STREAM("               Unaccounted for: " << total_pixels - (nan_label_count + negative_label_count + finite_label_count));
    ROS_INFO_STREAM("        Normal:");
    ROS_INFO_STREAM("               NaN count: " << nan_normal_count);
    ROS_INFO_STREAM("               Zero count: " << zero_normal_count);
    ROS_INFO_STREAM("               Finite count: " << finite_normal_count);
    ROS_INFO_STREAM("               Unaccounted for: " << total_pixels - (nan_normal_count + zero_normal_count + finite_normal_count));

    if (finite_depth_count != finite_normal_count)
    {
        ROS_WARN_STREAM("        Finite counts do not match between depth and normal images.");
    }

}

void SuperpixelDepthSegmenter::preprocessImages(const cv::Mat & cleaned_depth_img,
                                                const cv::Mat & cleaned_label_img,
                                                const cv::Mat & cleaned_normal_img,
                                                cv::Mat & preprocessed_depth_img,
                                                cv::Mat & preprocessed_label_img,
                                                cv::Mat & preprocessed_normal_img)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::preprocessImages]");

    if (params_.kernel_radius_ < 1)
    {
        preprocessed_depth_img = cleaned_depth_img.clone();
        preprocessed_label_img = cleaned_label_img.clone();
        preprocessed_normal_img = cleaned_normal_img.clone();
        return;
    }

    // Custom dilation implementation
    float max_depth = -std::numeric_limits<float>::max();
    int max_label = 0;
    cv::Vec3f max_normal(0, 0, 0);
    cv::Point max_depth_pixel(-1, -1);

    float min_depth = std::numeric_limits<float>::max();
    int min_label = 0;
    cv::Vec3f min_normal(0, 0, 0);
    cv::Point min_depth_pixel(-1, -1);

    cv::Mat dilated_depth_img = cleaned_depth_img.clone(); // cv::Mat(cleaned_depth_img.size(), CV_32F, cv::Scalar(0));
    cv::Mat dilated_label_img = cleaned_label_img.clone(); // cv::Mat(cleaned_label_img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat dilated_normal_img = cleaned_normal_img.clone(); // cv::Mat(cleaned_normal_img.size(), CV_32FC3, cv::Scalar(0));

    cv::Mat temp_dilated_depth_img = cv::Mat(cleaned_depth_img.size(), CV_32F, cv::Scalar(0));
    cv::Mat temp_dilated_label_img = cv::Mat(cleaned_label_img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp_dilated_normal_img = cv::Mat(cleaned_normal_img.size(), CV_32FC3, cv::Scalar(0));

    int new_r = 0;
    int new_c = 0;

    float depth = 0;

    for (int iter = 0; iter < params_.num_dilation_iterations_; iter++)
    {
        for (int r = 0; r < cleaned_depth_img.rows; r++)
        {
            for (int c = 0; c < cleaned_depth_img.cols; c++)
            {
                // max_depth = 0;
                // max_label = 0;
                // max_normal = cv::Vec3f(0, 0, 0);
                // max_depth_pixel = cv::Point(-1, -1);

                min_depth = std::numeric_limits<float>::max();
                min_label = 0;
                min_normal = cv::Vec3f(0, 0, 0);
                min_depth_pixel = cv::Point(-1, -1);

                for (int i = -params_.kernel_radius_; i <= params_.kernel_radius_; i++)
                {
                    for (int j = -params_.kernel_radius_; j <= params_.kernel_radius_; j++)
                    {
                        new_r = r + i;
                        new_c = c + j;

                        if (!isPixelInBounds(params_.k_c_, cv::Point(new_c, new_r)))
                        {
                            continue;
                        }

                        depth = dilated_depth_img.at<float>(new_r, new_c);

                        if (depth > 0.0 && depth < min_depth)
                        {
                            min_depth = depth;
                            min_label = dilated_label_img.at<uint8_t>(new_r, new_c);
                            min_normal = dilated_normal_img.at<cv::Vec3f>(new_r, new_c);
                            min_depth_pixel = cv::Point(new_c, new_r);
                            temp_dilated_depth_img.at<float>(r, c) = min_depth;
                            temp_dilated_label_img.at<uint8_t>(r, c) = min_label;
                            temp_dilated_normal_img.at<cv::Vec3f>(r, c) = min_normal;                            
                        }

                        // if (depth > max_depth)
                        // {
                        //     max_depth = depth;
                        //     max_label = dilated_label_img.at<uint8_t>(new_r, new_c);
                        //     max_normal = dilated_normal_img.at<cv::Vec3f>(new_r, new_c);
                        //     max_depth_pixel = cv::Point(new_c, new_r);
                        // }
                    }
                }

                // temp_dilated_depth_img.at<float>(r, c) = max_depth;
                // temp_dilated_label_img.at<uint8_t>(r, c) = max_label;
                // temp_dilated_normal_img.at<cv::Vec3f>(r, c) = max_normal;

                // temp_dilated_depth_img.at<float>(r, c) = min_depth;
                // temp_dilated_label_img.at<uint8_t>(r, c) = min_label;
                // temp_dilated_normal_img.at<cv::Vec3f>(r, c) = min_normal;
            }
        }

        dilated_depth_img = temp_dilated_depth_img.clone();
        dilated_label_img = temp_dilated_label_img.clone();
        dilated_normal_img = temp_dilated_normal_img.clone();
    }

    preprocessed_depth_img = dilated_depth_img.clone();
    preprocessed_label_img = dilated_label_img.clone();
    preprocessed_normal_img = dilated_normal_img.clone();

    // ROS_INFO_STREAM("       Comparing normal image to dilated normal image ...");
    // Compare normal image with dilated normal image
    // for (int r = 0; r < cleaned_normal_img.rows; r++)
    // {
    //     for (int c = 0; c < cleaned_normal_img.cols; c++)
    //     {
    //         cv::Vec3f normal = cleaned_normal_img.at<cv::Vec3f>(r, c);
    //         // cv::Vec3f added_normal = added_normal_img.at<cv::Vec3f>(r, c);

    //         if (cv::norm(normal) < DELTA)
    //         {
    //             continue;
    //         }


    //         cv::Vec3f dilated_normal = preprocessed_normal_img.at<cv::Vec3f>(r, c);

    //         ROS_INFO_STREAM("       Normal: (" << normal.val[0] << ", " << normal.val[1] << ", " << normal.val[2] << ")");
    //         // ROS_INFO_STREAM("       Added normal: (" << added_normal.val[0] << ", " << added_normal.val[1] << ", " << added_normal.val[2] << ")");
    //         ROS_INFO_STREAM("       Dilated Normal: (" << dilated_normal.val[0] << ", " << dilated_normal.val[1] << ", " << dilated_normal.val[2] << ")");
            
    //     }
    // }    
}

// void SuperpixelDepthSegmenter::dilate_img(const cv::Mat & image, cv::Mat & dilated_image)
// {
    // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // cv::dilate(image, dilated_image, element, cv::Point(-1, -1), 1);
// }

void SuperpixelDepthSegmenter::generateSuperpixels(const cv::Mat & depth_image,
                                                    const cv::Mat & label_image,
                                                    const cv::Mat & normal_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::generateSuperpixels]");

    // Generate superpixels
    for (int i = 0; i < params_.num_iterations_; i++)
    {
        /* Reset distance and cluster values. */
        distances_ = cv::Mat(depth_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max()));
        // clusters_ = cv::Mat(depth_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer

        /* Update distances and clusters */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int r = centers_[j][1] - params_.step_; r < centers_[j][1] + params_.step_; r++) 
            {
                for (int c = centers_[j][0] - params_.step_; c < centers_[j][0] + params_.step_; c++) 
                {                
                    cv::Point current(c, r);
                    if (isPixelValid(depth_image, label_image, normal_image, current, params_.k_c_)) 
                    {
                        float depth = depth_image.at<float>(r, c);
                        uint8_t label = label_image.at<uint8_t>(r, c);
                        cv::Vec3f normal = normal_image.at<cv::Vec3f>(r, c);

                        bool check = checkConstraints(j, 
                                                        depth,
                                                        label,
                                                        normal,
                                                        current);

                        double d = computeDistance(j, 
                                                    depth,
                                                    label,
                                                    normal,
                                                    current);

                        if (check && d < distances_.at<double>(r, c)) 
                        {
                            distances_.at<double>(r, c) = d;
                            clusters_.at<int>(r, c) = j;
                        }
                    }
                }
            }
        }

        /* Clear the center values. */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            centers_[j][0] = 0;
            centers_[j][1] = 0;
            // centers_[j][2] = 0;
            // centers_[j][3] = 0;
            // centers_[j][4] = 0;
            // centers_[j][5] = 0;
            // centers_[j][6] = 0;
            center_counts_[j] = 0;
        }

        superpixels_ = std::vector<std::vector<cv::Point>>(centers_.size());
        /* Compute the new cluster centers. */
        for (int r = 0; r < depth_image.rows; r++) 
        {
            for (int c = 0; c < depth_image.cols; c++) 
            {            
                cv::Point current(c, r);
                
                int cluster_id = clusters_.at<int>(r, c);

                if (cluster_id != -1) 
                {
                    // float depth = depth_image.at<float>(r, c);
                    // uint8_t label = label_image.at<uint8_t>(r, c);
                    // cv::Vec3f normal = normal_image.at<cv::Vec3f>(r, c);

                    centers_[cluster_id][0] += c;
                    centers_[cluster_id][1] += r;
                    // centers_[cluster_id][2] += depth;
                    // centers_[cluster_id][3] += label;
                    // centers_[cluster_id][4] += normal.val[0];
                    // centers_[cluster_id][5] += normal.val[1];
                    // centers_[cluster_id][6] += normal.val[2];
                    
                    center_counts_[cluster_id] += 1; 

                    superpixels_[cluster_id].push_back(current);
                }
            }
        }     

        /* Normalize the clusters. */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            if (center_counts_[j] == 0) 
            {
                continue;
            }

            centers_[j][0] /= center_counts_[j];
            centers_[j][1] /= center_counts_[j];
            // centers_[j][2] /= center_counts_[j];
            // centers_[j][3] /= center_counts_[j];
            // centers_[j][4] /= center_counts_[j];
            // centers_[j][5] /= center_counts_[j];
            // centers_[j][6] /= center_counts_[j];
        }

        /* Snap clusters to nearest pixel */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            // ROS_INFO_STREAM("       [" << j << "]: ");

            cv::Point center = cv::Point(centers_[j][0], centers_[j][1]);
            cv::Point new_center = findClosestPixel(j, center, depth_image, label_image, normal_image);
            float depth = depth_image.at<float>(new_center.y, new_center.x);
            uint8_t label = label_image.at<uint8_t>(new_center.y, new_center.x);
            cv::Vec3f normal = normal_image.at<cv::Vec3f>(new_center.y, new_center.x);
            centers_[j][0] = new_center.x;
            centers_[j][1] = new_center.y;
            centers_[j][2] = depth;
            centers_[j][3] = label;
            centers_[j][4] = normal.val[0];
            centers_[j][5] = normal.val[1];
            centers_[j][6] = normal.val[2];

            // refine normal via RANSAC
            // ROS_INFO_STREAM("           pixel: " << centers_[j][0] << ", " << centers_[j][1]);
            // ROS_INFO_STREAM("           depth: " << centers_[j][2]);
            // ROS_INFO_STREAM("           label: " << centers_[j][3]);
            // ROS_INFO_STREAM("           normal: " << centers_[j][4] << ", " << centers_[j][5] << ", " << centers_[j][6]);
            // ROS_INFO_STREAM("           counts: " << center_counts_[j]);
            
            cv::Vec3f candidate_normal = ransac(superpixels_[j], depth_image, normal);

            centers_[j][4] = candidate_normal.val[0];
            centers_[j][5] = candidate_normal.val[1];
            centers_[j][6] = candidate_normal.val[2];

            // ROS_INFO_STREAM("           candidate normal: " << candidate_normal.val[0] << ", " << candidate_normal.val[1] << ", " << candidate_normal.val[2]);
        }

    }

    // ROS_INFO_STREAM("       centers:");
    // for (int i = 0; i < (int) center_counts_.size(); i++)
    // {
    //     ROS_INFO_STREAM("           [" << i << "]: ");
    //     ROS_INFO_STREAM("               pixel: " << centers_[i][0] << ", " << centers_[i][1]);
    //     ROS_INFO_STREAM("               depth: " << centers_[i][2]);
    //     ROS_INFO_STREAM("               label: " << centers_[i][3]);
    //     ROS_INFO_STREAM("               normal: " << centers_[i][4] << ", " << centers_[i][5] << ", " << centers_[i][6]);
    //     ROS_INFO_STREAM("               counts: " << center_counts_[i]);
    // }
}

cv::Vec3f SuperpixelDepthSegmenter::ransac(const std::vector<cv::Point> & pixels, const cv::Mat & depth_image, const cv::Vec3f & og_normal)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::ransac]");

    // ROS_INFO_STREAM("       number of points: " << pixels.size());

    int K = 3; // number of points to sample
    int N = 25; // number of iterations
    double T = 0.01; // threshold

    cv::Vec3f normal = og_normal;

    if (pixels.size() < K)
        return normal;

    int max_inliers = 0;    
    for (int n = 0; n < N; n++)
    {
        // ROS_INFO_STREAM("       iteration: " << n);

        // ROS_INFO_STREAM("           sampling ...");
        // sample
        std::vector<cv::Point> sample;
        std::vector<int> indices;
        for (int i = 0; i < K; i++)
        {
            int idx = -1;
            while (idx == -1 || std::find(indices.begin(), indices.end(), idx) != indices.end())
                idx = rand() % pixels.size();

            sample.push_back(pixels[idx]);
            indices.push_back(idx);
        }

        // ROS_INFO_STREAM("           fitting ...");
        // fit
        Eigen::MatrixXd A(K, 3);
        Eigen::VectorXd b(K);

        for (int i = 0; i < K; i++)
        {
            cv::Point pixel = sample[i];
            cv::Vec3f worldPt;
            floorPixelToWorld(worldPt, pixel, depth_image.at<float>(pixel.y, pixel.x), params_.k_c_, params_.h_);

            A(i, 0) = worldPt.val[0];
            A(i, 1) = 1.0;
            A(i, 2) = worldPt.val[2];
            b(i) = worldPt.val[1];
        }

        Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        // ROS_INFO_STREAM("           computing inliers ...");
        // compute inliers
        std::vector<cv::Point> inliers;
        for (int i = 0; i < pixels.size(); i++)
        {
            cv::Point pixel = pixels[i];
            cv::Vec3f worldPt;
            floorPixelToWorld(worldPt, pixel, depth_image.at<float>(pixel.y, pixel.x), params_.k_c_, params_.h_);

            double error = std::abs(worldPt.val[1] - (x(0) * worldPt.val[0] + x(1) + x(2) * worldPt.val[2]));
            if (error < T)
                inliers.push_back(pixel);
        }

        // ROS_INFO_STREAM("           number of inliers: " << inliers.size());

        // ROS_INFO_STREAM("           updating ...");
        // update
        if (inliers.size() > max_inliers)
        {
            max_inliers = inliers.size();

            // update normal

            Eigen::MatrixXd A_best(max_inliers, 3);
            Eigen::VectorXd b_best(max_inliers);

            for (int i = 0; i < max_inliers; i++)
            {
                cv::Point pixel = inliers[i];
                cv::Vec3f worldPt;
                floorPixelToWorld(worldPt, pixel, depth_image.at<float>(pixel.y, pixel.x), params_.k_c_, params_.h_);

                A_best(i, 0) = worldPt.val[0];
                A_best(i, 1) = 1.0;
                A_best(i, 2) = worldPt.val[2];
                b_best(i) = worldPt.val[1];
            }

            Eigen::VectorXd x_best = A_best.colPivHouseholderQr().solve(b_best);

            cv::Vec3f new_normal(x_best(0), -1.0, x_best(2));

            normal = new_normal / cv::norm(new_normal);
        }

    }

    return normal;
}

cv::Point SuperpixelDepthSegmenter::findClosestPixel(const int & center_idx,
                                                        const cv::Point & center, 
                                                        const cv::Mat & depth_image,
                                                        const cv::Mat & label_image,
                                                        const cv::Mat & normal_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::findClosestPixel]");

    cv::Point new_center = center;

    // inefficient
    for (int delta = 1; delta < params_.step_; delta++)
    {
        for (int i = -delta; i <= delta; i++)
        {
            for (int j = -delta; j <= delta; j++)
            {
                cv::Point current(center.x + j, center.y + i);

                if (isPixelValid(depth_image, label_image, normal_image, current, params_.k_c_) && 
                    clusters_.at<int>(current.y, current.x) == center_idx)
                {
                    new_center = current;
                    return new_center;
                }
            }
        }
    }

    return new_center;
}

bool SuperpixelDepthSegmenter::checkConstraints(const int & center_idx, 
                                                const float & depth,
                                                const uint8_t & label,
                                                const cv::Vec3f & normal,
                                                const cv::Point & pixel)
{
    cv::Point center_pixel = cv::Point(centers_[center_idx][0], centers_[center_idx][1]);
    float center_depth = centers_[center_idx][2];
    uint8_t center_label = centers_[center_idx][3];
    cv::Vec3f center_normal = cv::Vec3f(centers_[center_idx][4], centers_[center_idx][5], centers_[center_idx][6]);

    cv::Vec3f worldPt;
    floorPixelToWorld(worldPt, pixel, depth, params_.k_c_, params_.h_);

    cv::Vec3f centerWorldPt;
    floorPixelToWorld(centerWorldPt, center_pixel, center_depth, params_.k_c_, params_.h_);

    bool normal_check = normal.dot(center_normal) > 0.75;

    bool plane_distance_check = std::abs( (centerWorldPt - worldPt).dot(center_normal))  < 0.01;

    return normal_check && plane_distance_check;
}

double SuperpixelDepthSegmenter::computeDistance(const int & center_idx, 
                                                    const float & depth,
                                                    const uint8_t & label,
                                                    const cv::Vec3f & normal,
                                                    const cv::Point & pixel)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::computeDistance]");

    cv::Point center_pixel = cv::Point(centers_[center_idx][0], centers_[center_idx][1]);
    float center_depth = centers_[center_idx][2];
    uint8_t center_label = centers_[center_idx][3];
    cv::Vec3f center_normal = cv::Vec3f(centers_[center_idx][4], centers_[center_idx][5], centers_[center_idx][6]);

    cv::Vec3f worldPt;
    floorPixelToWorld(worldPt, pixel, depth, params_.k_c_, params_.h_);

    cv::Vec3f centerWorldPt;
    floorPixelToWorld(centerWorldPt, center_pixel, center_depth, params_.k_c_, params_.h_);

    // ROS_INFO_STREAM("           center_idx: " << center_idx);
    // ROS_INFO_STREAM("           center_pixel: (r:" << center_pixel.y << ", c: " << center_pixel.x << ")");
    // ROS_INFO_STREAM("           center_world_pt: " << centerWorldPt);
    // ROS_INFO_STREAM("           center_depth: " << center_depth);
    // ROS_INFO_STREAM("           center_label: " << center_label);
    // ROS_INFO_STREAM("           center_normal: " << center_normal);

    // ROS_INFO_STREAM("           pixel: (r: " << pixel.y << ", c: " << pixel.x << ")");
    // ROS_INFO_STREAM("           world_pt: " << worldPt);
    // ROS_INFO_STREAM("           depth: " << depth);
    // ROS_INFO_STREAM("           label: " << label);
    // ROS_INFO_STREAM("           normal: " << normal);


    // Normal term
    double d_normal = (1.0 - normal.dot(center_normal));
    double max_d_normal = 2.0;
    double weighted_d_normal = params_.w_normal_ * (d_normal / max_d_normal);
    // double dc = sqrt(pow(color.val[0] - centers_[center_idx][0], 2) +
    //                  pow(color.val[1] - centers_[center_idx][1], 2) +
    //                  pow(color.val[2] - centers_[center_idx][2], 2));

    if (d_normal > max_d_normal)
    {
        ROS_WARN_STREAM("       d_normal exceeds max, d_normal: " << d_normal << ", max_d_normal: " << max_d_normal);
    }

    // ROS_INFO_STREAM("           d_normal: " << d_normal);

    // Position term

    double d_posn = std::abs( (worldPt - centerWorldPt).dot(center_normal) );
    double max_d_posn = params_.v_fov_;
    double weighted_d_posn = params_.w_pos_ * (d_posn / max_d_posn);

    if (d_posn > max_d_posn)
    {
        ROS_WARN_STREAM("       d_posn exceeds max, d_posn: " << d_posn << ", max_d_posn: " << max_d_posn);
    }

    // // Spatial term
    // double ds = sqrt(pow(pixel.x - centers_[center_idx][3], 2) +
    //                  pow(pixel.y - centers_[center_idx][4], 2));

    // ROS_INFO_STREAM("           d_posn: " << d_posn);

    // Compactness term
    double d_compact = sqrt(pow(center_pixel.x - pixel.x, 2) + pow(center_pixel.y - pixel.y, 2));
    double max_compact_dist = sqrt(pow(params_.step_, 2) + pow(params_.step_, 2));
    double weighted_d_compact = params_.w_compact_ * (d_compact / max_compact_dist);

    if (d_compact > max_compact_dist)
    {
        ROS_WARN_STREAM("       d_compact exceeds max, d_compact: " << d_compact << ", max_compact_dist: " << max_compact_dist);
    }

    // double d_compact = cv::norm(centerWorldPt - worldPt);
    // double max_compact_dist = 
    // double weighted_d_compact = params_.w_compact_ * d_compact;

    // return sqrt(pow(dc / params_.n_c_, 2) + pow(ds / params_.n_s_, 2));
    return weighted_d_normal + weighted_d_posn + weighted_d_compact;
}

// void SuperpixelDepthSegmenter::calculateStep(const cv::Mat & depth_image)
// {
//     // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::calculateStep]");

//     int width = depth_image.cols;
//     int height = depth_image.rows;
//     int num_pixels = width * height;
    
//     params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval
// }

void SuperpixelDepthSegmenter::reset_data(const cv::Mat & depth_image,
                                            const cv::Mat & label_image,
                                            const cv::Mat & normal_image)
{
    if (params_.warm_start_ && initialized_)
    {
        clusters_ = cv::Mat(depth_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer
        distances_ = cv::Mat(depth_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point

        // Keep centers as is

        center_counts_.assign(center_counts_.size(), 0);
    } else
    {
        clusters_.release();
        distances_.release();
        centers_.clear();
        center_counts_.clear();

        init_data(depth_image, label_image, normal_image);

        initialized_ = true;
    }

}

void SuperpixelDepthSegmenter::init_data(const cv::Mat & depth_image,
                                         const cv::Mat & label_image,
                                         const cv::Mat & normal_image)
{
    /* Initialize the cluster and distance matrices. */
    clusters_ = cv::Mat(depth_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer
    distances_ = cv::Mat(depth_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point

    /* Initialize the centers and counters. */
    // int rough_center_count = 0;
    centers_.clear();
    center_counts_.clear();
    for (int r = params_.step_; r < depth_image.rows - (params_.step_ / 2); r += params_.step_)
    {
        for (int c = params_.step_; c < depth_image.cols - (params_.step_ / 2); c += params_.step_)
        {        
            // ROS_INFO_STREAM("       (r, c): (" << r << ", " << c << ")");

            // float depth = depth_image.at<float>(r, c);

            // if (std::isnan(depth) || std::fabs(depth) < 1e-6)
            // {
            //     continue;
            // }

            std::vector<double> center;

            /* Find the local minimum (gradient-wise). */
            cv::Point originalCenter(c, r);
            cv::Point localMinimum = findLocalMinimum(depth_image, label_image, normal_image, originalCenter);
            float depth = depth_image.at<float>(localMinimum.y, localMinimum.x);
            uint8_t label = label_image.at<uint8_t>(localMinimum.y, localMinimum.x);
            cv::Vec3f normal = normal_image.at<cv::Vec3f>(localMinimum.y, localMinimum.x);

            if (!isPixelValid(depth_image, label_image, normal_image, localMinimum, params_.k_c_))
            {
            //     ROS_INFO_STREAM("       Invalid local minimum found");
            //     ROS_INFO_STREAM("           setting (" << r << ", " << c << ") to default ground floor value ...");
            //     // augment center to be default ground floor value
            //     localMinimum = originalCenter;
            //     depth = 0.385;
            //     label = 3;
            //     normal = cv::Vec3f(0.0, -1.0, 0.0);                
                continue;
            }


            /* Generate the center vector. */
            center.push_back(localMinimum.x);
            center.push_back(localMinimum.y);
            center.push_back(depth);
            center.push_back(label);
            center.push_back(normal.val[0]);
            center.push_back(normal.val[1]);
            center.push_back(normal.val[2]);

            /* Append to vector of centers. */
            centers_.push_back(center);
            center_counts_.push_back(0);
        }
    }

    // ROS_INFO_STREAM("       centers_.size(): " << centers_.size());
    // ROS_INFO_STREAM("       center_counts_.size(): " << center_counts_.size());

}

cv::Point SuperpixelDepthSegmenter::findLocalMinimum(const cv::Mat & depth_image, 
                                                        const cv::Mat & label_image,
                                                        const cv::Mat & normal_image,
                                                        const cv::Point & og_center)
{
    // double min_grad = std::numeric_limits<double>::max();
    cv::Point loc_min(-1, -1);
    // const cv::Point og_center = loc_min; 

    int deltaX = 3; // params_.step_; // 5;
    int deltaY = 3; // params_.step_; // 5;

    for (int r = og_center.y - deltaY; r <= og_center.y + deltaY; r++)
    {
        for (int c = og_center.x - deltaX; c <= og_center.x + deltaX; c++)
        {
            cv::Point current(c, r);

            // if (!isPixelInBounds(depth_image, current))
            // {
            //     continue;
            // }

            // double grad = sqrt(pow(color.val[0] - center_color.val[0], 2) +
            //                    pow(color.val[1] - center_color.val[1], 2) +
            //                    pow(color.val[2] - center_color.val[2], 2));

            if (!isPixelValid(depth_image, label_image, normal_image, current, params_.k_c_))
            {
                continue;
            } else
            {
                float depth = depth_image.at<float>(current.y, current.x);

                if (isPixelInBounds(params_.k_c_, loc_min)) // have found a valid pixel in the region, can compare now
                {
                    if (depth < depth_image.at<float>(loc_min.y, loc_min.x))
                    {
                        loc_min = cv::Point(c, r);
                    }
                } else // have not found a valid pixel in the region yet
                {
                    loc_min = cv::Point(c, r);
                }
            }
        }
    }

    return loc_min;
}

void SuperpixelDepthSegmenter::visualize(const cv::Mat & depth_image,
                                            const cv::Mat & label_image,
                                            const cv::Mat & normal_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::visualize]");

    // std::lock_guard<std::mutex> lock(img_mutex_);

    // if (notReceivedImage())
    // {
    //     ROS_WARN("Not ready to visualize, no images received yet.");
    //     return;
    // }

    // ROS_INFO_STREAM("       Publishing final depth image");
    fin_depth_img_ptr_->header = raw_depth_img_ptr_->header;
    fin_depth_img_ptr_->encoding = raw_depth_img_ptr_->encoding;
    fin_depth_img_ptr_->image = depth_image;
    fin_depth_img_pub_.publish(fin_depth_img_ptr_->toImageMsg());

    // ROS_INFO_STREAM("       Preparing final label image");
    fin_label_img_ptr_->header = raw_label_img_ptr_->header;
    fin_label_img_ptr_->encoding = raw_label_img_ptr_->encoding;
    fin_label_img_ptr_->image = label_image;
    fin_label_img_pub_.publish(fin_label_img_ptr_->toImageMsg());

    // ROS_INFO_STREAM("       Preparing final normal image");
    fin_normal_img_ptr_->header = raw_normal_img_ptr_->header;
    fin_normal_img_ptr_->encoding = raw_normal_img_ptr_->encoding;
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

    overlayCenters(color_depth_image);

    colorClusters(color_depth_image);

    colorClusterPointCloud(depth_image);

    colorCentroids();

    return;
}

void SuperpixelDepthSegmenter::overlayCenters(const cv::Mat & color_depth_image)
{
    // overlay center grid on color version of depth image
    cv::Mat overlaid_image = color_depth_image.clone();

    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(overlaid_image, color);

    center_grid_img_ptr_->header = fin_depth_img_ptr_->header;
    // center_grid_img_ptr_->header.stamp = ros::Time::now();
    center_grid_img_ptr_->encoding = sensor_msgs::image_encodings::BGR8;

    center_grid_img_ptr_->image = overlaid_image;
    center_grid_img_pub_.publish(center_grid_img_ptr_->toImageMsg());
}

void SuperpixelDepthSegmenter::convertDepthImageToColor(cv::Mat & color_depth_image, const cv::Mat & depth_image)
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

void SuperpixelDepthSegmenter::displayCenterGrid(cv::Mat & image, const cv::Vec3b & color)
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::displayCenterGrid]");
    
    // Display center grid
    for (int i = 0; i < (int) centers_.size(); i++) 
    {
        // ROS_INFO_STREAM("       center[" << i << "]: (" << centers_[i][0] << ", " << centers_[i][1] << ")");
        cv::circle(image, cv::Point(centers_[i][0], centers_[i][1]), 2, color, -1);
    }

    return;
}

void SuperpixelDepthSegmenter::colorClusters(const cv::Mat & color_depth_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::colorClusters]");
    // overlay center grid on color version of depth image
    cv::Mat color_cluster_image = color_depth_image.clone();

    // build ector of random colors for clusters
    // std::vector<cv::Scalar> colors(centers_.size());
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

            int cluster_id = clusters_.at<int>(r, c);
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

void SuperpixelDepthSegmenter::colorClusterPointCloud(const cv::Mat & depth_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::colorClusterPointCloud]");

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
            cv::Vec3f worldPt;
            floorPixelToWorld(worldPt, pixel, depth_image.at<float>(r, c), params_.k_c_, params_.h_);

            point.x = worldPt[0];
            point.y = worldPt[1];
            point.z = worldPt[2];

            int cluster_id = clusters_.at<int>(r, c);
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

void SuperpixelDepthSegmenter::colorCentroids()
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::colorCentroids]");

    // clear prior markers
    visualization_msgs::MarkerArray clear_marker_array;
    visualization_msgs::Marker clearMarker;
    clearMarker.id = 0;
    clearMarker.ns =  "clear";
    clearMarker.action = visualization_msgs::Marker::DELETEALL;
    clear_marker_array.markers.push_back(clearMarker);    
    colored_centroids_pub_.publish(clear_marker_array);

    visualization_msgs::MarkerArray marker_array;

    for (int i = 0; i < centers_.size(); i++)
    {
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

        cv::Point center_pixel = cv::Point(centers_[i][0], centers_[i][1]);
        float center_depth = centers_[i][2];
        cv::Vec3f center_normal = cv::Vec3f(centers_[i][4], centers_[i][5], centers_[i][6]);

        cv::Vec3f centerWorldPt;
        floorPixelToWorld(centerWorldPt, center_pixel, center_depth, params_.k_c_, params_.h_);

        marker.points.resize(2);
        double scale = 0.1;
        geometry_msgs::Point p1, p2;
        p1.x = centerWorldPt[0];
        p1.y = centerWorldPt[1];
        p1.z = centerWorldPt[2];
        p2.x = p1.x + scale * center_normal[0];
        p2.y = p1.y + scale * center_normal[1];
        p2.z = p1.z + scale * center_normal[2];

        // ROS_INFO_STREAM("       Centroid " << i << ": (" << p1.x << ", " << p1.y << ", " << p1.z << ")");
        // ROS_INFO_STREAM("       Normal " << i << ": (" << centers_[i][4] << ", " << centers_[i][5] << ", " << centers_[i][6] << ")");

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

// void SuperpixelDepthSegmenter::checkSparsity()
// {

//     int rows = fin_depth_img_ptr_->image.rows;
//     int cols = fin_depth_img_ptr_->image.cols;

//     // Sanity check middle pixel
//     ROS_INFO_STREAM("Depth image size --- rows: " << fin_depth_img_ptr_->image.rows << ", cols: " << fin_depth_img_ptr_->image.cols);
//     ROS_INFO_STREAM("Label image size --- rows: " << label_img_ptr_->image.rows << ", cols: " << label_img_ptr_->image.cols);
//     ROS_INFO_STREAM("Normal image size --- rows: " << normal_img_ptr_->image.rows << ", cols: " << normal_img_ptr_->image.cols);


//     for (int r = 0; r < rows; r++)
//     {
//         for (int c = 0; c < cols; c++)
//         {
//             if (fin_depth_img_ptr_->image.at<float>(r, c) != fin_depth_img_ptr_->image.at<float>(r, c))
//             {
//                 // ROS_ERROR_STREAM("Depth image has NaN value at row: " << r << ", col: " << c);
//                 nan_depth_count++;
//             } else if (fin_depth_img_ptr_->image.at<float>(r, c) == 0)
//             {
//                 zero_depth_count++;
//             } else
//             {
//                 // ROS_INFO_STREAM("Depth image value at row: " << r << ", col: " << c << " is: " << fin_depth_img_ptr_->image.at<float>(r, c));
//                 finite_depth_count++;
//             }

//             if (label_img_ptr_->image.at<uint8_t>(r, c) != label_img_ptr_->image.at<uint8_t>(r, c))
//             {
//                 // ROS_ERROR_STREAM("Label image has NaN value at row: " << r << ", col: " << c);
//                 nan_label_count++;
//             } else if (label_img_ptr_->image.at<uint8_t>(r, c) == 0)
//             {
//                 zero_label_count++;
//             } else
//             {
//                 finite_label_count++;
//             }

//             if (normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] ||
//                 normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] ||
//                 normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2])
//             {
//                 // ROS_ERROR_STREAM("Normal image has NaN value at row: " << r << ", col: " << c);
//                 nan_normal_count++;
//             } else if (normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] == 0 &&
//                        normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] == 0 &&
//                        normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2] == 0)
//             {
//                 zero_normal_count++;
//             } else
//             {
//                 // ROS_INFO_STREAM("Normal image value at row: " << r << ", col: " << c << " is: " << normal_img_ptr_->image.at<cv::Vec3f>(r, c));
//                 finite_normal_count++;
//             }
//         }
//     }

//     int total_pixels = rows * cols;

//     ROS_INFO_STREAM("Depth image NaN count: " << nan_depth_count);
//     ROS_INFO_STREAM("Depth image zero count: " << zero_depth_count);
//     ROS_INFO_STREAM("Depth image finite count: " << finite_depth_count);

//     // ROS_INFO_STREAM("Label image NaN count: " << nan_label_count);
//     // ROS_INFO_STREAM("Label image zero count: " << zero_label_count);
//     // ROS_INFO_STREAM("Label image finite count: " << finite_label_count);

//     // ROS_INFO_STREAM("Normal image NaN count: " << nan_normal_count);
//     // ROS_INFO_STREAM("Normal image zero count: " << zero_normal_count);
//     // ROS_INFO_STREAM("Normal image finite count: " << finite_normal_count);

//     ROS_INFO_STREAM("Depth image sparsity ratio: " << float(finite_depth_count) / total_pixels);
//     // ROS_INFO_STREAM("Label image sparsity ratio: " << float(finite_label_count) / total_pixels);
//     // ROS_INFO_STREAM("Normal image sparsity ratio: " << float(finite_normal_count) / total_pixels);

//     return;    
// }