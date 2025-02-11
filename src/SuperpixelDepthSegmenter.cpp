#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    nh_ = nh;

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

    prop_depth_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    prop_label_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    prop_normal_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    visited_ = cv::Mat(params_.k_c_, params_.k_c_, CV_8UC1, cv::Scalar(0));

    imagePreprocessor_ = new ImagePreprocessor(params_);
    visualizer_ = new Visualizer(params_, nh);
    ransac_ = new Ransac(params_);
}

SuperpixelDepthSegmenter::~SuperpixelDepthSegmenter()
{
    // delete tfListener_;
    delete imagePreprocessor_;
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

    imagePreprocessor_->setParams(params_);
    visualizer_->setParams(params_);
    ransac_->setParams(params_);
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
    imagePreprocessor_->cleanImages(raw_depth_img, raw_label_img, raw_normal_img, 
                                    cleaned_depth_img, cleaned_label_img, cleaned_normal_img, visited_);

    // ROS_INFO_STREAM(" after cleaning, (480, 480) is: " << cleaned_depth_img.at<float>(480, 480));

    cleanEnd = std::chrono::steady_clock::now();
    int64_t clean_total_time = std::chrono::duration_cast<std::chrono::microseconds>(cleanEnd - cleanBegin).count();
    double clean_total_time_sec = clean_total_time / 1.0e6; 

    fillBegin = std::chrono::steady_clock::now();

    cv::Mat filled_depth_img, filled_label_img, filled_normal_img;
    imagePreprocessor_->fillInImage(cleaned_depth_img, cleaned_label_img, cleaned_normal_img, visited_,
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
    imagePreprocessor_->preprocessImages(filled_depth_img, filled_label_img, filled_normal_img,
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
    visualizer_->visualize(preprocessed_depth_img, 
                            preprocessed_label_img, 
                            preprocessed_normal_img,
                            raw_depth_img_ptr_,
                            raw_label_img_ptr_,
                            raw_normal_img_ptr_,
                            centers_,
                            clusters_);

    // Publish planar regions
    // visualizer_->publishPlanarRegions(preprocessed_depth_img, centers_);

    return;
}


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
            
            // cv::Vec3f candidate_normal = ransac(superpixels_[j], depth_image, normal);
            cv::Vec3f candidate_normal = ransac_->run(superpixels_[j], depth_image, normal);

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