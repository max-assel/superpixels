#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    nh_ = nh;

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();
    params_.num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();
    params_.num_dilation_iterations_ = configYamlNode["superpixels"]["num_dilation_iterations"].as<int>();
    params_.w_normal_ = configYamlNode["superpixels"]["w_normal"].as<double>();
    params_.w_pos_ = configYamlNode["superpixels"]["w_pos"].as<double>();
    params_.w_compact_ = configYamlNode["superpixels"]["w_compact"].as<double>();
    params_.warm_start_ = configYamlNode["superpixels"]["warm_start"].as<bool>();
    params_.kernel_radius_ = configYamlNode["superpixels"]["kernel_radius"].as<int>();

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
        colors_[i] = cv::Scalar(rand() % 255,
                                rand() % 255, 
                                rand() % 255);
    }

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

bool SuperpixelDepthSegmenter::isPixelInBounds(const cv::Mat & image, const cv::Point & pixel)
{
    if (pixel.x < 0 || pixel.x >= image.cols || pixel.y < 0 || pixel.y >= image.rows)
    {
        return false;
    }

    return true;
}

bool SuperpixelDepthSegmenter::isPixelValid(const cv::Mat & depth_image, 
                                            const cv::Mat & label_image,
                                            const cv::Mat & normal_image,
                                            const cv::Point & pixel)
{
    if (!isPixelInBounds(depth_image, pixel))
    {
        return false;
    }

    float depth = depth_image.at<float>(pixel.y, pixel.x);

    if (std::isnan(depth) || std::fabs(depth) < 1e-6 || depth < 0)
    {
        return false;
    }

    // uint8_t label = label_image.at<uint8_t>(pixel.y, pixel.x);

    // if (std::isnan(label) || label < 0)
    // {
    //     ROS_WARN_STREAM("Passing depth check but failing label check.");
    //     return false;
    // }

    cv::Vec3f normal = normal_image.at<cv::Vec3f>(pixel.y, pixel.x);

    if (cv::norm(normal) < DELTA)
    {
        ROS_WARN_STREAM("Passing depth check but failing normal check.");
        return false;
    }

    return true;
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

    // Health check
    // healthCheck(raw_depth_img, raw_label_img, raw_normal_img);

    // Clean images
    cv::Mat cleaned_depth_img, cleaned_label_img, cleaned_normal_img;
    cleanImages(raw_depth_img, raw_label_img, raw_normal_img, 
                cleaned_depth_img, cleaned_label_img, cleaned_normal_img);

    cleanEnd = std::chrono::steady_clock::now();
    int64_t clean_total_time = std::chrono::duration_cast<std::chrono::microseconds>(cleanEnd - cleanBegin).count();
    double clean_total_time_sec = clean_total_time / 1.0e6; 

    // Health check
    // healthCheck(cleaned_depth_img, cleaned_label_img, cleaned_normal_img);

    preprocessBegin = std::chrono::steady_clock::now();

    // Pre-processing
    cv::Mat preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img;
    preprocessImages(cleaned_depth_img, cleaned_label_img, cleaned_normal_img,
                        preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    preprocessEnd = std::chrono::steady_clock::now();
    int64_t preprocess_total_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessBegin).count();
    double preprocess_total_time_sec = preprocess_total_time / 1.0e6; 

    // Health check
    // healthCheck(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    // checkSparsity();

    // if (!initialized_)
    // {

    superpixelBegin = std::chrono::steady_clock::now();

    // Pre-processing
    calculateStep(preprocessed_depth_img);

    // Clear data
    reset_data(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    // Initialize data
    init_data(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    //     initialized_ = true;
    // }

    // Generate superpixels
    generateSuperpixels(preprocessed_depth_img, preprocessed_label_img, preprocessed_normal_img);

    superpixelEnd = std::chrono::steady_clock::now();
    int64_t superpixel_total_time = std::chrono::duration_cast<std::chrono::microseconds>(superpixelEnd - superpixelBegin).count();
    double superpixel_total_time_sec = superpixel_total_time / 1.0e6; 

    ROS_INFO_STREAM_THROTTLE(3, "Timing ---- \n" << 
                                "   Image cleaning took: " << clean_total_time_sec << " seconds, \n" <<
                                "   Image preprocessing took " << preprocess_total_time_sec << " seconds, \n" <<
                                "   Superpixels took " << superpixel_total_time_sec << " seconds, \n" << 
                                "   Total: " << clean_total_time_sec + preprocess_total_time_sec + superpixel_total_time_sec << " seconds");

    // Visualize
    visualize(preprocessed_depth_img, 
                preprocessed_label_img, 
                preprocessed_normal_img);

    return;
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

    // zero invalid depth pixels
    for (int r = 0; r < cleaned_depth_img.rows; r++)
    {
        for (int c = 0; c < cleaned_depth_img.cols; c++)
        {
            float depth = raw_depth_img.at<float>(r, c);

            if (std::isnan(depth))
                cleaned_depth_img.at<float>(r, c) = 0;
            else if (std::fabs(depth) < 1e-6)
                cleaned_depth_img.at<float>(r, c) = 0;
            else if (depth < 0)
                cleaned_depth_img.at<float>(r, c) = 0;
            else
                cleaned_depth_img.at<float>(r, c) = depth;
        }
    }    

    // ROS_INFO_STREAM("       Checking label image ...");

    // Labels
    cleaned_label_img = cv::Mat(raw_label_img.size(), CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < cleaned_label_img.rows; r++)
    {
        for (int c = 0; c < cleaned_label_img.cols; c++)
        {
            uint8_t label = raw_label_img.at<uint8_t>(r, c);

            if (std::isnan(label))
                cleaned_label_img.at<uint8_t>(r, c) = 0;
            else if (label < 0)
                cleaned_label_img.at<uint8_t>(r, c) = 0;
            else
                cleaned_label_img.at<uint8_t>(r, c) = label;
        }
    }

    // ROS_INFO_STREAM("       Checking normal image ...");

    // Normals
    cleaned_normal_img = cv::Mat(raw_normal_img.size(), CV_32FC3, cv::Scalar(0));
    for (int r = 0; r < cleaned_normal_img.rows; r++)
    {
        for (int c = 0; c < cleaned_normal_img.cols; c++)
        {
            cv::Vec3f normal = raw_normal_img.at<cv::Vec3f>(r, c);

            if (std::isnan(normal.val[0]) || std::isnan(normal.val[1]) || std::isnan(normal.val[2]))
                cleaned_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
            else if (cv::norm(normal) < DELTA)
                cleaned_normal_img.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
            else
                cleaned_normal_img.at<cv::Vec3f>(r, c) = normal;
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
            } else if (std::fabs(depth) < 1e-6)
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

                        if (!isPixelInBounds(dilated_depth_img, cv::Point(new_c, new_r)))
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
            for (int c = centers_[j][0] - params_.step_; c < centers_[j][0] + params_.step_; c++) 
            {
                for (int r = centers_[j][1] - params_.step_; r < centers_[j][1] + params_.step_; r++) 
                {
                    cv::Point current(c, r);
                    if (isPixelValid(depth_image, label_image, normal_image, current)) 
                    {
                        float depth = depth_image.at<float>(r, c);
                        uint8_t label = label_image.at<uint8_t>(r, c);
                        cv::Vec3f normal = normal_image.at<cv::Vec3f>(r, c);

                        double d = computeDistance(j, 
                                                    depth,
                                                    label,
                                                    normal,
                                                    current);

                        if (d < distances_.at<double>(r, c)) 
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
            centers_[j][2] = 0;
            centers_[j][3] = 0;
            centers_[j][4] = 0;
            centers_[j][5] = 0;
            centers_[j][6] = 0;
            center_counts_[j] = 0;
        }

        /* Compute the new cluster centers. */
        for (int c = 0; c < depth_image.cols; c++) 
        {
            for (int r = 0; r < depth_image.rows; r++) 
            {
                int cluster_id = clusters_.at<int>(r, c);
                
                if (cluster_id != -1) 
                {
                    float depth = depth_image.at<float>(r, c);
                    uint8_t label = label_image.at<uint8_t>(r, c);
                    cv::Vec3f normal = normal_image.at<cv::Vec3f>(r, c);

                    centers_[cluster_id][0] += c;
                    centers_[cluster_id][1] += r;
                    centers_[cluster_id][2] += depth;
                    centers_[cluster_id][3] += label;
                    centers_[cluster_id][4] += normal.val[0];
                    centers_[cluster_id][5] += normal.val[1];
                    centers_[cluster_id][6] += normal.val[2];
                    
                    center_counts_[cluster_id] += 1;
                }
            }
        }     

        /* Normalize the clusters. */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            centers_[j][0] /= center_counts_[j];
            centers_[j][1] /= center_counts_[j];
            centers_[j][2] /= center_counts_[j];
            centers_[j][3] /= center_counts_[j];
            centers_[j][4] /= center_counts_[j];
            centers_[j][5] /= center_counts_[j];
            centers_[j][6] /= center_counts_[j];
        }
    }

    // ROS_INFO_STREAM("       center_counts:");
    // for (int i = 0; i < (int) center_counts_.size(); i++)
    // {
    //     ROS_INFO_STREAM("           center_counts_[" << i << "]: " << center_counts_[i]);
    // }
}

void SuperpixelDepthSegmenter::floorPixelToWorld(cv::Vec3f & worldPt,
                                                    const cv::Point & pixel,
                                                    const float & depth)
{
    worldPt[0] = (pixel.x - (params_.k_c_ / 2)) * (depth * 2 / (params_.h_ * params_.k_c_));
    worldPt[1] = depth;
    worldPt[2] = (pixel.y - (params_.k_c_ / 2)) * (depth * 2 / (params_.h_ * params_.k_c_));
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
    floorPixelToWorld(worldPt, pixel, depth);

    cv::Vec3f centerWorldPt;
    floorPixelToWorld(centerWorldPt, center_pixel, center_depth);

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

    double d_posn = std::fabs( (centerWorldPt - worldPt).dot(center_normal) );
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

    // return sqrt(pow(dc / params_.n_c_, 2) + pow(ds / params_.n_s_, 2));
    return weighted_d_normal + weighted_d_posn + weighted_d_compact;
}

void SuperpixelDepthSegmenter::calculateStep(const cv::Mat & depth_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::calculateStep]");

    int width = depth_image.cols;
    int height = depth_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval
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
    for (int c = params_.step_; c < depth_image.cols - (params_.step_ / 2); c += params_.step_)
    {
        for (int r = params_.step_; r < depth_image.rows - (params_.step_ / 2); r += params_.step_)
        {
            // ROS_INFO_STREAM("       (r, c): (" << r << ", " << c << ")");

            // float depth = depth_image.at<float>(r, c);

            // if (std::isnan(depth) || std::fabs(depth) < 1e-6)
            // {
            //     continue;
            // }

            std::vector<double> center;

            /* Find the local minimum (gradient-wise). */
            cv::Point originalCenter(c, r); // initialize to an invalid pixel
            cv::Point localMinimum = findLocalMinimum(depth_image, label_image, normal_image, originalCenter);
            float depth = depth_image.at<float>(localMinimum.y, localMinimum.x);
            uint8_t label = label_image.at<uint8_t>(localMinimum.y, localMinimum.x);
            cv::Vec3f normal = normal_image.at<cv::Vec3f>(localMinimum.y, localMinimum.x);

            // if (!isPixelValid(depth_image, label_image, normal_image, localMinimum))
            // {
            //     continue;
            // }


            /* Generate the center vector. */
            center.push_back(localMinimum.x);
            center.push_back(localMinimum.y);
            center.push_back(depth);
            center.push_back(label);
            center.push_back(normal.val[0]);
            center.push_back(normal.val[1]);
            center.push_back(normal.val[2]);
            // center.push_back(depth);

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

    int deltaX = (params_.step_ / 2); // 5;
    int deltaY = (params_.step_ / 2); // 5;

    for (int c = og_center.x - deltaX; c <= og_center.x + deltaX; c++)
    {
        for (int r = og_center.y - deltaY; r <= og_center.y + deltaY; r++)
        {
            cv::Point current(c, r);

            if (!isPixelInBounds(depth_image, current))
            {
                continue;
            }

            float depth = depth_image.at<float>(current.y, current.x);

            // double grad = sqrt(pow(color.val[0] - center_color.val[0], 2) +
            //                    pow(color.val[1] - center_color.val[1], 2) +
            //                    pow(color.val[2] - center_color.val[2], 2));

            if (!isPixelValid(depth_image, label_image, normal_image, current))
            {
                continue;
            } else
            {
                if (isPixelInBounds(depth_image, loc_min)) // have found a valid pixel in the region, can compare now
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
    // No publishing label image

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

    return;
}

void SuperpixelDepthSegmenter::overlayCenters(const cv::Mat & color_depth_image)
{
    // overlay center grid on color version of depth image
    cv::Mat overlaid_image = color_depth_image.clone();

    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(overlaid_image, color);

    center_grid_img_ptr_->header = fin_depth_img_ptr_->header;
    center_grid_img_ptr_->header.stamp = ros::Time::now();
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

            if (std::isnan(depth) || std::fabs(depth) < 1e-6)
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
    for (int c = 0; c < color_depth_image.cols; c++)
    {
        for (int r = 0; r < color_depth_image.rows; r++)
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
    colored_cluster_img_ptr_->header.stamp = ros::Time::now();
    colored_cluster_img_ptr_->encoding = sensor_msgs::image_encodings::BGR8;

    colored_cluster_img_ptr_->image = color_cluster_image;
    colored_cluster_img_pub_.publish(colored_cluster_img_ptr_->toImageMsg());

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