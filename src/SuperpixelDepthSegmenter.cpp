#include <superpixels/SuperpixelDepthSegmenter.h>

SuperpixelDepthSegmenter::SuperpixelDepthSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    nh_ = nh;

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();
    params_.num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();
    params_.warm_start_ = configYamlNode["superpixels"]["warm_start"].as<bool>();

    ROS_INFO_STREAM("   params_:");
    ROS_INFO_STREAM("       num_superpixels_: " << params_.num_superpixels_);
    ROS_INFO_STREAM("       num_iterations_: " << params_.num_iterations_);
    ROS_INFO_STREAM("       warm_start_: " << params_.num_iterations_);
    ROS_INFO_STREAM("       w_normal_: " << params_.w_normal_);
    ROS_INFO_STREAM("       w_pos_: " << params_.w_pos_);
    ROS_INFO_STREAM("       k_c_: " << params_.k_c_);
    ROS_INFO_STREAM("       v_fov_: " << params_.v_fov_);
    ROS_INFO_STREAM("       v_offset_: " << params_.v_offset_);
    ROS_INFO_STREAM("       h_: " << params_.h_);

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
    fin_normal_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    center_grid_img_pub_ = it.advertise("/superpixels/center_grid", 1);
    center_grid_img_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
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

    if (std::isnan(depth) || std::fabs(depth) < 1e-6)
    {
        return false;
    }

    uint8_t label = label_image.at<uint8_t>(pixel.y, pixel.x);

    if (std::isnan(label) || label < 0)
    {
        ROS_WARN_STREAM("Passing depth check but failing label check.");
        return false;
    }

    cv::Vec3b normal = normal_image.at<cv::Vec3b>(pixel.y, pixel.x);

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
        depth_img_ptr_ = cv_bridge::toCvCopy(raw_depth_img_msg_, sensor_msgs::image_encodings::TYPE_32FC1);
        label_img_ptr_ = cv_bridge::toCvCopy(raw_label_img_msg_, sensor_msgs::image_encodings::TYPE_8UC1);
        normal_img_ptr_ = cv_bridge::toCvCopy(raw_normal_img_msg_, sensor_msgs::image_encodings::TYPE_32FC3);

    } catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if (depth_img_ptr_->image.size() != label_img_ptr_->image.size() || 
        depth_img_ptr_->image.size() != normal_img_ptr_->image.size() ||
        label_img_ptr_->image.size() != normal_img_ptr_->image.size())
    {
        ROS_ERROR("Image sizes do not match.");
        return;
    }

    std::chrono::steady_clock::time_point timeBegin, timeEnd;
    timeBegin = std::chrono::steady_clock::now();

    // Pre-processing
    preprocessImages();

    // checkSparsity();

    // cv::Mat depth_image = depth_img_ptr_->image;
    // cv::Mat label_image = label_img_ptr_->image;
    // cv::Mat normal_image = normal_img_ptr_->image;

    // if (!initialized_)
    // {

    // Pre-processing
    calculateStep(fin_depth_img_ptr_->image);

    // Initialize data
    init_data(fin_depth_img_ptr_->image,
              fin_label_img_ptr_->image,
              fin_normal_img_ptr_->image);

    //     initialized_ = true;
    // }

    // Generate superpixels
    generateSuperpixels(fin_depth_img_ptr_->image,
                        fin_label_img_ptr_->image,
                        fin_normal_img_ptr_->image);

    timeEnd = std::chrono::steady_clock::now();
    int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count();
    double total_time_sec = total_time / 1.0e6; 
    ROS_INFO_STREAM("Superpixels took: " << total_time_sec << " seconds");

    return;
}

void SuperpixelDepthSegmenter::generateSuperpixels(const cv::Mat & depth_image,
                                                    const cv::Mat & label_image,
                                                    const cv::Mat & normal_image)
{
    // Generate superpixels
    for (int i = 0; i < params_.num_iterations_; i++)
    {
        /* Reset distance and cluster values. */
        distances_ = cv::Mat(depth_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max()));
        clusters_ = cv::Mat(depth_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer

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
                        cv::Vec3b normal = normal_img_ptr_->image.at<cv::Vec3b>(r, c);

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
                    cv::Vec3b normal = normal_img_ptr_->image.at<cv::Vec3b>(r, c);

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
}

void SuperpixelDepthSegmenter::floorPixelToWorld(cv::Vec3b & worldPt,
                                                    const cv::Point & pixel,
                                                    const float & depth)
{
    worldPt[0] = (pixel.x - params_.k_c_) * (depth / (params_.h_ * params_.k_c_));
    worldPt[1] = depth;
    worldPt[2] = (pixel.y - params_.k_c_) * (depth / (params_.h_ * params_.k_c_));
}

double SuperpixelDepthSegmenter::computeDistance(const int & center_idx, 
                                                    const float & depth,
                                                    const uint8_t & label,
                                                    const cv::Vec3b & normal,
                                                    const cv::Point & pixel)
{
    float center_depth = centers_[center_idx][2];
    uint8_t center_label = centers_[center_idx][3];
    cv::Vec3b center_normal = cv::Vec3b(centers_[center_idx][4], centers_[center_idx][5], centers_[center_idx][6]);

    // Normal term
    double d_normal = params_.w_normal_ * (1.0 - normal.dot(center_normal));
    // double dc = sqrt(pow(color.val[0] - centers_[center_idx][0], 2) +
    //                  pow(color.val[1] - centers_[center_idx][1], 2) +
    //                  pow(color.val[2] - centers_[center_idx][2], 2));

    // Position term
    cv::Vec3b worldPt;
    floorPixelToWorld(worldPt, pixel, depth);

    cv::Vec3b centerWorldPt;
    floorPixelToWorld(centerWorldPt, cv::Point(centers_[center_idx][0], centers_[center_idx][1]), center_depth);
    double d_posn = params_.w_pos_ * std::fabs( (centerWorldPt - worldPt).dot(center_normal) );

    // // Spatial term
    // double ds = sqrt(pow(pixel.x - centers_[center_idx][3], 2) +
    //                  pow(pixel.y - centers_[center_idx][4], 2));

    // return sqrt(pow(dc / params_.n_c_, 2) + pow(ds / params_.n_s_, 2));
    return d_normal + d_posn;
}


void SuperpixelDepthSegmenter::preprocessImages()
{
    // Depth
    fin_depth_img_ptr_->header = depth_img_ptr_->header;
    fin_depth_img_ptr_->encoding = depth_img_ptr_->encoding;

    cv::Mat dilated_depth_img;
    dilate_depth_image(depth_img_ptr_->image, dilated_depth_img);
    fin_depth_img_ptr_->image = dilated_depth_img;

    // Label
    fin_label_img_ptr_->header = label_img_ptr_->header;
    fin_label_img_ptr_->encoding = label_img_ptr_->encoding;

    cv::Mat dilated_label_img;
    dilate_depth_image(label_img_ptr_->image, dilated_label_img);
    fin_label_img_ptr_->image = dilated_label_img;

    // Normal
    fin_normal_img_ptr_->header = normal_img_ptr_->header;
    fin_normal_img_ptr_->encoding = normal_img_ptr_->encoding;

    cv::Mat dilated_normal_img;
    dilate_depth_image(normal_img_ptr_->image, dilated_normal_img);
    fin_normal_img_ptr_->image = dilated_normal_img;
}

void SuperpixelDepthSegmenter::dilate_depth_image(const cv::Mat & image, cv::Mat & dilated_image)
{
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(image, dilated_image, element, cv::Point(-1, -1), 3);
}

void SuperpixelDepthSegmenter::calculateStep(const cv::Mat & depth_image)
{
    // ROS_INFO_STREAM("   [SuperpixelDepthSegmenter::calculateStep]");

    int width = depth_image.cols;
    int height = depth_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double) params_.num_superpixels_); // superpixel grid interval
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
            cv::Vec3b normal = normal_image.at<cv::Vec3b>(localMinimum.y, localMinimum.x);

            // cv::Vec3b color = lab_image.at<cv::Vec3b>(localMinimum.y, localMinimum.x);

            if (!isPixelValid(depth_image, label_image, normal_image, localMinimum))
            {
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
            // center.push_back(depth);

            /* Append to vector of centers. */
            centers_.push_back(center);
            center_counts_.push_back(0);
        }
    }

    ROS_INFO_STREAM("       centers_.size(): " << centers_.size());
    ROS_INFO_STREAM("       center_counts_.size(): " << center_counts_.size());

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

void SuperpixelDepthSegmenter::visualize()
{
    std::lock_guard<std::mutex> lock(img_mutex_);

    if (notReceivedImage())
    {
        ROS_WARN("Not ready to visualize, no images received yet.");
        return;
    }

    fin_depth_img_pub_.publish(fin_depth_img_ptr_->toImageMsg());

    convertDepthImageToColor();

    overlayCenters();

    colorClusters();

    return;
}

void SuperpixelDepthSegmenter::overlayCenters()
{
    // overlay center grid on color version of depth image
    cv::Mat overlaid_image = color_depth_image_.clone();

    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(overlaid_image, color);

    center_grid_img_ptr_->header = fin_depth_img_ptr_->header;
    center_grid_img_ptr_->header.stamp = ros::Time::now();
    center_grid_img_ptr_->encoding = sensor_msgs::image_encodings::BGR8;

    center_grid_img_ptr_->image = overlaid_image;
    center_grid_img_pub_.publish(center_grid_img_ptr_->toImageMsg());
}

void SuperpixelDepthSegmenter::convertDepthImageToColor()
{
    color_depth_image_ = cv::Mat(fin_depth_img_ptr_->image.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    double min_depth = 0.0, max_depth = 0.0;
    cv::minMaxLoc(fin_depth_img_ptr_->image, &min_depth, &max_depth);

    // ROS_INFO_STREAM("Converting to 8UC3...");

    for (int r = 0; r < color_depth_image_.rows; r++)
    {
        for (int c = 0; c < color_depth_image_.cols; c++)
        {
            float depth = depth_img_ptr_->image.at<float>(r, c);

            if (std::isnan(depth) || std::fabs(depth) < 1e-6)
            {
                continue;
            }

            // ROS_INFO_STREAM("   (r, c): (" << r << ", " << c << ")");
            // ROS_INFO_STREAM("       depth: " << depth);

            int quantized_depth = (int) (depth * 255.0 / max_depth); // just scaling by max depth in image. If we do full max depth than image is really hard to see.

            cv::Vec3b color = cv::Vec3b(quantized_depth, quantized_depth, quantized_depth);
            color_depth_image_.at<cv::Vec3b>(r, c) = color;
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

void SuperpixelDepthSegmenter::colorClusters()
{
    // build ector of random colors for clusters
    std::vector<cv::Scalar> colors(centers_.size());
    for (int i = 0; i < (int) colors.size(); i++)
    {
        colors[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    }

    // iterate through valid pixels and color
    for (int c = 0; c < fin_depth_img_ptr_->image.cols; c++)
    {
        for (int r = 0; r < fin_depth_img_ptr_->image.rows; r++)
        {
            int cluster_id = clusters_.at<int>(r, c);
            if (cluster_id != -1)
            {
                cv::Scalar color = colors[cluster_id];
                color_depth_image_.at<cv::Vec3b>(r, c) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
    
}

void SuperpixelDepthSegmenter::checkSparsity()
{

    int rows = fin_depth_img_ptr_->image.rows;
    int cols = fin_depth_img_ptr_->image.cols;

    // Sanity check middle pixel
    ROS_INFO_STREAM("Depth image size --- rows: " << fin_depth_img_ptr_->image.rows << ", cols: " << fin_depth_img_ptr_->image.cols);
    ROS_INFO_STREAM("Label image size --- rows: " << label_img_ptr_->image.rows << ", cols: " << label_img_ptr_->image.cols);
    ROS_INFO_STREAM("Normal image size --- rows: " << normal_img_ptr_->image.rows << ", cols: " << normal_img_ptr_->image.cols);

    int nan_depth_count = 0;
    int nan_label_count = 0;
    int nan_normal_count = 0;

    int finite_depth_count = 0;
    int finite_label_count = 0;
    int finite_normal_count = 0;

    int zero_depth_count = 0;
    int zero_label_count = 0;
    int zero_normal_count = 0;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (fin_depth_img_ptr_->image.at<float>(r, c) != fin_depth_img_ptr_->image.at<float>(r, c))
            {
                // ROS_ERROR_STREAM("Depth image has NaN value at row: " << r << ", col: " << c);
                nan_depth_count++;
            } else if (fin_depth_img_ptr_->image.at<float>(r, c) == 0)
            {
                zero_depth_count++;
            } else
            {
                // ROS_INFO_STREAM("Depth image value at row: " << r << ", col: " << c << " is: " << fin_depth_img_ptr_->image.at<float>(r, c));
                finite_depth_count++;
            }

            if (label_img_ptr_->image.at<uint8_t>(r, c) != label_img_ptr_->image.at<uint8_t>(r, c))
            {
                // ROS_ERROR_STREAM("Label image has NaN value at row: " << r << ", col: " << c);
                nan_label_count++;
            } else if (label_img_ptr_->image.at<uint8_t>(r, c) == 0)
            {
                zero_label_count++;
            } else
            {
                finite_label_count++;
            }

            if (normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] ||
                normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] ||
                normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2] != normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2])
            {
                // ROS_ERROR_STREAM("Normal image has NaN value at row: " << r << ", col: " << c);
                nan_normal_count++;
            } else if (normal_img_ptr_->image.at<cv::Vec3f>(r, c)[0] == 0 &&
                       normal_img_ptr_->image.at<cv::Vec3f>(r, c)[1] == 0 &&
                       normal_img_ptr_->image.at<cv::Vec3f>(r, c)[2] == 0)
            {
                zero_normal_count++;
            } else
            {
                // ROS_INFO_STREAM("Normal image value at row: " << r << ", col: " << c << " is: " << normal_img_ptr_->image.at<cv::Vec3f>(r, c));
                finite_normal_count++;
            }
        }
    }

    int total_pixels = rows * cols;

    ROS_INFO_STREAM("Depth image NaN count: " << nan_depth_count);
    ROS_INFO_STREAM("Depth image zero count: " << zero_depth_count);
    ROS_INFO_STREAM("Depth image finite count: " << finite_depth_count);

    // ROS_INFO_STREAM("Label image NaN count: " << nan_label_count);
    // ROS_INFO_STREAM("Label image zero count: " << zero_label_count);
    // ROS_INFO_STREAM("Label image finite count: " << finite_label_count);

    // ROS_INFO_STREAM("Normal image NaN count: " << nan_normal_count);
    // ROS_INFO_STREAM("Normal image zero count: " << zero_normal_count);
    // ROS_INFO_STREAM("Normal image finite count: " << finite_normal_count);

    ROS_INFO_STREAM("Depth image sparsity ratio: " << float(finite_depth_count) / total_pixels);
    // ROS_INFO_STREAM("Label image sparsity ratio: " << float(finite_label_count) / total_pixels);
    // ROS_INFO_STREAM("Normal image sparsity ratio: " << float(finite_normal_count) / total_pixels);

    return;    
}