/*******************************
* Adapted from: https://github.com/PSMM/SLIC-Superpixels/tree/master
*  - To use with cv::Mat and ROS
*
*
 */
#include <superpixels/SuperpixelColorSegmenter.h>

SuperpixelColorSegmenter::SuperpixelColorSegmenter(ros::NodeHandle nh, const std::string & config_path)
{
    // ROS_INFO_STREAM("[SuperpixelColorSegmenter::SuperpixelColorSegmenter]");

    nh_ = nh;

    image_transport::ImageTransport it(nh);

    // Load configs
    YAML::Node configYamlNode = YAML::LoadFile(config_path);

    params_.num_superpixels_ = configYamlNode["superpixels"]["num_superpixels"].as<int>();
    params_.n_c_ = configYamlNode["superpixels"]["n_c"].as<int>();
    params_.n_s_ = configYamlNode["superpixels"]["n_s"].as<int>();
    params_.num_iterations_ = configYamlNode["superpixels"]["num_iterations"].as<int>();
    params_.warm_start_ = configYamlNode["superpixels"]["warm_start"].as<bool>();

    ROS_INFO_STREAM("   params_:");
    ROS_INFO_STREAM("       num_superpixels_: " << params_.num_superpixels_);
    ROS_INFO_STREAM("       n_c_: " << params_.n_c_);
    ROS_INFO_STREAM("       n_s_: " << params_.n_s_);
    ROS_INFO_STREAM("       num_iterations_: " << params_.num_iterations_);
    ROS_INFO_STREAM("       warm_start_: " << params_.num_iterations_);

    color_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    // read in example image
    const std::string superixels_path = ros::package::getPath("superpixels");
    const std::string image_name = "flower.jpg";

    color_image_ptr_->header.stamp = ros::Time::now();
    color_image_ptr_->header.frame_id = "N/A";
    color_image_ptr_->header.seq = 0;
    color_image_ptr_->encoding = sensor_msgs::image_encodings::BGR8;
    color_image_ptr_->image = cv::imread(superixels_path + "/images/" + image_name, cv::IMREAD_COLOR);

    if (color_image_ptr_->image.empty())
    {
        ROS_ERROR_STREAM("Could not read the image: " << image_name);
        return;
    } else
    {
        ROS_INFO_STREAM("Read the image: " << image_name);
    }

    center_grid_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    center_grid_image_ptr_->header = color_image_ptr_->header;
    center_grid_image_ptr_->encoding = color_image_ptr_->encoding;

    overlay_image_ptr_ = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    overlay_image_ptr_->header = color_image_ptr_->header;
    overlay_image_ptr_->encoding = color_image_ptr_->encoding;

    color_image_pub_ = it.advertise("/superpixels/color", 1);
    center_grid_image_pub_ = it.advertise("/superpixels/center_grid", 1);
    overlay_image_pub_ = it.advertise("/superpixels/overlaid_color", 1);

    // Pre-processing
    preprocessing();

    // Initialize data
    init_data(); 

    // ros::Duration(5.0).sleep(); // sleep for half a second

    return;
}

void SuperpixelColorSegmenter::run()
{
    // ROS_INFO_STREAM("[SuperpixelColorSegmenter::run]");

    // std::chrono::steady_clock::time_point timeBegin, timeEnd;
    // timeBegin = std::chrono::steady_clock::now();

    // Clear data
    reset_data();

    // Superpixels algorithm
    generateSuperpixels();

    // Create connectivity
    // createConnectivity();

    // timeEnd = std::chrono::steady_clock::now();
    // int64_t total_time = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count();
    // double total_time_sec = total_time / 1.0e6; 
    // ROS_INFO_STREAM("Superpixels took: " << total_time_sec << " seconds");

    return;
}

void SuperpixelColorSegmenter::visualize()
{

    // Display original image
    color_image_ptr_->header.stamp = ros::Time::now();
    color_image_pub_.publish(color_image_ptr_->toImageMsg());

    // Display center grid
    cv::Mat center_grid_image = color_image_ptr_->image.clone();
    cv::Vec3b color(255, 0, 255);
    displayCenterGrid(center_grid_image, color);
    center_grid_image_ptr_->image = center_grid_image;
    center_grid_image_ptr_->header.stamp = ros::Time::now();
    center_grid_image_pub_.publish(center_grid_image_ptr_->toImageMsg());

    // Display contours
    cv::Mat contours = cv::Mat::zeros(lab_image.size(), CV_8U);
    displayContours(contours);

    // Display clusters with cluster mean colors
    cv::Mat overlaidContours = color_image_ptr_->image.clone();
    displaySuperpixelsWithClusterMeans(overlaidContours);

    // Overlay contours
    cv::cvtColor(overlaidContours, overlaidContours, cv::COLOR_Lab2BGR);
    overlaidContours.setTo(cv::Scalar(255,255,255), contours);
    overlay_image_ptr_->image = overlaidContours;
    overlay_image_ptr_->header.stamp = ros::Time::now();
    overlay_image_pub_.publish(overlay_image_ptr_->toImageMsg());

    // ros::Duration(1.0).sleep(); // sleep for half a second

    return;
}

void SuperpixelColorSegmenter::preprocessing()
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::preprocessing]");

    cv::cvtColor(color_image_ptr_->image, lab_image, cv::COLOR_BGR2Lab);
    int width = lab_image.cols;
    int height = lab_image.rows;
    int num_pixels = width * height;
    
    params_.step_ = sqrt(num_pixels / (double)params_.num_superpixels_); // superpixel grid interval
}

void SuperpixelColorSegmenter::generateSuperpixels()
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::generateSuperpixels]");
    
    // Generate superpixels
    for (int i = 0; i < params_.num_iterations_; i++)
    {
        /* Reset distance values. */
        distances_ = cv::Mat(lab_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max()));

        /* Update distances and clusters */
        for (int j = 0; j < (int) centers_.size(); j++) 
        {
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int k = centers_[j][3] - params_.step_; k < centers_[j][3] + params_.step_; k++) 
            {
                for (int l = centers_[j][4] - params_.step_; l < centers_[j][4] + params_.step_; l++) 
                {
                    if (k >= 0 && k < lab_image.cols && l >= 0 && l < lab_image.rows) 
                    {
                        cv::Vec3b color = lab_image.at<cv::Vec3b>(l, k);
                        double d = computeDistance(j, color, cv::Point(k, l));

                        if (d < distances_.at<double>(l, k)) 
                        {
                            distances_.at<double>(l, k) = d;
                            clusters_.at<int>(l, k) = j;
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
        for (int c = 0; c < lab_image.cols; c++) 
        {
            for (int r = 0; r < lab_image.rows; r++) 
            {
                int c_id = clusters_.at<int>(r, c);
                
                if (c_id != -1) 
                {
                    cv::Vec3b color = lab_image.at<cv::Vec3b>(r, c);
                    
                    centers_[c_id][0] += color.val[0];
                    centers_[c_id][1] += color.val[1];
                    centers_[c_id][2] += color.val[2];
                    centers_[c_id][3] += c;
                    centers_[c_id][4] += r;
                    
                    center_counts_[c_id] += 1;
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
        }           
    }

    return;
}

void SuperpixelColorSegmenter::createConnectivity()
{
    int label = 0, adjlabel = 0;
    const int lims = (lab_image.cols * lab_image.rows) / centers_.size();
    
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};       

    // Create connectivity

    /* Initialize the new cluster matrix. */
    cv::Mat new_clusters = cv::Mat(lab_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer

    /* Go through all the pixels. */
    for (int c = 0; c < lab_image.cols; c++) 
    {
        for (int r = 0; r < lab_image.rows; r++) 
        {
            if (new_clusters.at<int>(r, c) == -1) // if pixel is not assigned
            {
                std::vector<cv::Point> elements;
                elements.push_back(cv::Point(c, r));
                
                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) 
                {
                    int nc = c + dx4[k], nr = r + dy4[k];
                    
                    if (nc >= 0 && nc < lab_image.cols && nr >= 0 && nr < lab_image.rows) 
                    {
                        if (new_clusters.at<int>(nr, nc) >= 0) // if neighboring pixel is assigned
                        {
                            adjlabel = new_clusters.at<int>(nr, nc); 
                        }
                    }
                }

                int count = 1;
                for (int i = 0; i < elements.size(); i++) 
                {
                    for (int k = 0; k < 4; k++) 
                    {
                        int nc = elements[i].x + dx4[k], nr = elements[i].y + dy4[k];
                        
                        if (nc >= 0 && nc < lab_image.cols && nr >= 0 && nr < lab_image.rows) 
                        {
                            if (new_clusters.at<int>(nr, nc) == -1 && clusters_.at<int>(r, c) == clusters_.at<int>(nr, nc)) 
                            {
                                elements.push_back(cv::Point(nc, nr));
                                new_clusters.at<int>(nr, nc) = label;
                                count += 1;
                            }
                        }
                    }
                }

                /* Use the earlier found adjacent label if a segment size is less than a limit. */
                if (count <= lims >> 2) 
                {
                    for (int i = 0; i < elements.size(); i++) 
                    {
                        new_clusters.at<int>(elements[i].y, elements[i].x) = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }

        }
    }    

    return;
}

void SuperpixelColorSegmenter::displayCenterGrid(cv::Mat & image, const cv::Vec3b & color)
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::displayCenterGrid]");
    
    // Display center grid
    for (int i = 0; i < (int) centers_.size(); i++) 
    {
        cv::circle(image, cv::Point(centers_[i][3], centers_[i][4]), 2, color, -1);
    }

    return;
}

void SuperpixelColorSegmenter::displayContours(cv::Mat & contours)
{
    // Display contours
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1}; // eight neighboring pixels
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1}; // eight neighboring pixels

    /* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
    cv::Mat isTaken = cv::Mat(lab_image.size(), CV_8U, cv::Scalar(0));

    /* Go through all the pixels. */
    for (int c = 0; c < lab_image.cols; c++) 
    {
        for (int r = 0; r < lab_image.rows; r++) 
        {
            int nr_p = 0;

            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) 
            {
                int c_neighbor = c + dx8[k], r_neighbor = r + dy8[k];
                
                if (c_neighbor >= 0 && c_neighbor < lab_image.cols && r_neighbor >= 0 && r_neighbor < lab_image.rows) 
                {
                    if (isTaken.at<uint8_t>(r_neighbor, c_neighbor) == 0 && 
                        clusters_.at<int>(r, c) != clusters_.at<int>(r_neighbor, c_neighbor))
                    {
                        nr_p += 1;
                    }
                }
            }

            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) 
            {
                contours.at<uint8_t>(r, c) = 1;
                isTaken.at<uint8_t>(r, c) = 1;
            }            

        }
    }

    return;
}

void SuperpixelColorSegmenter::displaySuperpixelsWithClusterMeans(cv::Mat & overlaid_image)
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::displaySuperpixelsWithClusterMeans]");

    std::vector<cv::Scalar> colors(centers_.size());
    
    // ROS_INFO_STREAM("       Gathering ...");

    // ROS_INFO_STREAM("           colors.size(): " << colors.size());


    /* Gather the colour values per cluster. */
    for (int c = 0; c < lab_image.cols; c++) 
    {
        for (int r = 0; r < lab_image.rows; r++) 
        {
            int index = clusters_.at<int>(r, c);

            if (index == -1 || index >= colors.size())
            {
                // ROS_ERROR_STREAM("       index: " << index);
                continue;
            }

            cv::Vec3b color = lab_image.at<cv::Vec3b>(r, c);

            colors[index].val[0] += color.val[0];
            colors[index].val[1] += color.val[1];
            colors[index].val[2] += color.val[2];
        }
    }
    
    // ROS_INFO_STREAM("       Averaging ...");

    // ROS_INFO_STREAM("           colors.size(): " << colors.size());

    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < colors.size(); i++) 
    {
        // ROS_INFO_STREAM("       center_counts_[i]: " << center_counts_[i]);
        colors[i].val[0] /= center_counts_[i];
        colors[i].val[1] /= center_counts_[i];
        colors[i].val[2] /= center_counts_[i];
    }
    
    // ROS_INFO_STREAM("       Coloring ...");

    /* Fill in. */
    for (int c = 0; c < lab_image.cols; c++) 
    {
        for (int r = 0; r < lab_image.rows; r++) 
        {
            int index = clusters_.at<int>(r, c);
            cv::Scalar ncolor = colors[index];
            overlaid_image.at<cv::Vec3b>(r, c) = cv::Vec3b(ncolor.val[0], ncolor.val[1], ncolor.val[2]);
        }
    }
}

double SuperpixelColorSegmenter::computeDistance(const int & center_idx, const cv::Vec3b & color, const cv::Point & pixel)
{
    // Color term
    double dc = sqrt(pow(color.val[0] - centers_[center_idx][0], 2) +
                     pow(color.val[1] - centers_[center_idx][1], 2) +
                     pow(color.val[2] - centers_[center_idx][2], 2));

    // Spatial term
    double ds = sqrt(pow(pixel.x - centers_[center_idx][3], 2) +
                     pow(pixel.y - centers_[center_idx][4], 2));
    
    return sqrt(pow(dc / params_.n_c_, 2) + pow(ds / params_.n_s_, 2));
}

cv::Point SuperpixelColorSegmenter::findLocalMinimum(const cv::Mat & image, const cv::Point & center)
{
    double min_grad = std::numeric_limits<double>::max();
    cv::Point loc_min = center;

    for (int i = center.x - 1; i <= center.x + 1; i++)
    {
        for (int j = center.y - 1; j <= center.y + 1; j++)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(j, i);
            double grad = sqrt(pow(color.val[0] - image.at<cv::Vec3b>(center.y, center.x).val[0], 2) +
                               pow(color.val[1] - image.at<cv::Vec3b>(center.y, center.x).val[1], 2) +
                               pow(color.val[2] - image.at<cv::Vec3b>(center.y, center.x).val[2], 2));

            if (grad < min_grad)
            {
                min_grad = grad;
                loc_min = cv::Point(i, j);
            }
        }
    }

    return loc_min;
}

void SuperpixelColorSegmenter::reset_data()
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::reset_data]");
    
    if (params_.warm_start_)
    {
        clusters_ = cv::Mat(lab_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer

        distances_ = cv::Mat(lab_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point
        
        // Keep centers as is

        center_counts_.assign(center_counts_.size(), 0);
    } else
    {
        clusters_.release();
        distances_.release();
        centers_.clear();
        center_counts_.clear();

        init_data();
    }

    return;
}

void SuperpixelColorSegmenter::init_data()
{
    // ROS_INFO_STREAM("   [SuperpixelColorSegmenter::init_data]");

    /* Initialize the cluster and distance matrices. */
    clusters_ = cv::Mat(lab_image.size(), CV_32S, cv::Scalar(-1)); // 32-bit signed integer
    distances_ = cv::Mat(lab_image.size(), CV_64F, cv::Scalar(std::numeric_limits<double>::max())); // 64-bit floating-point

    // ROS_INFO_STREAM("       image.cols: " << image.cols);
    // ROS_INFO_STREAM("       image.rows: " << image.rows);
    // ROS_INFO_STREAM("       params_.step_: " << params_.step_);

    /* Initialize the centers and counters. */
    centers_.clear();
    center_counts_.clear();
    for (int i = params_.step_; i < lab_image.cols - (params_.step_ / 2); i += params_.step_)
    {
        for (int j = params_.step_; j < lab_image.rows - (params_.step_ / 2); j += params_.step_)
        {

            // ROS_INFO_STREAM("       (i, j): (" << i << ", " << j << ")");


            std::vector<double> center;

            /* Find the local minimum (gradient-wise). */
            cv::Point localMinimum = findLocalMinimum(lab_image, cv::Point(i, j));
            cv::Vec3b color = lab_image.at<cv::Vec3b>(localMinimum.y, localMinimum.x);

            /* Generate the center vector. */
            center.push_back(color.val[0]);
            center.push_back(color.val[1]);
            center.push_back(color.val[2]);
            center.push_back(localMinimum.x);
            center.push_back(localMinimum.y);

            /* Append to vector of centers. */
            centers_.push_back(center);
            center_counts_.push_back(0);
        }
    }

    // ROS_INFO_STREAM("       centers_.size(): " << centers_.size());
    // ROS_INFO_STREAM("       center_counts_.size(): " << center_counts_.size());


    return;
}