#include <superpixels/ConvexHullifier.h>

ConvexHullifier::ConvexHullifier(const SuperpixelParams & params)
{
    params_ = params;

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);
}

void ConvexHullifier::setParams(const SuperpixelParams & params)
{
    params_ = params;
}

Eigen::Vector3d ConvexHullifier::projectPointOntoPlane(const Eigen::Vector3d & regionPt)
{
    Eigen::Vector3d planeCenter = Eigen::Vector3d(0, 0, 0); // center is the origin
    Eigen::Vector3d planeNormal = Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d diff = regionPt - planeCenter; 
    float dist = diff.dot(planeNormal);
    Eigen::Vector3d projPt = regionPt - dist * planeNormal;

    return projPt;
}

void ConvexHullifier::run(const std::vector<std::vector<double>> & centers,
                            const std::vector<std::vector<cv::Point>> & superpixels,
                            std::vector<std::vector<Eigen::Vector2d>> & superpixel_convex_hulls,
                            std::vector<Eigen::Matrix3d> & superpixel_rotations,
                            const cv::Mat & depth_img,
                            const cv_bridge::CvImagePtr & raw_depth_img_ptr)
{
    // ROS_INFO_STREAM("   [ConvexHullifier::run]");

    ros::Time lookupTime = raw_depth_img_ptr->header.stamp;
    std::string egocan_frame = raw_depth_img_ptr->header.frame_id;

    geometry_msgs::TransformStamped egocanFrameToWorldFrame = 
        tfBuffer_.lookupTransform("world", egocan_frame, lookupTime);

    std::vector<std::vector<Eigen::Vector2d>> superpixel_projections(centers.size());
    superpixel_convex_hulls = std::vector<std::vector<Eigen::Vector2d>>(centers.size());
    superpixel_rotations = std::vector<Eigen::Matrix3d>(centers.size());

    // Build convex hulls for each superpixel
    for (int i = 0; i < (int) centers.size(); i++)
    {
        // ROS_INFO_STREAM("       center " << i << ":");
        //////////////////////////////////////////////////////////////
        // 1. Build transfrom from egocan frame to superpixel frame //
        //////////////////////////////////////////////////////////////
        cv::Point center_pixel = cv::Point(centers[i][0], centers[i][1]);
        cv::Vec3f egocanPt;
        pixelToEgocanFrame(egocanPt, center_pixel, depth_img.at<float>(center_pixel.y, center_pixel.x), params_.k_c_, params_.h_);

        Eigen::Vector3d center(egocanPt.val[0], egocanPt.val[1], egocanPt.val[2]);
        Eigen::Vector3d normal(centers[i][4], centers[i][5], centers[i][6]);

        // ROS_INFO_STREAM("       center (egocan frame): " << center.transpose());
        // ROS_INFO_STREAM("       normal (egocan frame): " << normal.transpose());

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

        // ROS_INFO_STREAM("       e0: " << e0.transpose());
        // ROS_INFO_STREAM("       e1: " << e1.transpose());

        Eigen::Matrix3d regionRotMat;
        regionRotMat.row(0) = e0;
        regionRotMat.row(1) = e1;
        regionRotMat.row(2) = normal;

        superpixel_rotations[i] = regionRotMat;

        // ROS_INFO_STREAM("       regionRotMat: ");
        // ROS_INFO_STREAM("           " << regionRotMat.row(0));
        // ROS_INFO_STREAM("           " << regionRotMat.row(1));
        // ROS_INFO_STREAM("           " << regionRotMat.row(2));

        // Eigen::Quaterniond regionQuat(regionRotMat);

        // ROS_INFO_STREAM("               center: " << center.transpose());
        // ROS_INFO_STREAM("               regionQuat: " << regionQuat.x() << ", " << regionQuat.y() << ", " << regionQuat.z() << ", " << regionQuat.w());

        // Eigen::VectorXd region_pose_world_frame = transformHelperPoseStamped(center, regionQuat, egocanFrameToWorldFrame);

        // // get rotation matrix
        // Eigen::Matrix3d rotMat = calculateRotationMatrix(region_pose_world_frame[3], region_pose_world_frame[4], region_pose_world_frame[5]);

        // Eigen::Matrix4d worldToRegionTransform = Eigen::Matrix4d::Identity();
        // worldToRegionTransform.block<3, 3>(0, 0) = rotMat;
        // worldToRegionTransform.block<3, 1>(0, 3) = region_pose_world_frame.head(3); 

        // Eigen::Matrix4d egocanToWorldTransform = Eigen::Matrix4d::Identity();
        // // quaternion to rotation matrix
        // Eigen::Quaterniond egocanToWorldQuat(egocanFrameToWorldFrame.transform.rotation.w, 
        //                                         egocanFrameToWorldFrame.transform.rotation.x, 
        //                                         egocanFrameToWorldFrame.transform.rotation.y, 
        //                                         egocanFrameToWorldFrame.transform.rotation.z);
        // Eigen::Matrix3d egocanToWorldRotMat = egocanToWorldQuat.toRotationMatrix();
        // egocanToWorldTransform.block<3, 3>(0, 0) = egocanToWorldRotMat;
        // Eigen::Vector3d egocanToWorldTrans(egocanFrameToWorldFrame.transform.translation.x,
        //                                     egocanFrameToWorldFrame.transform.translation.y,
        //                                     egocanFrameToWorldFrame.transform.translation.z);
        // egocanToWorldTransform.block<3, 1>(0, 3) = egocanToWorldTrans;

        // Eigen::Matrix4d egocanToRegionTransform = egocanToWorldTransform * worldToRegionTransform;

        Eigen::Matrix4d egocanToRegionTransform;
        egocanToRegionTransform << regionRotMat, -regionRotMat * center,
                                    0, 0, 0, 1;

        // ROS_INFO_STREAM("       egocanToRegionTransform: ");
        // ROS_INFO_STREAM("           " << egocanToRegionTransform.row(0));
        // ROS_INFO_STREAM("           " << egocanToRegionTransform.row(1));
        // ROS_INFO_STREAM("           " << egocanToRegionTransform.row(2));
        // ROS_INFO_STREAM("           " << egocanToRegionTransform.row(3));

        ///////////////////////////////////////
        // 2. Transform points to superpixel //
        ///////////////////////////////////////

        superpixel_projections[i].resize(superpixels[i].size());
        for (int j = 0; j < superpixels[i].size(); j++)
        {
            // ROS_INFO_STREAM("           superpixel point " << j << ":");

            // Transform superpixel points into region frame
            cv::Point pixel = superpixels[i][j];
            float depth = depth_img.at<float>(pixel.y, pixel.x);

            cv::Vec3f egocanPt;
            pixelToEgocanFrame(egocanPt, pixel, depth, params_.k_c_, params_.h_);

            // ROS_INFO_STREAM("               (egocan frame): " << egocanPt.val[0] << ", " << egocanPt.val[1] << ", " << egocanPt.val[2]);

            Eigen::Vector4d egocanPtHomog(egocanPt.val[0], egocanPt.val[1], egocanPt.val[2], 1.0);
            Eigen::Vector4d regionPtHomog = egocanToRegionTransform * egocanPtHomog;

            Eigen::Vector3d regionPt = regionPtHomog.head(3);

            // ROS_INFO_STREAM("               (region frame): " << regionPt.transpose());

            // project points onto plane
            Eigen::Vector3d projPt = projectPointOntoPlane(regionPt);

            // ROS_INFO_STREAM("               projected (region frame): " << projPt.transpose());

            // check projpt, should be coplanar with center and normal

            superpixel_projections[i][j] = projPt.head(2);
        }

        // Calculate convex hull and store
        convexHull(superpixel_projections[i], superpixel_convex_hulls[i]);
        // ROS_INFO_STREAM("       hull:");
        // for (int j = 0; j < (int) superpixel_convex_hulls[i].size(); j++)
        // {
        //     ROS_INFO_STREAM("           point " << j << ": " << superpixel_convex_hulls[i][j][0] << ", " << superpixel_convex_hulls[i][j][1]);
        // }
    }

    return;
}

void ConvexHullifier::convexHull(const std::vector<Eigen::Vector2d> & superpixel_projections, 
                                    std::vector<Eigen::Vector2d> & convex_hull)
{
    grahamScan(superpixel_projections, convex_hull);
}

bool ConvexHullifier::polarSort(const Eigen::Vector2d & a, const Eigen::Vector2d & b, const Eigen::Vector2d & lowest)
{
    if (a == lowest)
    {
        return true;
    }
    if (b == lowest)
    {
        return false;
    }

    double angle_a = std::atan2(a[1] - lowest[1], a[0] - lowest[0]);
    double angle_b = std::atan2(b[1] - lowest[1], b[0] - lowest[0]);

    if (angle_a < angle_b)
    {
        return true;
    }
    else if (angle_a == angle_b)
    {
        return (a - lowest).norm() < (b - lowest).norm();
    }
    else
    {
        return false;
    }
}

bool ConvexHullifier::ccw(const Eigen::Vector2d & a, const Eigen::Vector2d & b, const Eigen::Vector2d & c)
{
    Eigen::Vector2d a_to_b = b - a;
    Eigen::Vector2d a_to_c = c - a;
    return (a_to_b[0] * a_to_c[1] - a_to_b[1] * a_to_c[0]) > 0;
}

void ConvexHullifier::grahamScan(const std::vector<Eigen::Vector2d> & superpixel_projections, 
                                    std::vector<Eigen::Vector2d> & convex_hull)
{
    // ROS_INFO_STREAM("           [ConvexHullifier::grahamScan]");

    if (superpixel_projections.size() < 3)
    {
        // ROS_INFO_STREAM("               not enough points for convex hull.");
        convex_hull = superpixel_projections;
        return;
    }

    // ROS_INFO_STREAM("               finding lowest y-coordinate ... ");
    // Find the point with the lowest y-coordinate
    Eigen::Vector2d lowest = superpixel_projections[0];
    // ROS_INFO_STREAM("                   initial lowest: " << lowest[0] << ", " << lowest[1]);
    for (int i = 1; i < (int) superpixel_projections.size(); i++)
    {
        Eigen::Vector2d current = superpixel_projections[i];
        // ROS_INFO_STREAM("                   point " << i << ": " << current[0] << ", " << current[1]);
        if (current[1] < lowest[1])
        {
            // ROS_INFO_STREAM("                   new lowest: " << current[0] << ", " << current[1]);
            lowest = current;
        }
        else if (current[1] == lowest[1])
        {
            if (current[0] < lowest[0])
            {
                // ROS_INFO_STREAM("                   new lowest: " << current[0] << ", " << current[1]);
                lowest = current;
            }
        }
    }

    // ROS_INFO_STREAM("           lowest: " << lowest[0] << ", " << lowest[1]);

    // ROS_INFO_STREAM("           sorting ... ");

    // Sort the points by polar angle with respect to the lowest point
    std::vector<Eigen::Vector2d> sorted = superpixel_projections;
    std::sort(sorted.begin(), sorted.end(), std::bind(&ConvexHullifier::polarSort, 
                                                        this, 
                                                        std::placeholders::_1, 
                                                        std::placeholders::_2, 
                                                        lowest)); 

    // ROS_INFO_STREAM("           sorted.");

    // Perform the Graham scan
    std::vector<Eigen::Vector2d> hull;
    hull.push_back(sorted[0]);
    // ROS_INFO_STREAM("           added point 0: " << sorted[0][0] << ", " << sorted[0][1]);
    hull.push_back(sorted[1]);
    // ROS_INFO_STREAM("           added point 1: " << sorted[1][0] << ", " << sorted[1][1]);

    for (int i = 2; i < (int) sorted.size(); i++)
    {
        while (hull.size() >= 2 && !ccw(hull[hull.size() - 2], hull[hull.size() - 1], sorted[i]))
        {
            hull.pop_back();
        }
        hull.push_back(sorted[i]);
    }

    convex_hull = hull;
}