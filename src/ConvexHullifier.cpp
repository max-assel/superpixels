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
                            const cv::Mat & depth_img,
                            const cv_bridge::CvImagePtr & raw_depth_img_ptr)
{
    ROS_INFO_STREAM("   [ConvexHullifier::run]");

    ros::Time lookupTime = raw_depth_img_ptr->header.stamp;
    std::string egocan_frame = raw_depth_img_ptr->header.frame_id;

    geometry_msgs::TransformStamped egocanFrameToWorldFrame = 
        tfBuffer_.lookupTransform("world", egocan_frame, lookupTime);

    std::vector<std::vector<cv::Vec3f>> superpixel_projections(centers.size());
    // std::vector<std::vector<cv::Point>> superpixel_convex_hulls(centers.size());

    // Build convex hulls for each superpixel
    for (int i = 0; i < (int) centers.size(); i++)
    {
        ROS_INFO_STREAM("       center " << i << ":");
        //////////////////////////////////////////////////////////////
        // 1. Build transfrom from egocan frame to superpixel frame //
        //////////////////////////////////////////////////////////////
        cv::Point center_pixel = cv::Point(centers[i][0], centers[i][1]);
        cv::Vec3f egocanPt;
        pixelToEgocanFrame(egocanPt, center_pixel, depth_img.at<float>(center_pixel.y, center_pixel.x), params_.k_c_, params_.h_);

        Eigen::Vector3d center(egocanPt.val[0], egocanPt.val[1], egocanPt.val[2]);
        Eigen::Vector3d normal(centers[i][4], centers[i][5], centers[i][6]);

        ROS_INFO_STREAM("       center (egocan frame): " << center.transpose());
        ROS_INFO_STREAM("       normal (egocan frame): " << normal.transpose());

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

        ROS_INFO_STREAM("       e0: " << e0.transpose());
        ROS_INFO_STREAM("       e1: " << e1.transpose());

        Eigen::Matrix3d regionRotMat;
        regionRotMat.row(0) = e0;
        regionRotMat.row(1) = e1;
        regionRotMat.row(2) = normal;

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

        ROS_INFO_STREAM("       egocanToRegionTransform: ");
        ROS_INFO_STREAM("           " << egocanToRegionTransform.row(0));
        ROS_INFO_STREAM("           " << egocanToRegionTransform.row(1));
        ROS_INFO_STREAM("           " << egocanToRegionTransform.row(2));
        ROS_INFO_STREAM("           " << egocanToRegionTransform.row(3));

        ///////////////////////////////////////
        // 2. Transform points to superpixel //
        ///////////////////////////////////////

        superpixel_projections[i].resize(superpixels[i].size());
        for (int j = 0; j < superpixels[i].size(); j++)
        {
            ROS_INFO_STREAM("           superpixel point " << j << ":");

            // Transform superpixel points into region frame
            cv::Point pixel = superpixels[i][j];
            float depth = depth_img.at<float>(pixel.y, pixel.x);

            cv::Vec3f egocanPt;
            pixelToEgocanFrame(egocanPt, pixel, depth, params_.k_c_, params_.h_);

            ROS_INFO_STREAM("               (egocan frame): " << egocanPt.val[0] << ", " << egocanPt.val[1] << ", " << egocanPt.val[2]);

            Eigen::Vector4d egocanPtHomog(egocanPt.val[0], egocanPt.val[1], egocanPt.val[2], 1.0);
            Eigen::Vector4d regionPtHomog = egocanToRegionTransform * egocanPtHomog;

            Eigen::Vector3d regionPt = regionPtHomog.head(3);

            ROS_INFO_STREAM("               (region frame): " << regionPt.transpose());

            // project points onto plane
            Eigen::Vector3d projPt = projectPointOntoPlane(regionPt);

            ROS_INFO_STREAM("               projected (region frame): " << projPt.transpose());

            // check projpt, should be coplanar with center and normal

            // superpixel_projections[i][j] = cv::Vec3f(regionPtHomog[0], regionPtHomog[1], regionPtHomog[2]);
        }

        // project points onto plane
        // superpixel_projections[i].resize(superpixels[i].size());
        // for (int j = 0; j < superpixels[i].size(); j++)
        // {
        //     cv::Point pixel = superpixels[i][j];
        //     float depth = depth_image.at<float>(pixel.y, pixel.x);

        //     cv::Vec3f egocanPt;
        //     pixelToEgocanFrame(egocanPt, pixel, depth, params_.k_c_, params_.h_);

        //     cv::Vec3f centerEgocanPt;
        //     pixelToEgocanFrame(centerEgocanPt, center, centers[i][2], params_.k_c_, params_.h_);

        //     cv::Vec3f projPt = projectPointOntoPlane(egocanPt, centerEgocanPt, normal);

        //     superpixel_projections[i][j] = projPt;
        // }
        
        // Get 2d plane points

        // Calculate convex hull and store
    }

    return;
}