import os

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration



def generate_launch_description():

    ####################
    # Launch Arguments #
    ####################
    rviz = True
    terrain_receiver = False
    depth_image_topic = "/floor_image"
    # label_image_topic = "/floor_labels"
    normal_image_topic = "/floor_normals"

    #######################
    # Package Directories #
    #######################

    superpixels_path = get_package_share_directory("superpixels")
    config_path = os.path.join(superpixels_path, "cfg", "depth.yaml")

    ############################
    # Declare Launch Arguments #
    ############################
    
    # Floor image parameters
    declare_k_c = DeclareLaunchArgument("k_c", default_value="512", description="k_c parameter for superpixel segmentation.")
    declare_v_fov = DeclareLaunchArgument("v_fov", default_value="60.0", description="Vertical field of view in degrees.")
    declare_v_offset = DeclareLaunchArgument("v_offset", default_value="0.0", description="Vertical offset in degrees.")
    
    # Dilation parameters
    declare_num_dilation_iterations = DeclareLaunchArgument("num_dilation_iterations", default_value="0", description="Number of dilation iterations for the superpixel segmentation.")
    declare_kernel_radius = DeclareLaunchArgument("kernel_radius", default_value="1", description="Radius of the kernel used for dilation in superpixel segmentation.")
    
    # Superpixel algorithm parameters
    declare_num_iterations = DeclareLaunchArgument("num_iterations", default_value="10", description="Number of iterations for the superpixel segmentation algorithm.")
    declare_num_superpixels = DeclareLaunchArgument("num_superpixels", default_value="200", description="Number of superpixels to generate in the segmentation.")
    declare_warm_start = DeclareLaunchArgument("warm_start", default_value="False", description="Whether to use warm start for the superpixel segmentation.")
    declare_constraint = DeclareLaunchArgument("constraint", default_value="True", description="Whether to apply constraints during superpixel segmentation.")
    declare_ransac = DeclareLaunchArgument("ransac", default_value="True", description="Whether to use RANSAC for superpixel segmentation.")
    declare_snapping = DeclareLaunchArgument("snapping", default_value="False", description="Whether to apply snapping during superpixel segmentation.")

    # Superpixel distance parameters
    declare_w_normal = DeclareLaunchArgument("w_normal", default_value="1.0", description="Weight for the normal distance in superpixel segmentation.")
    declare_w_pos = DeclareLaunchArgument("w_pos", default_value="1.0", description="Weight for the position distance in superpixel segmentation.")
    declare_w_compact = DeclareLaunchArgument("w_compact", default_value="3.0", description="Weight for the compactness in superpixel segmentation.")
    
    # RANSAC parameters
    declare_ransac_K = DeclareLaunchArgument("ransac_K", default_value="10", description="Number of RANSAC iterations for superpixel segmentation.")
    declare_ransac_N = DeclareLaunchArgument("ransac_N", default_value="25", description="Number of samples for RANSAC in superpixel segmentation.")
    declare_ransac_T = DeclareLaunchArgument("ransac_T", default_value="0.01", description="Threshold for RANSAC in superpixel segmentation.")

    #################
    # Include Nodes #
    #################
    superpixels_node = Node(
        package="superpixels",
        executable="superpixel_depth_segmentation_node",
        name="superpixel_depth_segmentation_node",
        output="screen",
        parameters=[
            {
                'k_c': LaunchConfiguration('k_c'),
                'v_fov': LaunchConfiguration('v_fov'),
                'v_offset': LaunchConfiguration('v_offset'),
                'num_dilation_iterations': LaunchConfiguration('num_dilation_iterations'),
                'kernel_radius': LaunchConfiguration('kernel_radius'),
                'num_iterations': LaunchConfiguration('num_iterations'),
                'num_superpixels': LaunchConfiguration('num_superpixels'),
                'warm_start': LaunchConfiguration('warm_start'),
                'constraint': LaunchConfiguration('constraint'),
                'ransac': LaunchConfiguration('ransac'),
                'snapping': LaunchConfiguration('snapping'),
                'w_normal': LaunchConfiguration('w_normal'),
                'w_pos': LaunchConfiguration('w_pos'),
                'w_compact': LaunchConfiguration('w_compact'),
                'ransac_K': LaunchConfiguration('ransac_K'),
                'ransac_N': LaunchConfiguration('ransac_N'),
                'ransac_T': LaunchConfiguration('ransac_T'),
                'config_file': config_path,
                'use_sim_time': True,
                'depth_image_topic': depth_image_topic,
                'normal_image_topic': normal_image_topic,
            }
        ]
    )

    rqt_node = Node(
        package="rqt_reconfigure",
        executable="rqt_reconfigure",
        name="rqt_reconfigure",
        output="screen"
    )

    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     condition=IfCondition(rviz),
    #     arguments=["-d", os.path.join(
    #                 superpixels_path, "rviz", "superpixels_depth.rviz",
    #             )
    #     ]
    # )

    ###########################
    # Full Launch Description #
    ###########################
    return LaunchDescription(
        [
            declare_k_c,
            declare_v_fov,
            declare_v_offset,
            declare_num_dilation_iterations,
            declare_kernel_radius,
            declare_num_iterations,
            declare_num_superpixels,
            declare_warm_start,
            declare_constraint,
            declare_ransac,
            declare_snapping,
            declare_w_normal,
            declare_w_pos,
            declare_w_compact,
            declare_ransac_K,
            declare_ransac_N,
            declare_ransac_T,
            superpixels_node,
            # rviz_node,
            rqt_node
        ]
    )
