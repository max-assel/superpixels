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

    realsense2_camera_path = get_package_share_directory("realsense2_camera")
    superpixels_path = get_package_share_directory("superpixels")

    config_path = os.path.join(superpixels_path, "cfg", "depth_online_hardware.yaml")

    ############################
    # Declare Launch Arguments #
    ############################
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )    

    set_use_sim_time = launch_ros.actions.SetParameter(name='use_sim_time', value=False)

    #################
    # Include Nodes #
    #################

    realsense_ld = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    realsense2_camera_path, "launch", "rs_d435_launch.py", # rs_d400_and_t265_launch
                )
            ),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }.items() 
        )
    
    depth_img_normal_estimation_node = Node(
            package='depth_img_normal_estimation',
            executable='depth_img_normal_estimation_node',
            name='depth_img_normal_estimation_node',
            output='screen',
            parameters=[
                {
                    'use_sim_time': LaunchConfiguration("use_sim_time")
                },
                {
                    'camera_depth_topic': '/D435/depth/image_rect_raw'
                },
                {
                    'camera_normals_topic': '/D435/normals'
                },
                {
                    'config_path': get_package_share_directory('depth_img_normal_estimation') + '/cfg/hardware.yaml'
                },
                {
                    'hardware': True
                }
            ]
    )

    # odom_to_base_aligned_tf2_cmd = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="odom_to_base_aligned_tf2",
    #     output="screen",
    #     arguments=[
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "odom",
    #         "base_aligned"
    #     ],
    #     parameters=[
    #         {
    #             "use_sim_time": LaunchConfiguration("use_sim_time"),
    #         }
    #     ]
    # )

    # odom_to_D435_depth_optical_frame_tf2_cmd = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="odom_to_D435_depth_optical_frame_tf2",
    #     output="screen",
    #     arguments=[
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "0",
    #         "odom",
    #         "D435_depth_optical_frame"
    #     ],
    #     parameters=[
    #         {
    #             "use_sim_time": LaunchConfiguration("use_sim_time"),
    #         }
    #     ]
    # )

    egocylindrical_propagator_config = os.path.join(
        get_package_share_directory('egocylindrical'),
        'cfg',
        'egocylindrical_propagator.yaml')

    
    egocylindrical_propagator_node = Node(
        package="egocylindrical",
        executable="egocylindrical_propagator_node",
        name="egocylindrical_propagator_node",
        output="screen",
        parameters=[egocylindrical_propagator_config]
    )
    
    # floor_image_node = Node(
    #     package="egocylindrical",
    #     executable="floor_image_node",
    #     name="floor_image_node",
    #     output="screen",
    #     remappings=[
    #         ('egocylindrical_points', 'data')
    #     ],
    #     parameters=[
    #         {
    #         'use_sim_time': False,  # Use simulation time if available
    #         'use_raw': False,
    #         'floor_image_topic': 'floor_image',
    #         'floor_labels_topic': 'floor_labels',
    #         'floor_labels_colored_topic': 'floor_labels_colored',
    #         'floor_normals_topic': 'floor_normals',
    #         'floor_labels_colored_topic': 'floor_labels_colored'
    #         }
    #     ]
    # )

    # superpixels_node = Node(
    #     package="superpixels",
    #     executable="superpixel_depth_segmentation_node",
    #     name="superpixel_depth_segmentation_node",
    #     output="screen",
    #     parameters=[config_path]
    # )

    ###########################
    # Full Launch Description #
    ###########################
    return LaunchDescription(
        [
            set_use_sim_time,
            declare_use_sim_time,
            realsense_ld,
            depth_img_normal_estimation_node,
            # odom_to_base_aligned_tf2_cmd,
            # odom_to_D435_depth_optical_frame_tf2_cmd,
            egocylindrical_propagator_node,
            # floor_image_node,
            # superpixels_node
        ]
    )
