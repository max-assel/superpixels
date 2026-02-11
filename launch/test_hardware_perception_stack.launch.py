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
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )    

    set_use_sim_time = launch_ros.actions.SetParameter(name='use_sim_time', value=False)

    #######################
    # Package Directories #
    #######################

    egocylindrical_path = get_package_share_directory("egocylindrical")
    depth_img_normal_estimation_path = get_package_share_directory("depth_img_normal_estimation")
    superpixels_path = get_package_share_directory("superpixels")
    realsense2_camera_path = get_package_share_directory("realsense2_camera")
    config_path = os.path.join(superpixels_path, "cfg", "depth_hardware.yaml")

    go2_description_path = get_package_share_directory("go2_description")
    go2_xacro_file_path = os.path.join(go2_description_path, "xacro", "robot_payload.xacro")

    go2_interface_path = get_package_share_directory("go2_interface")

    # Convert xacro to urdf and publish on /robot_description topic
    robot_description_command = Command(["xacro ", go2_xacro_file_path])

    ############################
    # Declare Launch Arguments #
    ############################
    
    #################
    # Include Nodes #
    #################
    realsense_ld = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    realsense2_camera_path, "launch", "rs_d435_launch.py",
                )
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
            }.items()
        )

    normal_estimation_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                depth_img_normal_estimation_path, "launch", "normal_estimation_hardware.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    semantic_egocan_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                egocylindrical_path, "launch", "semantic_egocan.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )  

    superpixels_node = Node(
        package="superpixels",
        executable="superpixel_depth_segmentation_node",
        name="superpixel_depth_segmentation_node",
        output="screen",
        parameters=[config_path]
    )

    rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=[
                "-d",
                os.path.join(
                    go2_interface_path, "rviz", "go2_foxy.rviz",
                )
            ],
            parameters=[
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }
            ]
        )

    robot_state_publisher_node = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description_command},
            {"publish_frequency": 200.0},
            {"ignore_timestamp": True},
            {'use_sim_time': LaunchConfiguration("use_sim_time")},
        ] # ,
    )

    odom_to_base_tf = launch_ros.actions.Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_to_base_static_transform_node",
        output="screen",
        arguments=[
            "0",
            "0",
            "0.30",
            "0",
            "0",
            "0",
            "odom",
            "base"
        ],
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ]
    )    

    ###########################
    # Full Launch Description #
    ###########################
    return LaunchDescription(
        [
            set_use_sim_time,
            declare_use_sim_time,
            realsense_ld,
            normal_estimation_ld,
            semantic_egocan_ld,
            rviz_node,
            superpixels_node,
            robot_state_publisher_node,
            odom_to_base_tf
        ]
    )
