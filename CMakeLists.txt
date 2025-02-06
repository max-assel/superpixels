cmake_minimum_required(VERSION 3.10)
project(superpixels)

set(CATKIN_PACKAGE_DEPENDENCIES
    roscpp
    roslib
    cv_bridge
    image_transport
    dynamic_reconfigure
    pcl_ros
    )

find_package(catkin REQUIRED COMPONENTS
          ${CATKIN_PACKAGE_DEPENDENCIES}
        )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL    REQUIRED)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

generate_dynamic_reconfigure_options(
  cfg/Parameters.cfg
)
###################################
## catkin specific configuration ##
###################################

catkin_package(
   INCLUDE_DIRS 
        include
        ${EIGEN3_INCLUDE_DIRS}
   LIBRARIES
        ${PROJECT_NAME}
        yaml-cpp
   CATKIN_DEPENDS
        ${CATKIN_PACKAGE_DEPENDENCIES}
)

###########
## Build ##
###########

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)

# Main library
add_library(${PROJECT_NAME}
    src/SuperpixelColorOpenCVSegmenter.cpp
    src/SuperpixelColorSegmenter.cpp
    src/SuperpixelDepthSegmenter.cpp
)

add_dependencies(${PROJECT_NAME} 
              ${${PROJECT_NAME}_EXPORTED_TARGETS} 
              ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        yaml-cpp
        )

target_compile_options(${PROJECT_NAME} PUBLIC ${FLAGS})

## Declare a C++ executable

# Superpixel algorithm node for depth images (custom implementation)

add_executable(superpixel_depth_segmentation_node
        src/SuperpixelDepthSegmentationNode.cpp
        )
add_dependencies(superpixel_depth_segmentation_node
        ${PROJECT_NAME}
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        ${PROJECT_NAME}_gencfg
        )
target_link_libraries(superpixel_depth_segmentation_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )

# Superpixel algorithm node for color images (custom implementation)

add_executable(superpixel_color_segmentation_node
        src/SuperpixelColorSegmentationNode.cpp
        )
add_dependencies(superpixel_color_segmentation_node
        ${PROJECT_NAME}
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        ${PROJECT_NAME}_gencfg
        )
target_link_libraries(superpixel_color_segmentation_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )

# Superpixel algorithm node for color images (OTS implementation)

add_executable(superpixel_color_opencv_segmentation_node
        src/SuperpixelColorOpenCVSegmentationNode.cpp
        )
add_dependencies(superpixel_color_opencv_segmentation_node
        ${PROJECT_NAME}
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        ${PROJECT_NAME}_gencfg
        )
target_link_libraries(superpixel_color_opencv_segmentation_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )
