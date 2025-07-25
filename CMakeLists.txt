cmake_minimum_required(VERSION 3.15)
project(superpixels)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(dependencies
    rclcpp
    # roslib
    cv_bridge
    image_transport
    pcl_ros
    convex_plane_decomposition
    convex_plane_decomposition_ros
    convex_plane_decomposition_msgs
    segmented_planes_terrain_model
    )


# Dependencies
foreach(Dependency IN ITEMS ${dependencies})
  find_package(${Dependency} REQUIRED)
endforeach()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL    REQUIRED)


# Need now, not sure why
find_package(Boost REQUIRED COMPONENTS 
              system 
              filesystem
              log 
              log_setup)

###########
## Build ##
###########

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
)

# Main library
add_library(${PROJECT_NAME}
        src/ConvexHullifier.cpp
        src/ImagePreprocessor.cpp
        src/Ransac.cpp
        # src/SuperpixelColorOpenCVSegmenter.cpp
        # src/SuperpixelColorSegmenter.cpp
        src/SuperpixelDepthSegmenter.cpp
        src/Visualizer.cpp
)

ament_target_dependencies(${PROJECT_NAME}
	${dependencies}
)

#target_link_libraries(${PROJECT_NAME}
#        yaml-cpp
#        )

target_compile_options(${PROJECT_NAME} PUBLIC ${FLAGS})

## Declare a C++ executable

# Superpixel algorithm node for depth images (custom implementation)

add_executable(superpixel_depth_segmentation_node
        src/SuperpixelDepthSegmentationNode.cpp
        )
ament_target_dependencies(superpixel_depth_segmentation_node
	${dependencies}
)
target_link_libraries(superpixel_depth_segmentation_node
        ${PROJECT_NAME}
        )

# Superpixel algorithm node for color images (custom implementation)

# add_executable(superpixel_color_segmentation_node
#         src/SuperpixelColorSegmentationNode.cpp
#         )
# ament_target_dependencies(superpixel_color_segmentation_node
# 	${dependencies}
# )        
# target_link_libraries(superpixel_color_segmentation_node
#         ${PROJECT_NAME}
#         )

# Superpixel algorithm node for color images (OTS implementation)

# add_executable(superpixel_color_opencv_segmentation_node
#         src/SuperpixelColorOpenCVSegmentationNode.cpp
#         )
# ament_target_dependencies(superpixel_color_opencv_segmentation_node
# 	${dependencies}
# )                
# target_link_libraries(superpixel_color_opencv_segmentation_node
#         ${PROJECT_NAME}
#         )
