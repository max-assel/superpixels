cmake_minimum_required(VERSION 3.10)
project(superpixels)

set(CATKIN_PACKAGE_DEPENDENCIES
    roscpp
    roslib
    cv_bridge
    image_transport
    )

find_package(catkin REQUIRED COMPONENTS
          ${CATKIN_PACKAGE_DEPENDENCIES}
        )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
    src/SuperpixelSegmenter.cpp
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

# Superpixel algorithm node

add_executable(superpixel_segmentation_node
        src/SuperpixelSegmentationNode.cpp
        )
add_dependencies(superpixel_segmentation_node
        ${PROJECT_NAME}
        ${${PROJECT_NAME}_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(superpixel_segmentation_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )
