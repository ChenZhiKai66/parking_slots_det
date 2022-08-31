cmake_minimum_required(VERSION 2.8.3)
project(parking_lot_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS  
  message_generation
  roscpp
  rospy
  std_msgs
  geometry_msgs
)


add_message_files(
  DIRECTORY msg
  FILES
    parking_lot.msg
    parking_lots.msg
)


generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
