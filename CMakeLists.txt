cmake_minimum_required(VERSION 3.16)
project(kftest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_BUILD_TYPE "Debug")

message(STATUS "Project path : ${PROJECT_SOURCE_DIR}")
message(STATUS "Binary path : ${PROJECT_BINARY_DIR}")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/Autoaim.msg"
    DEPENDENCIES
    std_msgs
)

rosidl_get_typesupport_target(
    cpp_typesupport_target
    ${PROJECT_NAME} rosidl_typesupport_cpp
)

ament_auto_add_executable(${PROJECT_NAME}_node
    DIRECTORY src
)

target_include_directories(${PROJECT_NAME}_node PUBLIC ${PROJECT_SOURCE_DIR}/Fliter)

# 添加 KF 库
add_library(kf_lib STATIC
    Fliter/KF/src/KF.cpp
    Fliter/EKF/src/EKF.cpp
    Fliter/UKF/src/UKF.cpp
    Fliter/Utils/src/BindMatrix.cpp
    Fliter/FliterBase/FilterBase.hpp
)

target_include_directories(kf_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/Fliter
)

include_directories(${PROJECT_SOURCE_DIR}/Fliter)

# 将 kf_lib 添加到链接列表中
target_link_libraries(${PROJECT_NAME}_node
    kf_lib
    ${cpp_typesupport_target}
    ${OpenCV_LIBS}

)

install(DIRECTORY
    launch
    config
    DESTINATION share/${PROJECT_NAME}
)

ament_auto_package()