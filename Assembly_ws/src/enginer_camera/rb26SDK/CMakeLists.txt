cmake_minimum_required(VERSION 3.16)
project(rb26SDK)

find_package(OpenCV REQUIRED)

add_library(rb26SDK
  src/hik/hik.cpp
)

# rb26SDK 会被链接进 ROS 2 的共享库，需要使用位置无关代码。
set_target_properties(rb26SDK PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

target_include_directories(rb26SDK
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hik
    ${OpenCV_INCLUDE_DIRS}
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(rb26SDK PUBLIC /opt/MVS/lib/64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(rb26SDK PUBLIC src/hik/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(rb26SDK
  PUBLIC
    MvCameraControl
    ${OpenCV_LIBS}
)
