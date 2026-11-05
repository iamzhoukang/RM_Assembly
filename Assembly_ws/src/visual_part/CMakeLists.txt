cmake_minimum_required(VERSION 3.16)
project(visual_part)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT OpenVINO_DIR)
  if(DEFINED ENV{OpenVINO_DIR})
    set(OpenVINO_DIR "$ENV{OpenVINO_DIR}")
  elseif(EXISTS "$ENV{HOME}/intel/openvino_2025.4.1/runtime/cmake/OpenVINOConfig.cmake")
    set(OpenVINO_DIR "$ENV{HOME}/intel/openvino_2025.4.1/runtime/cmake")
  endif()
endif()
find_package(OpenVINO 2025.4.1 EXACT REQUIRED COMPONENTS Runtime)

# ROS 2 Humble 和系统 OpenCV 使用 libstdc++ 新 ABI，拒绝误用 ABI=0 的 Python wheel。
if(NOT OpenVINO_GLIBCXX_USE_CXX11_ABI STREQUAL "1")
  message(FATAL_ERROR
    "visual_part requires an OpenVINO C++ Runtime built with _GLIBCXX_USE_CXX11_ABI=1")
endif()

get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION_RELEASE)
if(NOT OPENVINO_RUNTIME_LIBRARY)
  get_target_property(OPENVINO_RUNTIME_LIBRARY openvino::runtime IMPORTED_LOCATION)
endif()
get_filename_component(OPENVINO_RUNTIME_DIRECTORY "${OPENVINO_RUNTIME_LIBRARY}" DIRECTORY)

# 海康相机 SDK 和相机封装库。
add_subdirectory(rb26SDK)

add_library(camera_core SHARED
  src/HkCamera.cpp
)

target_include_directories(camera_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_core
  PUBLIC
    rb26SDK
    ${OpenCV_LIBS}
)

target_compile_features(camera_core PUBLIC cxx_std_17)

# OpenVINO YOLO 推理与 [1, N, 6] 输出解码。
add_library(yolo_core SHARED
  src/YoloDetector.cpp
)

target_include_directories(yolo_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_core
  PUBLIC
    openvino::runtime
    ${OpenCV_LIBS}
)

target_compile_features(yolo_core PUBLIC cxx_std_17)
set_target_properties(yolo_core PROPERTIES
  INSTALL_RPATH "${OPENVINO_RUNTIME_DIRECTORY}"
)

# OpenVINO LiteHRNet 推理与关键点热力图解码。
add_library(litehr_core SHARED
  src/LiteHRDetector.cpp
)

target_include_directories(litehr_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(litehr_core
  PUBLIC
    openvino::runtime
    ${OpenCV_LIBS}
)

target_compile_features(litehr_core PUBLIC cxx_std_17)
set_target_properties(litehr_core PROPERTIES
  INSTALL_RPATH "${OPENVINO_RUNTIME_DIRECTORY}"
)

# 联合识别层只串联 YOLO 和 LiteHRNet；不采图、不读取 YAML，也不是 ROS 节点。
# 这里复用两个底层库，所以新增算法时可继续在此处组合，而不用改动模型解码器。
add_library(visual_core SHARED
  src/VisualDetector.cpp
)

target_include_directories(visual_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(visual_core
  PUBLIC
    yolo_core
    litehr_core
)

target_compile_features(visual_core PUBLIC cxx_std_17)

# visual_core 安装后从同一 lib 目录查找两个底层算法库。
set_target_properties(visual_core PROPERTIES
  INSTALL_RPATH "$ORIGIN;${OPENVINO_RUNTIME_DIRECTORY}"
)

# 基于语义关键点的 RANSAC/SQPnP 位姿求解。
add_library(pnp_core SHARED
  src/PnPSolver.cpp
)

target_include_directories(pnp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pnp_core
  PUBLIC
    visual_core
    ${OpenCV_LIBS}
)

target_compile_features(pnp_core PUBLIC cxx_std_17)
set_target_properties(pnp_core PROPERTIES
  INSTALL_RPATH "$ORIGIN;${OPENVINO_RUNTIME_DIRECTORY}"
)

# 唯一的 ROS 2 节点：读取 YAML 参数、从海康相机取帧并调用联合识别层。
# 节点同时完成 PnP 求解、调试显示和 TF 发布。
add_executable(visual_node
  src/VisualNode.cpp
)

target_link_libraries(visual_node
  camera_core
  visual_core
  pnp_core
  ${OpenCV_LIBS}
)

ament_target_dependencies(visual_node
  rclcpp
  ament_index_cpp
  geometry_msgs
  tf2
  tf2_ros
)

target_compile_features(visual_node PUBLIC cxx_std_17)

# ros2 run 的可执行文件位于 lib/visual_part，算法库位于上一层 lib。
set_target_properties(visual_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/..;${OPENVINO_RUNTIME_DIRECTORY}"
)

install(
  TARGETS camera_core yolo_core litehr_core visual_core pnp_core visual_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY rb26SDK/include/hik DESTINATION include/rb26SDK/include)
install(FILES rb26SDK/include/sdk.h DESTINATION include/rb26SDK/include)

# 当前仓库将 config/、model/ 放在工作空间根目录。
# 安装后相对模型路径以 share/visual_part 为根，启动目录不会影响寻址。
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../config/"
  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../model/"
  DESTINATION share/${PROJECT_NAME}/model)


if(BUILD_TESTING)
  enable_testing()
  add_executable(vision_self_check tests/vision_self_check.cpp)
  target_link_libraries(vision_self_check visual_core pnp_core)
  add_test(NAME vision_self_check COMMAND vision_self_check)
endif()

ament_package()
