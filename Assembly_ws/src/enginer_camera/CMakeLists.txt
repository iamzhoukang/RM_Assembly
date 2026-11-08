cmake_minimum_required(VERSION 3.10)
project(enginer_camera)

# ==========================================
# 0. 全局编译选项
# ==========================================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 强制全项目（包含被引入的静态库）生成 -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 1. 外部依赖系统路径配置
# ==========================================
# 海康 MVS SDK 的路径（默认 /opt/MVS）在 rb26SDK/CMakeLists.txt 中按架构配置
# x86_64: /opt/MVS/lib/64    aarch64: src/hik/lib/arm64

# ==========================================
# 2. 寻找系统与 ROS2 依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_calibration_parsers REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)

# ==========================================
# 3. 编译海康相机的底层驱动 SDK 库
# ==========================================
add_subdirectory(rb26SDK)

# ==========================================
# 4. 编译相机采集节点
# ==========================================
add_executable(camera_node
  src/camera_node.cpp
)

target_include_directories(camera_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  camera_calibration_parsers
  ament_index_cpp
)

# rb26SDK 以 PUBLIC 方式导出 MVS 头文件与库路径
target_link_libraries(camera_node
  rb26SDK
  ${OpenCV_LIBS}
  MvCameraControl
  pthread
  dl
)

# ==========================================
# 5. 安装规则 (打包可执行文件、配置与 Launch 文件)
# ==========================================
install(TARGETS camera_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 跳过版权检查（许可证文件尚未添加）
  set(ament_cmake_copyright_FOUND TRUE)
  # 跳过 cpplint（本项目尚未纳入 Git 仓库）
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
