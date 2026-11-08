# enginer_camera 相机发布节点

本包用 rb26SDK 驱动一台海康 CS016，发布 RGB 图像和相机内参，供后续双网络识别、PnP 使用。CameraNode 的类定义和 main() 放在 src/camera_node.cpp 中，回调使用 lambda，节点内只有文件读取、SDK 初始化、设备打开和空帧这四处失败分支。

## 文件分工

| 文件 | 作用 |
|---|---|
| src/camera_node.cpp | 加载参数、打开相机、定时取帧、发布图像和 CameraInfo |
| config/camera.yaml | ROS 节点参数，包含序列号、曝光、增益、翻转、坐标系和取帧周期 |
| config/calibration.yaml | ROS 标准相机标定文件，包含尺寸和 K、D、R、P |
| launch/camera_launch.py | 找到安装目录的 camera.yaml，作为节点参数文件传入 |
| rb26SDK/ | 相机 SDK 封装和厂商头文件，当前 CMake 编译海康实现 |
| CMakeLists.txt | 查找依赖、编译 SDK 和节点、安装 config 与 launch |
| package.xml | 声明 ROS 包的构建和运行依赖 |

## 头文件与代码的对应关系

### C++ 标准库

| 头文件 | 当前使用的类型或函数 | 作用 |
|---|---|---|
| chrono | std::chrono::milliseconds(period_ms) | 将毫秒整数转换成定时器需要的时长 |
| cstdlib | EXIT_SUCCESS、EXIT_FAILURE | 给操作系统返回正常或失败的退出码 |
| memory | std::unique_ptr、std::make_unique、std::make_shared | 管理相机对象及节点对象的生命周期 |
| stdexcept | std::runtime_error | 向 main() 报告文件读取、SDK 初始化或设备打开失败 |
| string | std::string、c_str() | 保存参数、文件路径和序列号，将序列号提供给 SDK |

algorithm、cmath 和 optional 已从当前节点移除。此前版本使用的 all_of、isfinite 和 optional 已随代码精简删除。

### ROS 和图像工具

| 头文件 | 当前使用的类型或函数 | 作用 |
|---|---|---|
| ament_index_cpp/get_package_share_directory.hpp | get_package_share_directory("enginer_camera") | 获取安装后的包共享目录，定位默认标定文件 |
| camera_calibration_parsers/parse.hpp | readCalibration() | 把标准标定 YAML 读入 CameraInfo |
| cv_bridge/cv_bridge.h | cv_bridge::CvImage、toImageMsg() | 将 SDK 返回的 cv::Mat 复制为 ROS Image 消息 |
| rclcpp/rclcpp.hpp | Node、declare_parameter、create_publisher、create_wall_timer、spin、日志宏 | 提供节点、参数、消息通信和执行器功能 |
| sensor_msgs/msg/camera_info.hpp | sensor_msgs::msg::CameraInfo | 保存内参、畸变、图像尺寸和消息头 |
| sensor_msgs/msg/image.hpp | sensor_msgs::msg::Image | 保存像素、编码、尺寸、行跨度和消息头 |
| sensor_msgs/image_encodings.hpp | sensor_msgs::image_encodings::RGB8 | 标识三通道、每通道 8 位、RGB 排列 |
| CameraExample.hpp | sdk::CameraExample<sdk::HikCamera> | 使用海康实现取帧，保存最近一帧并提供 FPS、分辨率打印；设备操作继承自 HikCamera |

CameraExample.hpp 在 HikCamera 上补充最近一帧缓存及分辨率、FPS 打印。当前节点通过 CameraExample<HikCamera> 获取图像，底层由 HikCamera 完成设备操作和像素转换。

## 从 main() 开始读代码

### main()

rclcpp::init() 初始化 ROS。make_shared<CameraNode>() 创建节点并执行构造函数，spin() 持续处理定时器回调。构造或回调抛出的 std::exception 在 main() 中记录，然后执行 shutdown()，以失败退出码结束。

### CameraNode 构造函数

构造函数先声明 frame_id 和 period_ms，再调用 loadCalibration() 和 initCamera()。准备好数据及设备后创建两个发布器，最后创建取帧定时器。

SensorDataQoS 用于图像和 CameraInfo。订阅端要使用兼容的 QoS，例如 rclcpp::SensorDataQoS() 或命令行的 best_effort。

定时器回调写作下面这样。

```cpp
[this]() {
    publishFrame();
}
```

[this] 保存当前节点指针，使回调能调用当前对象的 publishFrame()。定时器由节点成员 timer_ 持有，使用 rclcpp::spin() 的单线程执行器时，取帧及发布在回调中依次完成。

### loadCalibration()

get_package_share_directory() 查找包安装目录，拼出 config/calibration.yaml。参数 calibration_file 可覆盖默认路径。readCalibration() 将文件内容读入 camera_info_，读取失败则抛出异常。

camera.yaml 中的参数键名需要和 declare_parameter() 的字符串一致。标定文件采用相机标定工具的标准格式，由独立的解析器读取。

当前 calibration.yaml 的 K[0] 为零，表示未标定。其矩阵已经全部填写数值，可以先用于采集链路验证。后续 PnP 需要真实内参，标定导出的矩阵、尺寸应对应最终分辨率和翻转设置。代码按要求直接加载这些值。

### initCamera()

先声明并读取设备参数，再执行三步初始化。

1. CameraSDKInit() 初始化 MVS。返回失败时抛出异常。
2. make_unique<CameraExample<HikCamera>>() 创建相机对象。
3. CameraInit() 查找指定序列号的设备，打开、配置并启动采集。

SDK 的实参顺序如下。

```cpp
camera_->CameraInit(
    serial_number_.c_str(),
    white_balance,
    exposure_us,
    gain,
    gamma
);
```

这些参数含不同类型，顺序需要按 hik.hpp 的声明填写。serial_number_ 是成员字符串，生命周期覆盖相机对象，SDK 保存的序列号指针持续有效。camera_ 在节点销毁时自动销毁，HikCamera 的析构负责停止采集、关闭设备、销毁句柄和结束 MVS。

### publishFrame()

getFrame(camera_flip_, camera_mirror_) 取回 RGB 图像。SDK 用 cv::flip() 实现上下翻转和左右镜像，返回的像素数据独立于厂商采集缓冲区。空帧时记录限频警告并跳过本次发布。

成功取帧后，用 now() 记录主机收到解码帧的时刻，同一帧的 Image 和 CameraInfo 共用该时间戳及 frame_id。getFrame() 当前返回 cv::Mat，尚未暴露硬件曝光时间戳。

CameraInfo 的 width、height 写入当前图像尺寸，K、D、R、P 保持文件中的数值。节点直接发布这些数据，使用者应让标定对应当前图像。

CvImage 把消息头、RGB8 编码和 cv::Mat 组织起来，toImageMsg() 复制像素到 Image，再先后发布 CameraInfo 和 Image。

## YAML 参数表

| camera.yaml 参数 | 读取位置 | 当前配置及含义 |
|---|---|---|
| serial_number | initCamera() | 已填写的设备序列号，使用字符串 |
| exposure_time | initCamera() | 2000，单位微秒，即 2 ms |
| gain | initCamera() | 0.0，使用设备最小增益；SDK 将 0～1 映射到设备增益范围 |
| gamma | initCamera() | 1.0，SDK 尝试设置设备的 Gamma 属性 |
| auto_white_balance | initCamera() | true，开启连续自动白平衡 |
| camera_flip | initCamera() | false，上下翻转 |
| camera_mirror | initCamera() | false，左右镜像 |
| period_ms | 构造函数 | 33，主机取帧回调的毫秒周期 |
| frame_id | 构造函数 | camera_optical，标识光学坐标系 |
| calibration_file | loadCalibration() | 可选，标定 YAML 的绝对路径；默认使用包内 config/calibration.yaml |

camera.yaml 顶层的 camera_node 对应节点名，ros__parameters 下填写节点参数。declare_parameter() 的默认值在未传入该参数时生效。当前按启动时的参数配置相机。

period_ms 控制定时器调用节奏，实际发布帧率还受相机、曝光、取帧等待和消息处理影响。Gamma 是否支持以完整相机型号和 MVS 实际返回为准，SDK 当前对 Gamma 设置失败打印日志并继续初始化。

## 发布接口

| 话题 | 消息类型 | 内容 |
|---|---|---|
| /image_raw | sensor_msgs/msg/Image | rgb8 原始图像 |
| /camera_info | sensor_msgs/msg/CameraInfo | 标定参数、当前尺寸、同帧时间戳和坐标系 |

代码使用相对话题 image_raw 和 camera_info，默认命名空间下解析为上表路径。frame_id 是相机光学坐标系名称，图像坐标约定为 X 向右、Y 向下、Z 向前。后续感知配置与相机 TF 使用同一个名称。

## 编译与启动

运行环境为 Ubuntu 22.04、ROS 2 Humble、OpenCV 和海康 MVS SDK。厂商动态库路径沿用 rb26SDK/CMakeLists.txt 的配置。

在 Assembly_ws 中执行下面的命令。

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select enginer_camera
source install/setup.bash
ros2 launch enginer_camera camera_launch.py
```

也可以显式给可执行程序传入源码目录中的参数文件。

```bash
ros2 run enginer_camera camera_node --ros-args \
  --params-file src/enginer_camera/config/camera.yaml
```

自定义标定文件路径时追加下面的参数。

```bash
-p calibration_file:=/绝对路径/calibration.yaml
```

修改源码配置后，重新构建并 source，确认使用的是修改后的安装配置。

在另一个已加载 ROS 环境的终端检查消息。

```bash
ros2 run rqt_image_view rqt_image_view
ros2 topic echo /camera_info --once --qos-reliability best_effort
```

图像查看器选择 /image_raw。先看红蓝颜色是否正确、画面方向是否符合安装，再核对 CameraInfo 的尺寸、frame_id 和消息时间戳。

## 本次修正

参数名统一为 camera.yaml 中已有的 auto_white_balance 和 period_ms。默认标定路径统一为 calibration.yaml，原文件中的 fx、fy、cx、cy 和畸变符号占位已替换成数值占位。launch 入口现在会加载 camera.yaml。代码保留你已修复的 SDK 参数顺序。

当前完成配置键名、数值类型、路径、启动入口和代码差异检查。ROS 2 编译、厂商库链接及 CS016 实机取帧由 Ubuntu 环境验证。


## 画面颜色与亮度调整

当前使用 CameraExample<HikCamera>，图像按照 rgb8 发布。Bayer 转换采用 COLOR_BayerRGGB2RGB 等四字母排列，使输出实际 RGB 顺序和消息编码一致。合成色块验证覆盖 RGGB、GRBG、GBRG、BGGR 四种排列，预期 RGB 为 [220, 100, 30]，修复后的结果均与预期一致。

曝光初值改为 2000 微秒，增益比例改为 0.0。原来的 gain = 1.0 对应设备最大增益，容易放大亮度。实际曝光由现场光线和镜头光圈决定，先在最低增益下调整曝光时间。参数在启动时读取，修改 YAML 后重新启动节点。该初值仍需在 CS016 上确认，颜色修复也需通过红色和蓝色实物复核。
