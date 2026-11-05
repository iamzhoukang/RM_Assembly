# RM Assembly 视觉识别

当前由一个 `visual_node` 负责取图、YOLO 整体框、LiteHRNet 12 个关键点、PnP 和 TF 发布。
识别算法类不依赖 ROS，不包含机械臂运动学或控制逻辑。

```text
HkCamera / 图片 / 视频
          ↓
YoloDetector → VisualDetector → LiteHRDetector
                    ↓
               PnPSolver
                    ↓
    camera_optical_frame → exchange_target
```

## 模型类别

- `pillar = 0`：装配柱的局部框。
- `exchange = 1`：装配站整体框，覆盖装配柱及外壳。

它们不是站体的内侧面、外侧面。当前 LiteHRNet 模型预测柱上的 5 个点与外壳/灯条上的 7 个点，所以默认使用 `exchange` 框裁图。只把 `target_class_id` 改成 0，可能裁掉外壳关键点。

联合识别使用普通结果结构体：

```cpp
VisualResult result = detector.Detect(frame);
if (result.success) {
    // result.box：本次送入关键点网络的框
    // result.keypoints：固定顺序的 12 个语义关键点
    // result.valid_keypoint_count：达到分数阈值的点数
}
```

`success` 表示识别流程成功，不等于 PnP 成功或位姿精度合格。PnP 保持独立判断。

## 当前严格过滤参数

- YOLO 置信度至少 `0.60`，保留原有设置。
- 只有得分达到 `0.60` 的关键点参与 PnP。
- 至少 5 个有效关键点；RANSAC 剔除外点后，也必须至少保留 5 个内点。
- RANSAC 内点重投影阈值仍为 `3 px`，正深度检查和 LM 精修保留。
- LiteHR 回退默认关闭。

关键点阈值和最少有效点数参考港科示例；RANSAC 后也要求 5 个内点是本项目增加的约束。
`pnp.min_correspondences` 同时控制输入点和内点的最低数量。
`3 px` 是 RANSAC 的内点判定阈值，不是最终平均重投影误差发布门槛。
这些检查可能增加遮挡或模糊场景中的漏检，仍需用无目标录像和真实目标录像实测。

## 构建和启动

需要 ROS 2 Humble、OpenCV、海康 MVS SDK 和 ABI=1 的 OpenVINO 2025.4.1 C++ Runtime。
本次没有更改现有 OpenVINO 版本和 ABI 要求。

在仓库根目录执行：

```bash
cd Assembly_ws
colcon build --packages-select visual_part --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
ros2 run visual_part visual_node --ros-args --params-file config/visual.yaml
```

先按实际相机填写序列号和标定参数。配置中的近似内参只用于识别流程联调。

## 离线识别

无需初始化或连接相机；当前包的构建与链接仍依赖 MVS SDK。

```bash
# 使用录像：一帧对应一次定时器调用，播完停止处理，不循环。
ros2 run visual_part visual_node --ros-args --params-file config/visual.yaml \
  -p input.source:=video -p input.path:=/absolute/path/test.mp4

# 使用图片：每个定时周期重复识别该图片。
ros2 run visual_part visual_node --ros-args --params-file config/visual.yaml \
  -p input.source:=image -p input.path:=/absolute/path/test.jpg
```

调试窗口显示两类 YOLO 框及其分数，选中目标另外绘制关键点。
无桌面环境时设置 `-p debug.show_image:=false`。
视频按 `inference.period_ms` 的节奏处理，不保证按原视频帧率播放；离线 TF 时间戳是回放时间。
离线图片和视频仍会走现有 PnP/TF 流程，分辨率与内参必须匹配才能解释位姿。

## 可选 LiteHR 回退

```bash
ros2 run visual_part visual_node --ros-args --params-file config/visual.yaml \
  -p input.source:=video -p input.path:=/absolute/path/test.mp4 \
  -p litehr.use_fallback:=true
```

默认关闭，先用本队录像验证。触发条件是 YOLO 有检测框，但没有指定类别的整体框。
先用整图预测关键点；至少 4 个点达到关键点分数阈值且能够组成有效区域时，以它们的包围框再次运行 LiteHR。
YOLO 完全没有检测框时结束该帧，不从纯背景启动回退。

回退结果的 `used_fallback=true`、`class_id=-1`、`confidence=0`，不会把关键点分数伪装成 YOLO 置信度。
它只提供识别候选，不能保证目标真实存在，后续仍需 PnP 和实测质量检查。

## 自检

```bash
colcon test --packages-select visual_part --event-handlers console_direct+
colcon test-result --verbose
```

`vision_self_check` 生成小型 OpenVINO 模型，检查类别选框、语义点对应、回退开关、低分拒绝、失败后清空旧结果及 PnP 有效点/内点数量门槛。
它不评估真实场景识别率，也不替代 ROS 节点、相机和实机联调。

## 后续接口

视觉继续发布 `camera_optical_frame → exchange_target`，机器人侧提供基座到相机的安装/运动 TF。
安装结构确定后再做对应标定；相机支架会运动时，需要使用图像采集时刻的关节状态。
不要仅修改 `tf.parent_frame` 就把 PnP 结果称作基座坐标，坐标值必须实际经过变换。
