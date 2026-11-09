# enginer_net 算法阅读说明

当前包含 YoloDetector、LiteHRDetector 和 Detector 三个普通 C++ 类。Detector 负责组织两个网络、选择目标并筛选关键点。YOLO 返回原图检测框，LiteHRNet 接收原图及一个 xyxy 框，返回原图关键点和热图分数。两者使用 OpenVINO 同步推理，每个类实例顺序复用自己的 InferRequest。

## 目录

```text
enginer_net/
├── include/
│   ├── yolo_detector.hpp
│   ├── litehr_detector.hpp
│   └── detector.hpp
├── src/
│   ├── yolo_detector.cpp
│   ├── litehr_detector.cpp
│   └── detector.cpp
├── config/net.yaml
├── weights/
├── CMakeLists.txt
├── package.xml
└── read.md
```

## LiteHRNet 的输入输出

```cpp
enginer_net::LiteHRConfig config;
config.model_path = "/绝对路径/hrnet_heatmap.xml";
config.device = "CPU";

enginer_net::LiteHRDetector litehr(config);
enginer_net::LiteHRResult result = litehr.predictBBox(image_bgr, detection.xyxy);
```

image_bgr 表示 BGR 图像。bbox 的顺序是 x1、y1、x2、y2，坐标来自原图。模型目录也可以作为 model_path，目录中优先选择 hrnet_heatmap.xml。

LiteHRConfig 的 input_size 为 cv::Size，mean、std 为 cv::Vec3f，分别在类型上固定为宽高两个值、RGB 三个值。对应 Python 版检查输入尺寸长度为 2、均值和标准差长度为 3。模型的静态尺寸仍会在构造时核对。

| 配置成员 | 默认值 | 含义 |
|---|---|---|
| model_path | 调用时填写 | XML 文件或模型目录 |
| device | CPU | OpenVINO 推理设备 |
| input_size | 256×256 | 网络输入宽、高 |
| mean | 123.675、116.28、103.53 | RGB 均值 |
| std | 58.395、57.12、57.375 | RGB 标准差 |
| bbox_padding | 1.10 | 裁剪区域的扩张比例，取自港科示例运行配置 |

config/net.yaml 已补充 hrnet_model、hrnet_device、hrnet_input_size、hrnet_mean、hrnet_std、bbox_padding，沿用港科配置名。当前算法类接收 LiteHRConfig；YAML 由后续 net_node 读取，再逐项填入结构体。直接创建类时使用的是传入结构体的值。

## 头文件与函数

| 头文件 | 对应类型或函数 | 用途 |
|---|---|---|
| string | std::string | 模型路径和设备名称 |
| vector | std::vector | 保存关键点、分数和候选模型文件 |
| opencv2/core.hpp | Mat、Size、Vec3f、Vec4f、Point2f | 图像、输入尺寸、归一化参数和坐标 |
| openvino/openvino.hpp | Core、CompiledModel、InferRequest、Tensor | 读取模型、编译和执行推理 |
| algorithm | transform、max、max_element | 统一设备名、计算裁剪尺寸、找热图峰值 |
| cctype | std::toupper | 设备名转大写 |
| cmath | std::isnan | 对应 NumPy argmax 和 sign 对 NaN 的处理 |
| filesystem | path、目录遍历 | 按港科规则解析模型路径 |
| stdexcept | runtime_error、invalid_argument | 报告加载和输入契约错误 |
| opencv2/imgproc.hpp | getAffineTransform、warpAffine | 仿射裁剪和越界补黑 |

## 建议抄写顺序

先写 LiteHRConfig、LiteHRResult 和类声明，再写构造函数、preprocess()、decodeHeatmaps()，最后写 predictBBox()。predictBBox() 只组织预处理、绑定张量、推理、解码四步。

### 构造函数

保存输入尺寸、均值、标准差和框扩张比例。按港科规则解析模型文件和设备名，读取模型、编译、创建推理请求。使用 partial_shape，允许模型 batch 动态。核对输入 rank 为 4、静态通道为 3，以及静态宽高与配置一致。

YOLO 模型输入是固定的 [1,3,640,640]，LiteHRNet 当前权重则为 [B,3,256,256]，batch 动态；每次 predictBBox() 实际输入一张裁剪图。

### preprocess()

1. 检查图像为二维、三通道的 BGR 图像，对应 Python 的 H×W×3 检查。
2. 框的宽高各至少按 1 像素处理，再乘 bbox_padding。
3. 根据网络输入宽高比扩展裁剪区域。区域中心保持在原框中心。
4. 使用中心、左侧中点和第三点，构造三点仿射变换。正向矩阵用于裁剪，逆向矩阵用于关键点回到原图。
5. warpAffine() 使用双线性插值，原图外的区域填 0。裁剪完成后转换为 float32。
6. 按 R、G、B 三个平面写入张量，使用 (像素 - mean) / std 归一化。

这里的归一化使用 0～255 像素值和给定均值标准差。YOLO 使用的除以 255 属于另一套预处理，应分别对照实现。

PreparedImage 把本次输入 Tensor 和 inverse_affine 放在一起返回，避免帧之间混用变换参数。

### decodeHeatmaps()

当前权重输出 [B,12,128,128]，按 Python 版取第一个 batch。对每个通道找第一个最大值，得到热图坐标与分数。

峰值在内部时，比较左右、上下相邻值，按差值的符号分别修正正负 0.25 个热图像素。峰值位于边界时直接使用整数位置。

随后计算 (坐标 + 0.5) × 输入尺寸 / 热图尺寸，再用逆仿射变换回到原图。该流程对应港科运行代码中的 _decode_heatmaps()。

返回结果保持模型通道顺序，当前 12 点为 TL、TR、BL、BR、ring、light_BR、light_TR、shell_R、shell_M、shell_L、light_TL、light_BL。scores 保存原始热图峰值。模型通道数由输出读取，结果坐标按计算值返回。

低分关键点剔除、最少可见点数量、检测类别选择和回退，已在 Detector 中接入，对应港科 YoloHRNetBackend 及 _YoloHRNetPipeline。

## YOLO 错误处理对照

| 当前检查 | 港科对应行为 | 处理依据 |
|---|---|---|
| XML 路径存在、后缀、目录候选数 | _resolve_openvino_model_path() 显式检查 | 保留模型解析规则 |
| 输入模型为四维、batch=1、C=3 | OpenVinoYoloBackend 构造函数显式检查 | 保留 |
| 图像非空、CV_8UC3 | 港科入口检查 H×W×3 并转换 uint8；空输入在数组或 OpenCV 运算中失败 | C++ 按 Vec3b 读取像素前固定类型契约，避免按错误内存布局读取 |
| 输出元素数可被 6 整除 | NumPy reshape(-1,6) 不满足条件时抛出错误 | C++ 指针遍历前显式对应这个失败条件 |
| 低于置信度阈值时跳过 | 原 predict() 的 score >= conf 过滤 | 正常筛选流程 |

代码中的路径、形状和六列检查都有原版对应。C++ 的图像类型检查固定了调用契约，原图输入当前为 CV_8UC3 BGR。OpenVINO 和 YAML 解析失败交给库抛出异常，再由调用者统一处理。

names 行为也已对齐。metadata.yaml 不存在或 names 不是映射时保留默认名称；文件存在但没有 names 时返回空名称表，与 Python 中缺省的空字典一致。置信度 0.55 仍采用本项目当前设置，港科示例配置为 0.30。

当前 YOLO 的六列解析适用于检测权重。Pose 权重应使用独立的 42 列解析入口。

## LiteHRNet 错误处理对照

| 当前检查 | 港科对应行为 |
|---|---|
| 模型路径及候选文件 | 与 YOLO 共用的 Python 路径解析规则，优先名改为 hrnet_heatmap |
| 模型 rank、静态通道、静态宽高 | _validate_model_shape() |
| 三通道图像 | predict_bbox() 的 H×W×3 检查 |
| 输出为非空 batch 的四维张量 | Python 对 output[0] 的索引以及对 K、H、W 的形状解包 |
| 热图面积非零 | np.argmax() 对空数组抛出异常 |

空图、OpenCV 不支持的像素深度及其他图像运算错误交给 OpenCV 处理。bbox 宽高小于等于零时，按原版 max(...,1.0) 构造裁剪。越界框由 warpAffine 补黑。

## 编译

在 Assembly_ws 中运行下面命令，构建三个算法库。依赖为 OpenCV、yaml-cpp 和 OpenVINO C++ Runtime。

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select enginer_net
source install/setup.bash
```

CMake 找不到 OpenVINO 时，先加载本机 Runtime 环境，或在构建参数中设置 OpenVINO_DIR，指向包含 OpenVINOConfig.cmake 的目录。

## 验证范围

LiteHRNet 已通过真实 C++ 编译，并与港科 Python 实现进行 CPU 数值对照，覆盖原图内外的框、小数坐标、退化框、不同 padding、动态 batch 和空间维度、非正方形输入、浮点图像、热图边界、并列峰值和 NaN。真实 litehrnet30_v11.0 权重在多组框上输出的 12 点和分数也在容差内一致。

模型 rank、通道、高度、宽度不符合配置的四种错误，两版均拒绝。Ubuntu 的完整 colcon 构建与真实画面精度仍需在你的设备上验证。


## Detector：建议从这里开始抄写

文件为 include/detector.hpp 和 src/detector.cpp。它把港科的 _YoloHRNetPipeline 与 YoloHRNetBackend 合在同一个普通 C++ 类里，复用已经完成的两个推理类。它接收一帧 BGR 图像，返回一组关键点观察结果。

建议按以下顺序阅读和抄写：

1. DetectorConfig：配置两个网络、目标类别、回退及点筛选参数。
2. DetectionMetadata 与 KeypointObservation：先明确输出保存哪些内容。
3. Detector 的构造函数：保存配置，并创建 YOLO、LiteHRNet。
4. expandBBox()：按比例扩大框并裁到图像边界。
5. keypointBBox()：利用整图粗定位结果生成回退框。
6. predictInstances()：YOLO 检测、类别选择、必要时回退、逐框 LiteHRNet 推理。
7. detect()：规范图像类型、选择最高 YOLO 分数目标、按名称取点并进行最终筛选。

### 输入、调用与输出

```cpp
#include "detector.hpp"

enginer_net::DetectorConfig config;
config.yolo_model = "/绝对路径/yolo/best.xml";
config.hrnet.model_path = "/绝对路径/litehr/hrnet_heatmap.xml";

// 创建一次、顺序处理各帧；不要在每帧里重新加载模型。
enginer_net::Detector detector(config);
auto observation = detector.detect(image_bgr);
```

此处使用结构体默认值，其余参数可以在创建 detector 前逐项赋值。当前算法库不主动读取 net.yaml；后续 net_node 声明 ROS 参数、加载 YAML 后，需要把值填入 DetectorConfig，其中 hrnet_model 对应 config.hrnet.model_path，hrnet_input_size 对应 config.hrnet.input_size，其余 LiteHR 参数也填入 hrnet 成员。

如果图像来自相机的 RGB8 消息，net_node 应先用 cv_bridge 转为 BGR8，再传入这里。Detector 不根据画面内容猜测颜色顺序。

| 输出成员 | 类型 | 用途 |
|---|---|---|
| keypoints_2d | vector<Point2d> | 筛选后的原图二维坐标，供后续 PnP 使用 |
| keypoint_names | vector<string> | 与坐标一一对应的名称，用于寻找三维模型点 |
| confidences | vector<double> | 对应关键点的原始热图峰值 |
| bbox_xyxy | Vec4f | 实际送入 LiteHRNet 的框，顺序 x1、y1、x2、y2 |
| metadata | DetectionMetadata | 目标来源、YOLO 结果和筛选前的完整关键点，用于后续调试绘制 |

metadata.status_keypoints_2d、status_keypoint_scores 和 status_keypoint_names 保存筛选前的点；selected_keypoint_names 与最终输出名称一致。metadata.yolo_detections 复用 YoloDetections 保存检测框和类别名称表；内容对应港科的检测字典列表。

### 正常路径与回退路径

```text
BGR 图像 → YOLO
  ├─ 完全没有检测框 → 抛出无目标异常
  ├─ 有 bbox_class_id 对应的框 → crop_margin 扩框
  └─ 有其他类别，但没有目标类别框
       → 整图 LiteHRNet 粗定位
       → 取分数 ≥ fallback_score_threshold 的点
       → 至少 fallback_min_keypoints 个点，且能组成有效框
       → fallback_crop_margin 扩框
  ↓
逐框 LiteHRNet → 选择 YOLO 分数最高的实例 → 按名称取点 → 最终置信度筛选
```

回退框的实例分数取整图粗定位的全部关键点分数均值，bbox_class_id 设为 -1，bbox_source 为 hrnet_fallback；正常检测对应 bbox_source 为 yolo。metadata.source 均为 yolo_hrnet。

按原版先完成所有候选框的 LiteHRNet 推理，再选最高 YOLO 分数目标。该目标可见点不足时直接抛异常，不会改选分数第二的目标。

### 两组置信度配置

| 参数 | 当前值 | 生效位置 |
|---|---|---|
| fallback_score_threshold | 0.01 | 整图粗定位生成回退框 |
| fallback_min_keypoints | 4 | 组成粗定位框的最少点数 |
| fallback_crop_margin | 0.5 | 回退框每侧外扩原框宽/高的 50% |
| crop_margin | 0.0 | 正常 YOLO 框外扩；独立于 LiteHR 的 bbox_padding |
| drop_low_confidence_keypoints | true | 最终结果删除低分点 |
| visibility_score_threshold | 0.60 | 最终保留分数大于等于此值的点 |
| min_visible_keypoints | 5 | 最终保留点数不足时抛异常 |
| min_keypoint_score | 0.0 | 不删除低分点时，任意点低于此值则整次失败 |

这些编排参数采用港科示例系统配置；YOLO 置信度仍保留本项目的 0.55。visibility_score_threshold 使用 optional<double> 表示可选值；设为 std::nullopt 时按港科规则改用 min_keypoint_score。默认 0.60 已有值，直接使用即可。

### Detector 的头文件与错误处理

| 头文件 | 使用位置与作用 |
|---|---|
| optional | 可选可见性阈值；keypointBBox() 用 nullopt 表示粗定位没有有效框 |
| string、vector | 配置、点名、候选实例和输出数组 |
| litehr_detector.hpp、yolo_detector.hpp | 组合已有推理类及其数据类型 |
| algorithm | clamp 限制坐标、min/max 求框、max_element 选择目标 |
| cmath | isfinite 检查回退框坐标；isnan 对齐原版数值处理 |
| map | 按名称查找关键点索引，同名时使用最后一个索引 |
| numeric | accumulate 计算回退粗定位分数之和 |
| limits | infinity 作为最小分数统计初值 |
| iomanip、sstream | 保持低分异常中三位小数的格式 |
| stdexcept | runtime_error、invalid_argument、out_of_range |
| utility | move 将逐框推理结果放入候选实例，避免复制点数组 |

| 失败条件 | 与港科的对应关系 |
|---|---|
| 图像不是二维三通道 | _as_bgr_image() 的 H×W×3 检查；非 uint8 图像先裁剪到 0～255 再截断转换 |
| 没有实例，包括粗定位不能组成框 | detect() 的 found no target instance |
| 最终可见点不足 | visible keypoints below threshold |
| 关闭删点但有点低于最低分数 | keypoint score below threshold |
| 点和分数数量不同 | _select_keypoints_by_name() 的长度检查；回退路径也对应 NumPy 布尔索引的长度要求 |
| 配置点名索引超过模型点数 | NumPy 索引会失败；C++ 显式抛 out_of_range，避免越界访问 |
| 配置要求 0 个粗定位点且实际没有点 | NumPy 对空数组求 min/max 会失败，C++ 显式对应该失败条件 |

模型加载、推理和 OpenCV 错误继续向调用者传播。Detector 自己抛出的异常提示已改为中文，异常类型与触发条件保持原版语义。这里不吞异常、不增加重试或 ROS 日志；后续由 net_node 捕获异常，通过 RCLCPP_WARN/ERROR 打印 e.what()。OpenVINO、OpenCV 等第三方库抛出的原始异常继续向上传递。C++ 固定类型已经表达的 Nx2 坐标和四元素框，不重复进行 Python 数组形状检查。

### 本次 Detector 验证

已将真实 C++ Detector 与港科 Python 的 YoloHRNetBackend 做 23 组对照，覆盖正常多框、扩框、整图回退、空检测、粗定位失败、可见点不足、关闭删点、最低分数失败、名称子集/重复/越界、空名称、可选阈值、非法通道与浮点图像转换。正常、回退、浮点图像三组使用项目真实 LiteHRNet 权重；其余使用可控输出的 OpenVINO 测试模型。

关键点坐标、分数、框和调试元数据在数值容差内一致，主要业务异常触发条件一致；当前提示已改为中文，最初对照时使用与港科相同的英文消息。NumPy 隐式失败在 C++ 中使用相应的安全异常表达，类型和措辞不要求逐字相同。完整 ROS 2 构建、相机联调、PnP 与位姿发布尚未验证或接入。
