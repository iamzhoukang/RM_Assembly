# enginer_net 算法阅读说明

当前包含 YoloDetector 和 LiteHRDetector 两个普通 C++ 类。YOLO 返回原图检测框，LiteHRNet 接收原图及一个 xyxy 框，返回原图关键点和热图分数。两者使用 OpenVINO 同步推理，每个类实例顺序复用自己的 InferRequest。

## 目录

```text
enginer_net/
├── include/
│   ├── yolo_detector.hpp
│   └── litehr_detector.hpp
├── src/
│   ├── yolo_detector.cpp
│   └── litehr_detector.cpp
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

低分关键点剔除、最少可见点数量、检测类别选择和回退，属于港科 YoloHRNetBackend 及双网络编排层，后续在 detector 中接入。

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

在 Assembly_ws 中运行下面命令，构建两个算法库。依赖为 OpenCV、yaml-cpp 和 OpenVINO C++ Runtime。

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select enginer_net
source install/setup.bash
```

CMake 找不到 OpenVINO 时，先加载本机 Runtime 环境，或在构建参数中设置 OpenVINO_DIR，指向包含 OpenVINOConfig.cmake 的目录。

## 验证范围

LiteHRNet 已通过真实 C++ 编译，并与港科 Python 实现进行 CPU 数值对照，覆盖原图内外的框、小数坐标、退化框、不同 padding、动态 batch 和空间维度、非正方形输入、浮点图像、热图边界、并列峰值和 NaN。真实 litehrnet30_v11.0 权重在多组框上输出的 12 点和分数也在容差内一致。

模型 rank、通道、高度、宽度不符合配置的四种错误，两版均拒绝。Ubuntu 的完整 colcon 构建与真实画面精度仍需在你的设备上验证。
