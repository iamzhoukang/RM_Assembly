---
license: agpl-3.0
datasets:
- hkustenterprize/RM26_engineer_exchange
pipeline_tag: object-detection
tags:
- robomaster
---
RM2026 工程机器人视觉模型的 **OpenVINO FP16** 部署权重，用于 NUC / 边缘推理。

```
RM26_engineer_exchange_mode/
├── yolopose/v11.02/          # 单阶段：检测 + 12 关键点
├── yolo/v10.12/              # 仅检测 bbox（两阶段 pipeline 第一阶段）
└── litehr/
    ├── litehrnet30_v11.0/    # 两阶段：12 点热力图
    ├── litehrnet30_v9.1/     # 同上架构，v9.1 数据训练
    └── litehrnet18_v11.0/    # 轻量版 12 点热力图
```

每个 `openvino_best/` 含 `.xml` + `.bin`（LiteHR 另含 `.onnx` 备用）。

---

## yolopose / v11.02

单阶段 YOLO11s-Pose，同时输出 pillar / exchange 的 bbox 与 12 关键点。

| 项 | 值 |
|----|-----|
| **Base model** | YOLO11s-pose（Ultralytics，自 `yolo_v9.121` 权重继续训练） |
| **训练数据** | `dataset_finalversion`（45,400 张） |
| **运行时** | OpenVINO FP16 |
| **文件** | `best.xml`, `best.bin`, `metadata.yaml` |

**Input**

| 属性 | 值 |
|------|-----|
| 名称 | `x` |
| Shape | `[1, 3, 640, 640]` |
| 类型 | FP32 |
| 预处理 | BGR `uint8` → letterbox 640×640 → `/255` → NCHW |

**Output**

| 属性 | 值 |
|------|-----|
| Shape | `[1, 300, 42]` |
| 类型 | FP32 |
| 含义 | 最多 300 个检测；每行 42 维 = `x1,y1,x2,y2,score,class` + 12×`(kx, ky, kscore)` |
| 类别 | `0` pillar，`1` exchange |
| 关键点 | 12 点：TL, TR, BL, BR, ring, light_BR, light_TR, shell_R, shell_M, shell_L, light_TL, light_BL |

后处理：置信度阈值 + NMS（导出时 `nms: true`）。

---

## yolo / v10.12

单阶段 YOLO26s 检测，仅 bbox，无关键点。常与 LiteHR 组成两阶段 pipeline。

| 项 | 值 |
|----|-----|
| **Base model** | YOLO26s（Ultralytics，自 `yolo26s_det_v10.08` 继续训练） |
| **训练数据** | `dataset_v10.0_det` |
| **运行时** | OpenVINO FP16，**end2end**（内置 NMS） |
| **文件** | `best.xml`, `best.bin`, `metadata.yaml` |

**Input**

| 属性 | 值 |
|------|-----|
| Shape | `[1, 3, 640, 640]` |
| 类型 | FP32 |
| 预处理 | 同 YOLO Pose |

**Output**

| 属性 | 值 |
|------|-----|
| Shape | `[1, 300, 6]` |
| 类型 | FP32 |
| 含义 | 最多 300 个检测；每行 `x1, y1, x2, y2, score, class_id` |
| 类别 | `0` pillar，`1` exchange |

---

## litehr / litehrnet30_v11.0

Top-down 姿态估计：输入 **已裁剪的 ROI**，输出 12 通道热力图，需 DARK 解码得到关键点。

| 项 | 值 |
|----|-----|
| **Base model** | MMPose LiteHRNet-30 + `PillarHeatmapHeadWithVis` |
| **训练数据** | `dataset_v11.0`（exchange12 标注） |
| **运行时** | OpenVINO FP16 |
| **文件** | `hrnet_heatmap.xml`, `hrnet_heatmap.bin`, `hrnet_heatmap.onnx` |

**Input**

| 属性 | 值 |
|------|-----|
| 名称 | `input` |
| Shape | `[B, 3, 256, 256]`（B 动态） |
| 类型 | FP32 |
| 预处理 | BGR → RGB；按 bbox 仿射裁剪至 256×256；ImageNet 归一化 `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]` |

**Output**

| 属性 | 值 |
|------|-----|
| 名称 | `heatmaps` |
| Shape | `[B, 12, 128, 128]` |
| 类型 | FP32 |
| 后处理 | DARK 解码 → 12 个 `(x, y)` 映射回原图坐标；可选 visibility 分支 |

**关键点**（索引 0–11）：pillar 5 点 + exchange 7 点（同 `dataset_finalversion` schema）。

> 需外部提供 bbox（通常由 `yolo/v10.12` 给出 pillar 框）。

---

## litehr / litehrnet30_v9.1

与 **litehrnet30_v11.0** 同架构、同 I/O；差异仅在训练数据版本（`dataset_v9.1` vs `dataset_v11.0`）。

| Input | `[B, 3, 256, 256]` FP32 |
| Output | `[B, 12, 128, 128]` FP32 heatmaps |

---

## litehr / litehrnet18_v11.0

LiteHRNet-**18** 轻量版，I/O 与 v30 相同，参数量更小。

| 项 | 值 |
|----|-----|
| **Base model** | MMPose LiteHRNet-18 + `PillarHeatmapHeadWithVis` |
| **训练数据** | `dataset_v11.0`（exchange12） |
| **文件** | `litehrnet18_exchange12_heatmap.xml`, `.bin`, `.onnx` |

| Input | `[B, 3, 256, 256]` FP32 |
| Output | `[B, 12, 128, 128]` FP32 heatmaps |

---

## 部署组合

| 方案 | 模型 | 说明 |
|------|------|------|
| 单阶段 | `yolopose/v11.02` | bbox + 12 kp 一次输出 |
| 两阶段 | `yolo/v10.12` → `litehr/*` | YOLO 检框 + HRNet 精修关键点 |
