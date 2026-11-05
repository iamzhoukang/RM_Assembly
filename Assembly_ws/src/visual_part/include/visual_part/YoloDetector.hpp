#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

// 单个yolo的检测结果
struct YoloResult {
    int class_id;     // 检测到的目标类别ID
    float confidence; // 检测到的目标置信度
    cv::Rect box;     // 检测到的目标边界框
};

class YoloDetector {
private:
    // 模型输入尺寸,从openvino模型中获取
    int input_w;
    int input_h;

    // 模型最多输出的目标数量
    int num_boxes;

    // 置信度阈值
    float scoreThreshold;

    // letterbox,还原参数坐标
    float scale;
    int pad_w;
    int pad_h;

    // OpenVINO 运行时对象。
    ov::Core core;                    // 读取模型和cpu/gpu等设备
    ov::CompiledModel compiled_model; // 编译后模型 = trt engine
    ov::InferRequest infer_request;   // 执行一次推理 = trt context

    // 图像预处理：
    // BGR -> letterbox -> RGB -> NCHW -> float32 / 255。
    void preprocess(cv::Mat &frame);

    // 解析 [1, N, 6] 输出并映射回原图。
    void postprocessing(int original_width, int original_height);

public:
    std::vector<YoloResult> detectResults;

    YoloDetector(const std::string &modelPath, float scoreThreshold,
                 const std::string &device = "CPU");

    ~YoloDetector() = default;

    // 完成预处理、推理和后处理。
    bool Detect(cv::Mat &frame);
};
