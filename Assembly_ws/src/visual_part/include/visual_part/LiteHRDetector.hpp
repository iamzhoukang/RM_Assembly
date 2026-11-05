#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


//单个关键点的解码结果 
struct KeypointResult
{
    int keypoint_id; //关键点编号：0～11
    float confidence;        // 热力图最大值
    cv::Point2f point;        // 还原到原图上的坐标
};

class LiteHRDetector
{
    private:
        //模型输入尺寸
        int input_w;
        int input_h;

        //模型输出热力图尺寸
        int heatmap_w;
        int heatmap_h;

        //关键点数量
        int num_keypoints;

        //扩大 yolo 检测框,防止目标边缘被裁切
        float bbox_padding;

        //从模型输入图坐标还原到原图坐标
        cv::Mat inverse_affine;

        //openvino 模型推理相关
        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;

        //根据 YOLO 检测框完成仿射裁剪和图像归一化
        void preprocess(
            cv::Mat &frame,
            const cv::Rect &box);

        // 从 12 张热力图中解码关键点
        void postprocessing();

    public:
            // 当前目标的 12 个关键点
            std::vector<KeypointResult> keypointResults;

            LiteHRDetector(  const std::string &modelPath,
            float bboxPadding = 1.10F,
            const std::string &device = "CPU");

             ~LiteHRDetector() = default;

            //对一个 YOLO 检测框进行关键点推理
            bool Detect(
                cv::Mat &frame,
                const cv::Rect &box);

};