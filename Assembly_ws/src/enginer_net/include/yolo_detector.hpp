#pragma once

#include <map>
#include <string>
#include <vector>


#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace enginer_net {

    struct YoloDetection
    {
        cv::Vec4f xyxy; //原图像素坐标,依次为x1,y1,x2,y2
        float score; //置信度
        int class_id; 
    };
    
    struct YoloDetections
    {
        std::vector<YoloDetection> detections;
        std::map<int, std::string> names;
    };


    class YoloDetector
    {
        private:
            struct PreparedImage
            {
                ov::Tensor tensor;
                double scale;
                int pad_x;
                int pad_y;
            };

        PreparedImage preprocess(const cv::Mat& image_bgr);

        ov::CompiledModel compiled_model_;
        ov::InferRequest infer_request_;
        int input_width_;
        int input_height_;
        std::map<int, std::string> names_;

        public:
        explicit YoloDetector(const std::string& model_path,const std::string&device = "CPU");

        YoloDetections predict(const cv::Mat& image_bgr, float confidence = 0.55f);

    };

};


