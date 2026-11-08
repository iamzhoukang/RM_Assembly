#include "yolo_detector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace enginer_net
{
    YoloDetector::YoloDetector(const std::string& model_path,const std::string& device)
    {   
        //查找模型文件
        std::filesystem::path path(model_path);
        if(std::filesystem::is_directory(path))
        {
            const auto preferred = path / "best.xml";
            if(std::filesystem::is_regular_file(preferred)){
                path = preferred;
            }else{
                std::vector<std::filesystem::path> candidates;
                for(const auto& entry : std::filesystem::directory_iterator(path)){
                    if(entry.is_regular_file() && entry.path().extension() == ".xml"){
                        candidates.push_back(entry.path());
                    }
                }
                if(candidates.size() != 1){
                    throw std::runtime_error("模型目录中要唯一的XML文件"+ model_path);
                }
                path = candidates.front();
            }
        }

        if(!std::filesystem::is_regular_file(path) || path.extension() != ".xml"){
            throw std::runtime_error("模型文件不存在或不是XML文件: " + path.string());
        }

        //加载模型
        std::string target_device = device.empty() ? "CPU" : device;
        std::transform(target_device.begin(),target_device.end(),target_device.begin(),
                        [](unsigned char c){return static_cast<char>(std::toupper(c));});
        if(target_device == "0" || target_device.rfind("CUDA",0) == 0){
            target_device = "GPU";
        }

        ov::Core core;
        const auto model = core.read_model(path.string());
        compiled_model_ = core.compile_model(model, target_device);
        infer_request_ = compiled_model_.create_infer_request();

        //获取输入输出信息
        const auto shape = compiled_model_.input(0).get_shape();
        if(shape.size() !=4 || shape[0] != 1 || shape[1] != 3){
            throw std::runtime_error("模型输入形状不符合要求，必须是[1,3,H,W]");
        }
        input_height_ = static_cast<int>(shape[2]);
        input_width_ = static_cast<int>(shape[3]);

        names_ = {{0,"local_target"},{1,"exchange"}};
        const auto metadata_path = path.parent_path() / "metadata.yaml";
        if(std::filesystem::exists(metadata_path)){
            auto metadata = YAML::LoadFile(metadata_path.string());
            const auto names = metadata["names"];
            if(names && names.IsMap()){
                names_.clear();
                for(const auto& it : names){
                    names_[it.first.as<int>()] = it.second.as<std::string>();
                }
            }
        }
    }

    YoloDetector::PreparedImage YoloDetector::preprocess(const cv::Mat& image_bgr)
    {
        if(image_bgr.empty() || image_bgr.type() != CV_8UC3){
            throw std::runtime_error("输入图像必须是非空的CV_8UC3类型");
        }

        const double scale = std::min(
            static_cast<double>(input_width_) / image_bgr.cols,
            static_cast<double>(input_height_) / image_bgr.rows
        );
        const int resize_width = static_cast<int>(std::nearbyint(image_bgr.cols * scale));
        const int resize_height = static_cast<int>(std::nearbyint(image_bgr.rows * scale));
        const int pad_x = (input_width_ - resize_width) / 2;
        const int pad_y = (input_height_ - resize_height) / 2;

        cv::Mat resized;
        cv::resize(image_bgr, resized, cv::Size(resize_width, resize_height),0,0,cv::INTER_LINEAR);
        cv::Mat canvas(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect(pad_x, pad_y, resize_width, resize_height)));

        ov::Tensor tensor(ov::element::f32,
                        {1,3,static_cast<size_t>(input_height_),static_cast<size_t>(input_width_)});
        float* data = tensor.data<float>();
        const size_t plane_size = static_cast<size_t>(input_height_) * input_width_;
        for(int y = 0; y < input_height_; ++y){
            const auto* row = canvas.ptr<cv::Vec3b>(y);
            for(int x = 0; x < input_width_; ++x){
                const size_t index = static_cast<size_t>(y) * input_width_ + x;
                // BGR 下标 2 对应 R，写入 R 平面，同时完成 HWC -> NCHW 和归一化
                data[index] = row[x][2] / 255.0f; // R
                data[index + plane_size] = row[x][1] / 255.0f; // G
                data[2 * plane_size + index] = row[x][0] / 255.0f; // B
            }
        }
        return {tensor, scale, pad_x, pad_y};
    }

    YoloDetections YoloDetector::predict(const cv::Mat& image_bgr, float confidence)
    {
        const auto prepared = preprocess(image_bgr);
        infer_request_.set_input_tensor(prepared.tensor);
        infer_request_.infer();
        auto output = infer_request_.get_output_tensor(0);
        if(output.get_size() % 6 != 0){
            throw std::runtime_error("YOLO模型输出无法按[x1,y1,x2,y2,score,class_id]解析");
        }
        const float* data = output.data<float>();

        YoloDetections result;
        result.names = names_;
        for(size_t i =0; i<output.get_size();i+=6){
            if(!(data[i+4] >= confidence)){
                continue;
            }

            YoloDetection detection;
            for(int axis = 0; axis < 4; ++axis){
                const bool is_x = (axis % 2 == 0);
                const int padding = is_x ? prepared.pad_x : prepared.pad_y;
                const double limit = is_x ? image_bgr.cols : image_bgr.rows;
                const float coordinate = (data[i + axis] - padding) / static_cast<float>(prepared.scale);
                detection.xyxy[axis] = std::clamp(coordinate, 0.0f, static_cast<float>(limit));
            }
            detection.score = data[i + 4];
            detection.class_id = static_cast<int>(data[i + 5]);
            result.detections.push_back(detection);
        }
        std::sort(result.detections.begin(), result.detections.end(),
                  [](const YoloDetection& a, const YoloDetection& b){
                      return a.score > b.score;
                  });
        return result;

    }
}