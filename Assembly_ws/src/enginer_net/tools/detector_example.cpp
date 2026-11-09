#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "detector.hpp"

int main(int argc, char** argv)
{
    try {
        const std::filesystem::path share_dir =
            ament_index_cpp::get_package_share_directory("enginer_net");
        const auto params = YAML::LoadFile((share_dir / "config/net.yaml").string())
                                ["net_node"]["ros__parameters"];

        enginer_net::DetectorConfig config;
        config.yolo_model = (share_dir / params["yolo_model"].as<std::string>()).string();
        config.yolo_device = params["yolo_device"].as<std::string>();
        config.yolo_conf = params["yolo_conf"].as<double>();
        config.hrnet.model_path = (share_dir / params["hrnet_model"].as<std::string>()).string();
        config.hrnet.device = params["hrnet_device"].as<std::string>();
        config.hrnet.input_size = cv::Size(
            params["hrnet_input_size"][0].as<int>(), params["hrnet_input_size"][1].as<int>());
        for (int i = 0; i < 3; ++i) {
            config.hrnet.mean[i] = params["hrnet_mean"][i].as<float>();
            config.hrnet.std[i] = params["hrnet_std"][i].as<float>();
        }
        config.hrnet.bbox_padding = params["bbox_padding"].as<double>();
        config.bbox_class_id = params["bbox_class_id"].as<int>();
        config.crop_margin = params["crop_margin"].as<double>();
        config.keypoint_names = params["keypoint_names"].as<std::vector<std::string>>();
        config.fallback_score_threshold = params["fallback_score_threshold"].as<double>();
        config.fallback_min_keypoints = params["fallback_min_keypoints"].as<int>();
        config.fallback_crop_margin = params["fallback_crop_margin"].as<double>();
        config.drop_low_confidence_keypoints = params["drop_low_confidence_keypoints"].as<bool>();
        config.min_keypoint_score = params["min_keypoint_score"].as<double>();
        config.visibility_score_threshold = params["visibility_score_threshold"].as<double>();
        config.min_visible_keypoints = params["min_visible_keypoints"].as<int>();

        const std::string image_path = argc > 1 ? argv[1] : (share_dir / "tools/example.jpg").string();
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("读取图片失败：" + image_path);
        }

        enginer_net::Detector detector(config);
        const auto result = detector.detect(image);

        cv::Mat display = image.clone();
        const auto& box = result.bbox_xyxy;
        cv::rectangle(display, cv::Point(cvRound(box[0]), cvRound(box[1])),
                      cv::Point(cvRound(box[2]), cvRound(box[3])), cv::Scalar(0, 255, 255), 2);
        for (size_t i = 0; i < result.keypoints_2d.size(); ++i) {
            const auto& point = result.keypoints_2d[i];
            const cv::Point position(cvRound(point.x), cvRound(point.y));
            cv::circle(display, position, 4, cv::Scalar(0, 255, 0), -1);
            const auto label = cv::format("%s %.2f", result.keypoint_names[i].c_str(),
                                          result.confidences[i]);
            cv::putText(display, label, position + cv::Point(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        std::cout << "推理完成，保留关键点数量：" << result.keypoints_2d.size() << std::endl;
        cv::imshow("Detector", display);
        cv::waitKey(0);
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "推理示例运行失败：" << error.what() << std::endl;
        return 1;
    }
}
