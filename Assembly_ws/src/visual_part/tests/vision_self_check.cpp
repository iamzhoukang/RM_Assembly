#include "visual_part/VisualDetector.hpp"
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// 使用固定输出的小模型，验证选框、回退和失败处理，不依赖真实相机或训练权重。
void saveModel(const std::filesystem::path &path, const ov::Shape &inputShape,
               const ov::Shape &outputShape, const std::vector<float> &values) {
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, inputShape);
    auto output = ov::opset8::Constant::create(ov::element::f32, outputShape, values);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    ov::serialize(model, path.string());
}

int main() {
    auto directory = std::filesystem::temp_directory_path() /
                     ("assembly_vision_test_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    try {
        auto exchange = directory / "exchange.xml";
        auto pillar = directory / "pillar.xml";
        auto empty = directory / "empty.xml";
        auto heatmap = directory / "heatmap.xml";
        auto low = directory / "low.xml";
        saveModel(exchange, {1, 3, 640, 640}, {1, 2, 6},
                  {100, 100, 500, 500, 0.9F, 1, 200, 200, 300, 300, 0.99F, 0});
        saveModel(pillar, {1, 3, 640, 640}, {1, 1, 6}, {200, 200, 300, 300, 0.9F, 0});
        saveModel(empty, {1, 3, 640, 640}, {1, 1, 6}, {0, 0, 0, 0, 0, 0});
        std::vector<float> peaks(12 * 16 * 16, 0);
        for (int i = 0; i < 12; ++i) {
            int x = 3 + (i % 4) * 3;
            int y = 3 + (i / 4) * 3;
            peaks[i * 256 + y * 16 + x] = 0.8F;
        }
        saveModel(heatmap, {1, 3, 256, 256}, {1, 12, 16, 16}, peaks);
        saveModel(low, {1, 3, 256, 256}, {1, 12, 16, 16}, std::vector<float>(peaks.size(), 0));
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat noFrame;

        VisualDetector normal(exchange.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.2F);
        VisualResult result = normal.Detect(frame);
        check(result.success && !result.used_fallback && result.class_id == 1,
              "must select exchange even when pillar has a higher score");
        check(result.valid_keypoint_count == 12 && result.yolo_results.size() == 2,
              "must retain both detection classes and twelve semantic keypoints");
        check(result.keypoint(VisualKeypointName::PillarTopLeft).valid, "semantic mapping failed");
        check(!normal.Detect(noFrame).success, "empty frame must not retain a successful result");

        VisualDetector noFallback(pillar.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.2F);
        check(!noFallback.Detect(frame).success, "fallback must be disabled by default");
        VisualDetector fallback(pillar.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.2F, "CPU",
                                "CPU", true);
        result = fallback.Detect(frame);
        check(result.success && result.used_fallback && result.class_id == -1 &&
                  result.confidence == 0 && result.box.area() > 0,
              "fallback must use keypoints without inventing a YOLO class or score");
        VisualDetector noDetection(empty.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.2F, "CPU",
                                   "CPU", true);
        check(!noDetection.Detect(frame).success, "no YOLO detections must not trigger fallback");
        VisualDetector lowScore(pillar.string(), low.string(), 0.6F, 1, 1.1F, 0.2F, "CPU", "CPU",
                                true);
        check(!lowScore.Detect(frame).success, "low-confidence coarse keypoints must be rejected");

        YoloDetector yolo(exchange.string(), 0.6F);
        check(yolo.Detect(frame), "synthetic YOLO inference failed");
        check(!yolo.Detect(noFrame) && yolo.detectResults.empty(), "YOLO retained stale results");
        LiteHRDetector hr(heatmap.string());
        check(hr.Detect(frame, cv::Rect(100, 100, 200, 200)), "synthetic LiteHR inference failed");
        check(!hr.Detect(frame, cv::Rect()) && hr.keypointResults.empty(),
              "LiteHR retained stale results");
        bool rejected = false;
        try {
            YoloDetector invalid(exchange.string(), std::numeric_limits<float>::quiet_NaN());
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "NaN threshold must be rejected at startup");

        std::filesystem::remove_all(directory);
        std::cout << "Vision self-check passed (selection, semantics, fallback, failure and stale "
                     "results).\n";
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove_all(directory);
        std::cerr << "Vision self-check failed: " << error.what() << '\n';
        return 1;
    }
}
