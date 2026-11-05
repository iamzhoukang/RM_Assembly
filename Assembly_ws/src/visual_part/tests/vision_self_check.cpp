#include "visual_part/VisualDetector.hpp"
#include "visual_part/PnPSolver.hpp"
#include <opencv2/calib3d.hpp>
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

void checkPnP(const VisualResult &weakResult) {
    PnPSolver::ObjectPointArray objectPoints = {{{-0.1, -0.04, 0.04},
                                                 {-0.1, 0.04, 0.04},
                                                 {-0.1, -0.04, -0.04},
                                                 {-0.1, 0.04, -0.04},
                                                 {0, 0, 0},
                                                 {-0.112, 0.114870, 0.035953},
                                                 {-0.112, 0.114870, 0.167312},
                                                 {-0.112, 0.084871, 0.276449},
                                                 {-0.112, 0, 0.325449},
                                                 {-0.112, -0.084871, 0.276449},
                                                 {-0.112, -0.114870, 0.167312},
                                                 {-0.112, -0.114870, 0.035953}}};
    cv::Matx33d camera(1000, 0, 720, 0, 1000, 540, 0, 0, 1);
    std::vector<cv::Point2d> projected;
    cv::projectPoints(std::vector<cv::Point3d>(objectPoints.begin(), objectPoints.end()),
                      cv::Vec3d(0.1, -0.2, 0.15), cv::Vec3d(0.03, -0.02, 1.5), camera, cv::Mat(),
                      projected);
    VisualResult observation;
    observation.success = true;
    for (std::size_t i = 0; i < objectPoints.size(); ++i) {
        observation.keypoints[i] = {static_cast<VisualKeypointName>(i), cv::Point2f(projected[i]),
                                    0.8F, true};
    }
    observation.valid_keypoint_count = 12;
    PnPSolver solver(objectPoints);
    auto pose = solver.Solve(observation, camera, cv::Mat());
    check(pose.has_value() && pose->inlier_count == 12, "clean PnP must remain valid");
    check(std::abs(pose->camera_from_target(2, 3) - 1.5) < 0.0001, "PnP reconstruction changed");
    check(!solver.Solve(weakResult, camera, cv::Mat()).has_value(),
          "low-score keypoints must not produce a pose");

    VisualResult limited = observation;
    for (std::size_t i = 5; i < limited.keypoints.size(); ++i)
        limited.keypoints[i].valid = false;
    limited.valid_keypoint_count = 5;
    check(solver.Solve(limited, camera, cv::Mat()).has_value(), "five good points must pass");
    limited.keypoints[4].valid = false;
    limited.valid_keypoint_count = 4;
    check(!solver.Solve(limited, camera, cv::Mat()).has_value(), "four points must fail");

    // 输入点数合格，RANSAC 剔除外点后不足门槛，也必须拒绝发布。
    PnPOptions requireAll;
    requireAll.min_correspondences = 12;
    PnPSolver strictSolver(objectPoints, requireAll);
    check(strictSolver.Solve(observation, camera, cv::Mat()).has_value(),
          "twelve good points must meet the configured gate");
    observation.keypoints[11].point += cv::Point2f(150, -120);
    cv::setRNGSeed(7);
    pose = solver.Solve(observation, camera, cv::Mat());
    check(pose.has_value() && pose->inlier_count >= 5 && pose->inlier_count < 12,
          "outlier fixture must retain a usable subset");
    cv::setRNGSeed(7);
    check(!strictSolver.Solve(observation, camera, cv::Mat()).has_value(),
          "RANSAC inlier count must meet the configured minimum");
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
        auto weak = directory / "weak.xml";
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
        std::vector<float> weakPeaks = peaks;
        for (float &value : weakPeaks)
            value *= 0.5F;
        saveModel(weak, {1, 3, 256, 256}, {1, 12, 16, 16}, weakPeaks);
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat noFrame;

        VisualDetector normal(exchange.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.6F);
        VisualResult result = normal.Detect(frame);
        check(result.success && !result.used_fallback && result.class_id == 1,
              "must select exchange even when pillar has a higher score");
        check(result.valid_keypoint_count == 12 && result.yolo_results.size() == 2,
              "must retain both detection classes and twelve semantic keypoints");
        check(result.keypoint(VisualKeypointName::PillarTopLeft).valid, "semantic mapping failed");
        check(!normal.Detect(noFrame).success, "empty frame must not retain a successful result");

        VisualDetector noFallback(pillar.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.6F);
        check(!noFallback.Detect(frame).success, "fallback must be disabled by default");
        VisualDetector fallback(pillar.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.6F, "CPU",
                                "CPU", true);
        result = fallback.Detect(frame);
        check(result.success && result.used_fallback && result.class_id == -1 &&
                  result.confidence == 0 && result.box.area() > 0,
              "fallback must use keypoints without inventing a YOLO class or score");
        VisualDetector noDetection(empty.string(), heatmap.string(), 0.6F, 1, 1.1F, 0.6F, "CPU",
                                   "CPU", true);
        check(!noDetection.Detect(frame).success, "no YOLO detections must not trigger fallback");
        VisualDetector lowScore(pillar.string(), low.string(), 0.6F, 1, 1.1F, 0.6F, "CPU", "CPU",
                                true);
        check(!lowScore.Detect(frame).success, "low-confidence coarse keypoints must be rejected");

        VisualDetector weakTarget(exchange.string(), weak.string(), 0.6F, 1, 1.1F, 0.6F);
        VisualResult weakResult = weakTarget.Detect(frame);
        check(weakResult.success && weakResult.valid_keypoint_count == 0,
              "0.4 keypoint scores must be excluded by the 0.6 threshold");
        checkPnP(weakResult);

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
                     "results, strict keypoint and PnP gates).\n";
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove_all(directory);
        std::cerr << "Vision self-check failed: " << error.what() << '\n';
        return 1;
    }
}
