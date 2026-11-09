#include "litehr_detector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace enginer_net
{

LiteHRDetector::LiteHRDetector(const LiteHRConfig& config)
    : input_size_(config.input_size),
      mean_(config.mean),
      std_(config.std),
      bbox_padding_(config.bbox_padding)
{
    std::filesystem::path path(config.model_path);
    if (std::filesystem::is_directory(path)) {
        const auto preferred = path / "hrnet_heatmap.xml";
        if (std::filesystem::is_regular_file(preferred)) {
            path = preferred;
        } else {
            std::vector<std::filesystem::path> candidates;
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".xml") {
                    candidates.push_back(entry.path());
                }
            }
            if (candidates.size() != 1) {
                throw std::runtime_error("模型目录中需要唯一的 XML 文件：" + config.model_path);
            }
            path = candidates.front();
        }
    }
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!std::filesystem::is_regular_file(path) || extension != ".xml") {
        throw std::runtime_error("OpenVINO XML 模型不存在或扩展名不正确：" + path.string());
    }

    std::string device = config.device.empty() ? "CPU" : config.device;
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (device == "0" || device.rfind("CUDA", 0) == 0) {
        device = "GPU";
    }

    ov::Core core;
    const auto model = core.read_model(path.string());
    compiled_model_ = core.compile_model(model, device);
    infer_request_ = compiled_model_.create_infer_request();

    // 允许动态 batch，检查规则对应 _validate_model_shape()。
    const auto shape = compiled_model_.input(0).get_partial_shape();
    if (shape.rank().is_dynamic() || shape.rank().get_length() != 4) {
        throw std::runtime_error("LiteHRNet 输入必须为 NCHW 四维张量");
    }
    if (shape[1].is_static() && shape[1].get_length() != 3) {
        throw std::runtime_error("LiteHRNet 输入通道必须为 3");
    }
    if (shape[2].is_static() && shape[2].get_length() != input_size_.height) {
        throw std::runtime_error("LiteHRNet 模型高度与配置不一致");
    }
    if (shape[3].is_static() && shape[3].get_length() != input_size_.width) {
        throw std::runtime_error("LiteHRNet 模型宽度与配置不一致");
    }
}

LiteHRDetector::PreparedImage LiteHRDetector::preprocess(
    const cv::Mat& image_bgr, const cv::Vec4f& bbox_xyxy)
{
    if (image_bgr.dims != 2 || image_bgr.channels() != 3) {
        throw std::invalid_argument("LiteHRNet 输入必须为 H×W 三通道 BGR 图像");
    }

    const double x1 = bbox_xyxy[0];
    const double y1 = bbox_xyxy[1];
    const double x2 = bbox_xyxy[2];
    const double y2 = bbox_xyxy[3];
    cv::Vec2f scale(
        static_cast<float>(std::max(x2 - x1, 1.0)),
        static_cast<float>(std::max(y2 - y1, 1.0)));
    scale *= static_cast<float>(bbox_padding_);
    const double aspect = static_cast<double>(input_size_.width) / input_size_.height;
    if (scale[0] > scale[1] * aspect) {
        scale[1] = static_cast<float>(scale[0] / aspect);
    } else {
        scale[0] = static_cast<float>(scale[1] * aspect);
    }

    // 与 _bbox_to_square_affine() 相同，用三点确定正向和逆向仿射变换。
    cv::Point2f source[3];
    source[0] = cv::Point2f(static_cast<float>((x1 + x2) * 0.5),
                            static_cast<float>((y1 + y2) * 0.5));
    source[1] = source[0] + cv::Point2f(-0.5f * scale[0], 0.0f);
    const auto source_direction = source[0] - source[1];
    source[2] = source[1] + cv::Point2f(-source_direction.y, source_direction.x);

    cv::Point2f target[3];
    target[0] = cv::Point2f(input_size_.width * 0.5f, input_size_.height * 0.5f);
    target[1] = target[0] + cv::Point2f(-0.5f * input_size_.width, 0.0f);
    const auto target_direction = target[0] - target[1];
    target[2] = target[1] + cv::Point2f(-target_direction.y, target_direction.x);

    const cv::Mat affine = cv::getAffineTransform(source, target);
    const cv::Mat inverse = cv::getAffineTransform(target, source);
    cv::Mat crop;
    cv::warpAffine(image_bgr, crop, affine, input_size_,
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::Mat float_crop;
    crop.convertTo(float_crop, CV_32F);

    ov::Tensor tensor(ov::element::f32,
                      {1, 3, static_cast<size_t>(input_size_.height),
                       static_cast<size_t>(input_size_.width)});
    float* data = tensor.data<float>();
    const size_t plane_size = static_cast<size_t>(input_size_.height) * input_size_.width;
    for (int y = 0; y < input_size_.height; ++y) {
        const auto* row = float_crop.ptr<cv::Vec3f>(y);
        for (int x = 0; x < input_size_.width; ++x) {
            const size_t index = static_cast<size_t>(y) * input_size_.width + x;
            // 原图为 BGR，按 RGB 平面写入；归一化为 (像素 - mean) / std。
            data[index] = (row[x][2] - mean_[0]) / std_[0];
            data[plane_size + index] = (row[x][1] - mean_[1]) / std_[1];
            data[2 * plane_size + index] = (row[x][0] - mean_[2]) / std_[2];
        }
    }
    return {tensor, inverse};
}

LiteHRResult LiteHRDetector::decodeHeatmaps(
    ov::Tensor output, const cv::Mat& inverse_affine) const
{
    const auto shape = output.get_shape();
    // Python 对 output[0].shape 解包为 K、H、W，错误形状同样会失败。
    if (shape.size() != 4 || shape[0] == 0) {
        throw std::runtime_error("LiteHRNet 输出必须为非空的 [B,K,H,W] 热图");
    }
    const size_t count = shape[1];
    const size_t height = shape[2];
    const size_t width = shape[3];
    const size_t plane_size = height * width;
    if (plane_size == 0) {
        throw std::runtime_error("LiteHRNet 热图为空，无法寻找峰值");
    }
    const float* heatmaps = output.data<float>();
    LiteHRResult result;
    result.keypoints.resize(count);
    result.scores.resize(count);

    for (size_t k = 0; k < count; ++k) {
        const float* heatmap = heatmaps + k * plane_size;
        // np.argmax 返回第一个最大值；存在 NaN 时返回第一个 NaN 的位置。
        const float* peak = std::max_element(
            heatmap, heatmap + plane_size,
            [](float a, float b) {
                return !std::isnan(a) && (std::isnan(b) || a < b);
            });
        const size_t offset = static_cast<size_t>(peak - heatmap);
        const size_t x = offset % width;
        const size_t y = offset / width;
        double px = static_cast<double>(x);
        double py = static_cast<double>(y);
        if (x >= 1 && x + 1 < width && y >= 1 && y + 1 < height) {
            const float dx = heatmap[offset + 1] - heatmap[offset - 1];
            const float dy = heatmap[offset + width] - heatmap[offset - width];
            // 对应 np.sign()，相邻热图值决定四分之一像素修正的方向。
            const auto sign = [](float value) {
                if (std::isnan(value)) {
                    return value;
                }
                return static_cast<float>((value > 0.0f) - (value < 0.0f));
            };
            px += sign(dx) * 0.25;
            py += sign(dy) * 0.25;
        }
        const float crop_x = static_cast<float>((px + 0.5) * input_size_.width / width);
        const float crop_y = static_cast<float>((py + 0.5) * input_size_.height / height);
        const double original_x = inverse_affine.at<double>(0, 0) * crop_x
                                  + inverse_affine.at<double>(0, 1) * crop_y
                                  + inverse_affine.at<double>(0, 2);
        const double original_y = inverse_affine.at<double>(1, 0) * crop_x
                                  + inverse_affine.at<double>(1, 1) * crop_y
                                  + inverse_affine.at<double>(1, 2);
        result.keypoints[k] = cv::Point2f(static_cast<float>(original_x), static_cast<float>(original_y));
        result.scores[k] = *peak;
    }
    return result;
}

LiteHRResult LiteHRDetector::predictBBox(const cv::Mat& image_bgr, const cv::Vec4f& bbox_xyxy)
{
    const auto prepared = preprocess(image_bgr, bbox_xyxy);
    infer_request_.set_input_tensor(prepared.tensor);
    infer_request_.infer();
    return decodeHeatmaps(infer_request_.get_output_tensor(0), prepared.inverse_affine);
}

}  // namespace enginer_net
