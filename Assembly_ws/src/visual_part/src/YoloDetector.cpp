#include "visual_part/YoloDetector.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

// 构造函数：读取并编译 OpenVINO 模型
YoloDetector::YoloDetector(const std::string &modelPath, float scoreThreshold,
                           const std::string &device) {
    if (!std::isfinite(scoreThreshold) || scoreThreshold < 0.0F || scoreThreshold > 1.0F) {
        throw std::runtime_error("YOLO confidence threshold must be between 0 and 1");
    }

    this->scoreThreshold = scoreThreshold;

    // 读取 XML 模型，BIN 文件会被自动加载。
    std::shared_ptr<ov::Model> model = this->core.read_model(modelPath);

    // 当前 YoloDetector 只处理单输入、单输出模型。
    if (model->inputs().size() != 1 || model->outputs().size() != 1) {
        throw std::runtime_error("YOLO model must have one input and one output");
    }

    const ov::Output<ov::Node> input = model->input(0);

    const ov::Output<ov::Node> output = model->output(0);

    const ov::Shape input_shape = input.get_shape();

    const ov::Shape output_shape = output.get_shape();

    // 检查输入格式：[1, 3, H, W]
    // 不限制 H 和 W，因此可以兼容不同输入分辨率。
    if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3) {
        throw std::runtime_error("YOLO input must be [1, 3, H, W]");
    }

    // 后续预处理会通过 float* 写入输入 Tensor。
    if (input.get_element_type() != ov::element::f32) {
        throw std::runtime_error("YOLO input type must be float32");
    }

    this->input_h = static_cast<int>(input_shape[2]);

    this->input_w = static_cast<int>(input_shape[3]);

    if (this->input_h <= 0 || this->input_w <= 0) {
        throw std::runtime_error("YOLO input height or width is invalid");
    }

    // 检查输出格式：[1, N, 6]
    // N 是模型最大输出数量，可以是 100、300、500 等。
    // 6 表示 x1、y1、x2、y2、score、class_id。
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[2] != 6) {
        throw std::runtime_error("YOLO output must be [1, N, 6]");
    }

    if (output.get_element_type() != ov::element::f32) {
        throw std::runtime_error("YOLO output type must be float32");
    }

    this->num_boxes = static_cast<int>(output_shape[1]);

    if (this->num_boxes <= 0) {
        throw std::runtime_error("YOLO output box count is invalid");
    }

    // 将模型编译到 CPU、GPU 或其他指定设备。
    this->compiled_model = this->core.compile_model(model, device);

    // 创建推理请求，后续每帧重复使用。
    this->infer_request = this->compiled_model.create_infer_request();

    std::cout << "YOLO model loaded: " << modelPath << std::endl;

    std::cout << "Input: [1, 3, " << this->input_h << ", " << this->input_w << "]" << std::endl;

    std::cout << "Output: [1, " << this->num_boxes << ", 6]" << std::endl;

    std::cout << "Device: " << device << std::endl;
}

void YoloDetector::preprocess(cv::Mat &frame) {
    // 1. 计算等比例缩放尺寸
    const float scale_x = static_cast<float>(this->input_w) / static_cast<float>(frame.cols);

    const float scale_y = static_cast<float>(this->input_h) / static_cast<float>(frame.rows);

    this->scale = std::min(scale_x, scale_y);

    int resized_w = static_cast<int>(std::round(frame.cols * this->scale));

    int resized_h = static_cast<int>(std::round(frame.rows * this->scale));

    // 防止极端尺寸下出现 0 或由于浮点误差超出模型尺寸。
    resized_w = std::clamp(resized_w, 1, this->input_w);

    resized_h = std::clamp(resized_h, 1, this->input_h);

    // 将缩放后的图像放在模型输入画布中央。
    this->pad_w = (this->input_w - resized_w) / 2;

    this->pad_h = (this->input_h - resized_h) / 2;

    cv::Mat resized;

    cv::resize(frame, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    // 2. 获取 OpenVINO 输入 Tensor
    ov::Tensor input_tensor = this->infer_request.get_input_tensor(0);

    float *input_data = input_tensor.data<float>();

    const int area = this->input_w * this->input_h;

    // 使用 114 填充整个 letterbox 背景，并同时完成 /255。
    std::fill(input_data, input_data + area * 3, 114.0F / 255.0F);

    // 3. BGR -> RGB、HWC -> NCHW、归一化
    for (int y = 0; y < resized_h; ++y) {
        const cv::Vec3b *row = resized.ptr<cv::Vec3b>(y);

        for (int x = 0; x < resized_w; ++x) {
            const cv::Vec3b &pixel = row[x];

            const int dst_index = (y + this->pad_h) * this->input_w + (x + this->pad_w);

            // OpenCV 图像顺序是 BGR。
            // OpenVINO 模型输入顺序要求 RGB。
            input_data[dst_index] = static_cast<float>(pixel[2]) / 255.0F;

            input_data[area + dst_index] = static_cast<float>(pixel[1]) / 255.0F;

            input_data[area * 2 + dst_index] = static_cast<float>(pixel[0]) / 255.0F;
        }
    }
}

// YOLO 输出后处理
// 输出格式：[1, N, 6]
// 每行：[x1, y1, x2, y2, score, class_id]
void YoloDetector::postprocessing(int original_width, int original_height) {
    // 获取 OpenVINO 推理输出。
    ov::Tensor output_tensor = this->infer_request.get_output_tensor(0);

    const float *output_data = output_tensor.data<const float>();

    for (int i = 0; i < this->num_boxes; ++i) {
        // 每个检测结果占据连续的 6 个 float。
        const float *detection = output_data + i * 6;

        const float confidence = detection[4];

        if (!std::isfinite(confidence) || confidence < this->scoreThreshold) {
            continue;
        }

        float x1 = detection[0];
        float y1 = detection[1];
        float x2 = detection[2];
        float y2 = detection[3];

        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
            continue;
        }

        // 从 letterbox 坐标还原到原图坐标
        x1 = (x1 - this->pad_w) / this->scale;
        y1 = (y1 - this->pad_h) / this->scale;
        x2 = (x2 - this->pad_w) / this->scale;
        y2 = (y2 - this->pad_h) / this->scale;

        // 将坐标限制在原图范围内。
        x1 = std::clamp(x1, 0.0F, static_cast<float>(original_width));

        y1 = std::clamp(y1, 0.0F, static_cast<float>(original_height));

        x2 = std::clamp(x2, 0.0F, static_cast<float>(original_width));

        y2 = std::clamp(y2, 0.0F, static_cast<float>(original_height));

        // 过滤宽度或者高度无效的框。
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        // 左上角向下取整，右下角向上取整，
        // 防止整数转换后丢失边界像素。
        const int left = static_cast<int>(std::floor(x1));

        const int top = static_cast<int>(std::floor(y1));

        const int right = static_cast<int>(std::ceil(x2));

        const int bottom = static_cast<int>(std::ceil(y2));

        if (!std::isfinite(detection[5]) || detection[5] < 0 ||
            static_cast<double>(detection[5]) > std::numeric_limits<int>::max())
            continue;
        const int class_id = static_cast<int>(std::round(detection[5]));

        YoloResult result;

        result.class_id = class_id;
        result.confidence = confidence;
        result.box = cv::Rect(left, top, right - left, bottom - top);

        this->detectResults.push_back(result);
    }

    // 让置信度最高的结果排在最前面。
    std::sort(this->detectResults.begin(), this->detectResults.end(),
              [](const YoloResult &left, const YoloResult &right) {
                  return left.confidence > right.confidence;
              });
}

// yolo的推理入口
bool YoloDetector::Detect(cv::Mat &frame) {
    this->detectResults.clear();
    if (frame.empty() || frame.type() != CV_8UC3) return false;

    const int original_width = frame.cols;
    const int original_height = frame.rows;

    try {
        // 将相机的 BGR 图像写入 OpenVINO 输入 Tensor
        this->preprocess(frame);

        // 同步推理
        this->infer_request.infer();

        // 将 [1, N, 6] 输出还原到原图坐标。
        this->postprocessing(original_width, original_height);

    } catch (const std::exception &exception) {
        std::cerr << "YOLO inference failed: " << exception.what() << std::endl;

        this->detectResults.clear();

        return false;
    }
    return !this->detectResults.empty();
}
