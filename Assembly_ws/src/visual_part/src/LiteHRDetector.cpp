#include "visual_part/LiteHRDetector.hpp"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

LiteHRDetector::LiteHRDetector(
    const std::string &modelPath,
    float bboxPadding,
    const std::string &device
)
{
    if(bboxPadding <= 0.0F){
        throw std::runtime_error(
            "检测框扩展比例必须大于 0");
    }

    this->bbox_padding = bboxPadding;

    try{
        //读取 XML 模型，BIN 文件会被自动加载。
        std::shared_ptr<ov::Model> model =
            this->core.read_model(modelPath);

        //LiteHRDetector 只应有一个图像输入和一个热力图输出
        if (model->inputs().size() != 1 ||
            model->outputs().size() != 1) {
            throw std::runtime_error(
                "LiteHR 模型必须只有一个输入和一个输出");
        }

        const ov::Output<ov::Node> input = model->input(0);
        const ov::PartialShape input_shape = input.get_partial_shape();

           // 输入必须是 [N, 3, H, W]
        if (input_shape.rank().is_dynamic() ||
            input_shape.rank().get_length() != 4) {
            throw std::runtime_error(
                "LiteHR 输入必须是四维 NCHW 张量");
        }

        // 批次可以是动态的，也可以固定为 1
        if (input_shape[0].is_static() &&
            input_shape[0].get_length() != 1) {
            throw std::runtime_error(
                "LiteHR 输入批次必须为 1 或动态值");
        }

          // 通道必须明确为 3
        if (input_shape[1].is_dynamic() ||
            input_shape[1].get_length() != 3) {
            throw std::runtime_error(
                "LiteHR 输入必须为 3 通道图像");
        }

        // 当前预处理需要明确的输入宽高
        if (input_shape[2].is_dynamic() ||
            input_shape[3].is_dynamic()) {
            throw std::runtime_error(
                "LiteHR 输入图像的宽高必须为静态值");
        }

        if (input.get_element_type() != ov::element::f32) {
            throw std::runtime_error(
                "LiteHR 输入数据类型必须为 float32");
        }

        this->input_h = static_cast<int>(input_shape[2].get_length());
        this->input_w = static_cast<int>(input_shape[3].get_length());
        if (this->input_h <= 0 ||
            this->input_w <= 0) {
            throw std::runtime_error(
                "LiteHR 输入图像的宽高无效");
        }

        // 原模型批次为动态值 [? ,3,256,256]。
        // 实际程序每次处理一个 YOLO 框，因此固定为 batch=1。
        model->reshape(
            ov::PartialShape{
                1,
                3,
                this->input_h,
                this->input_w
            });

        const ov::Output<ov::Node> output =
            model->output(0);

        const ov::Shape output_shape =
            output.get_shape();

         // 输出必须是 [1, K, heatmap_h, heatmap_w]
        if (output_shape.size() != 4 ||
            output_shape[0] != 1) {
            throw std::runtime_error(
                "LiteHR 输出必须为 [1, K, H, W]");
        }

        if (output.get_element_type() != ov::element::f32) {
            throw std::runtime_error(
                "LiteHR 输出数据类型必须为 float32");
        }

        this->num_keypoints =
            static_cast<int>(output_shape[1]);

        this->heatmap_h =
            static_cast<int>(output_shape[2]);

        this->heatmap_w =
            static_cast<int>(output_shape[3]);

        if (this->num_keypoints <= 0 ||
            this->heatmap_h <= 0 ||
            this->heatmap_w <= 0) {
            throw std::runtime_error(
                "LiteHR 输出张量形状无效");
        }

        this->compiled_model =
            this->core.compile_model(
                model,
                device);

        this->infer_request =
            this->compiled_model.create_infer_request();

        std::cout
            << "[LiteHR] 模型加载成功："
            << modelPath
            << std::endl;

        std::cout
            << "[LiteHR] 输入张量形状：[1, 3, "
            << this->input_h << ", "
            << this->input_w << "]"
            << std::endl;

        std::cout
            << "[LiteHR] 输出张量形状：[1, "
            << this->num_keypoints << ", "
            << this->heatmap_h << ", "
            << this->heatmap_w << "]"
            << std::endl;

        std::cout
            << "[LiteHR] 推理设备："
            << device
            << std::endl;
    }
    catch (const std::exception &exception) {
        throw std::runtime_error(
            "LiteHR 模型初始化失败：" +
            std::string(exception.what()));
    }
}

void LiteHRDetector::preprocess(
    cv::Mat &frame,
    const cv::Rect &box)
{
    if (frame.empty()) {
        throw std::runtime_error(
            "LiteHR 输入图像为空");
    }

    if (frame.type() != CV_8UC3) {
        throw std::runtime_error(
            "LiteHR 输入图像必须为 CV_8UC3 格式");
    }

    if (box.width <= 0 ||
        box.height <= 0) {
        throw std::runtime_error(
            "LiteHR 输入检测框无效");
    }

    // 计算 YOLO 框中心
    const cv::Point2f center(
         static_cast<float>(box.x) +
            static_cast<float>(box.width) * 0.5F,
        static_cast<float>(box.y) +
            static_cast<float>(box.height) * 0.5F
    );

     // 将检测框扩大，防止边缘关键点被裁掉
    float crop_w =
        std::max(
            static_cast<float>(box.width),
            1.0F) *
        this->bbox_padding;

    float crop_h =
        std::max(
            static_cast<float>(box.height),
            1.0F) *
        this->bbox_padding;

     //调整裁剪区域宽高比，使其与模型输入一致
    const float input_aspect =
        static_cast<float>(this->input_w) /
        static_cast<float>(this->input_h);

    if (crop_w > crop_h * input_aspect) {
        crop_h = crop_w / input_aspect;
    }
    else {
        crop_w = crop_h * input_aspect;
    }

    //在原图裁剪区域中选取三个点
    cv::Point2f source_points[3];

    source_points[0] = center;

    source_points[1] =
        center +
        cv::Point2f(
            -crop_w * 0.5F,
            0.0F);

    const cv::Point2f source_direction =
        source_points[0] -
        source_points[1];

    source_points[2] =
        source_points[1] +
        cv::Point2f(
            -source_direction.y,
            source_direction.x);

      //在模型输入图中选择对应的三个点
    const cv::Point2f destination_center(
        static_cast<float>(this->input_w) * 0.5F,
        static_cast<float>(this->input_h) * 0.5F);

    cv::Point2f destination_points[3];

    destination_points[0] =
        destination_center;

    destination_points[1] =
        destination_center +
        cv::Point2f(
            -static_cast<float>(this->input_w) * 0.5F,
            0.0F);

    const cv::Point2f destination_direction =
        destination_points[0] -
        destination_points[1];

    destination_points[2] =
        destination_points[1] +
        cv::Point2f(
            -destination_direction.y,
            destination_direction.x);

     // 原图坐标 -> 模型输入坐标
    const cv::Mat affine =
        cv::getAffineTransform(
            source_points,
            destination_points);

    // 模型输入坐标 -> 原图坐标
    // 后处理时用它还原关键点
    this->inverse_affine =
        cv::getAffineTransform(
            destination_points,
            source_points);

    // 仿射裁剪为模型输入尺寸
    cv::Mat crop;

    cv::warpAffine(
        frame,
        crop,
        affine,
        cv::Size(
            this->input_w,
            this->input_h),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));

    //获取 OpenVINO 输入 Tensor
    ov::Tensor input_tensor =
        this->infer_request.get_input_tensor(0);

    float *input_data =
        input_tensor.data<float>();

    const int area =
        this->input_w *
        this->input_h;

    // BGR -> RGB、HWC -> NCHW，并按照训练参数归一化
    for (int y = 0; y < this->input_h; ++y) {
        const cv::Vec3b *row =
            crop.ptr<cv::Vec3b>(y);

        for (int x = 0; x < this->input_w; ++x) {
            const cv::Vec3b &pixel = row[x];

            const int index =
                y * this->input_w + x;

            // RGB 第一个通道：R
            input_data[index] =
                (static_cast<float>(pixel[2]) - 123.675F) /
                58.395F;

            // RGB 第二个通道：G
            input_data[area + index] =
                (static_cast<float>(pixel[1]) - 116.280F) /
                57.120F;

            // RGB 第三个通道：B
            input_data[area * 2 + index] =
                (static_cast<float>(pixel[0]) - 103.530F) /
                57.375F;
        }
    }
}

void LiteHRDetector::postprocessing()
{
    if (this->inverse_affine.empty() ||
        this->inverse_affine.rows != 2 ||
        this->inverse_affine.cols != 3) {
        throw std::runtime_error(
            "LiteHR 反仿射矩阵无效");
    }

    //获取模型输出：[1, K, heatmap_h, heatmap_w]
    ov::Tensor output_tensor =
        this->infer_request.get_output_tensor(0);

    const float *output_data =
        output_tensor.data<const float>();

    const int heatmap_area =
        this->heatmap_w *
        this->heatmap_h;

    this->keypointResults.clear();
    this->keypointResults.reserve(
        this->num_keypoints);

    //每个关键点对应一张热力图
    for (int keypoint_id = 0;
         keypoint_id < this->num_keypoints;
         ++keypoint_id) {
        const float *heatmap =
            output_data +
            keypoint_id * heatmap_area;

        //找到当前热力图中的最大值
        int max_index = 0;
        float max_value = heatmap[0];

        for (int index = 1;
             index < heatmap_area;
             ++index) {
            if (heatmap[index] > max_value) {
                max_value = heatmap[index];
                max_index = index;
            }
        }

        if (!std::isfinite(max_value)) {
            throw std::runtime_error(
                "LiteHR 热力图中包含无效数值");
        }

        // 一维索引转换成热力图坐标
        const int max_y =
            max_index / this->heatmap_w;

        const int max_x =
            max_index % this->heatmap_w;

        float heatmap_x =
            static_cast<float>(max_x);

        float heatmap_y =
            static_cast<float>(max_y);

        // 根据峰值两侧的大小，将坐标细化四分之一个像素
        if (max_x >= 1 &&
            max_x < this->heatmap_w - 1 &&
            max_y >= 1 &&
            max_y < this->heatmap_h - 1) {
            const float difference_x =
                heatmap[
                    max_y * this->heatmap_w +
                    max_x + 1] -
                heatmap[
                    max_y * this->heatmap_w +
                    max_x - 1];

            const float difference_y =
                heatmap[
                    (max_y + 1) * this->heatmap_w +
                    max_x] -
                heatmap[
                    (max_y - 1) * this->heatmap_w +
                    max_x];

            if (difference_x > 0.0F) {
                heatmap_x += 0.25F;
            }
            else if (difference_x < 0.0F) {
                heatmap_x -= 0.25F;
            }

            if (difference_y > 0.0F) {
                heatmap_y += 0.25F;
            }
            else if (difference_y < 0.0F) {
                heatmap_y -= 0.25F;
            }
        }

        //热力图坐标转换成模型输入图坐标
        const float input_x =
            (heatmap_x + 0.5F) *
            static_cast<float>(this->input_w) /
            static_cast<float>(this->heatmap_w);

        const float input_y =
            (heatmap_y + 0.5F) *
            static_cast<float>(this->input_h) /
            static_cast<float>(this->heatmap_h);

        //使用反仿射矩阵转换成原图坐标
        const float original_x =
            static_cast<float>(
                this->inverse_affine.at<double>(0, 0) *
                    input_x +
                this->inverse_affine.at<double>(0, 1) *
                    input_y +
                this->inverse_affine.at<double>(0, 2));

        const float original_y =
            static_cast<float>(
                this->inverse_affine.at<double>(1, 0) *
                    input_x +
                this->inverse_affine.at<double>(1, 1) *
                    input_y +
                this->inverse_affine.at<double>(1, 2));

        KeypointResult result;

        result.keypoint_id = keypoint_id;
        result.confidence = max_value;
        result.point =
            cv::Point2f(
                original_x,
                original_y);

        this->keypointResults.emplace_back(
            result);
    }
}

bool LiteHRDetector::Detect(cv::Mat &frame,const cv::Rect &box)
{
    // 每次推理前清除上一次结果，避免保留旧关键点
    this->keypointResults.clear();

    if (frame.empty()) {
        return false;
    }

    if (box.width <= 0 || box.height <= 0) {
        return false;
    }

    try {
        // YOLO 框裁剪、仿射变换、归一化并写入 Tensor
        this->preprocess(frame,box);

        // 同步执行 LiteHRNet 推理
        this->infer_request.infer();

        // 热力图解码并还原到原图坐标
        this->postprocessing();
    }
    catch (const std::exception &exception) {
        std::cerr
            << "[LiteHR] 推理失败："
            << exception.what()
            << std::endl;

        this->keypointResults.clear();

        return false;
    }

    // 正常情况下应该得到模型定义数量的关键点
    return
        static_cast<int>(
        this->keypointResults.size()) == this->num_keypoints;
}
