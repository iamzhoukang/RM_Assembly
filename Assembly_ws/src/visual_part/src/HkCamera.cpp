#include "visual_part/HkCamera.hpp"

#include <iostream>

bool HkCamera::initCamera(
    const std::string &serial_number,
    bool auto_white_balance,
    int exposure_time_us,
    double gain_factor,
    double gamma,
    bool flip,
    bool mirror)
{
    // 防止同一个对象重复初始化相机。
    if (initialized_) {
        return true;
    }

    if (serial_number.empty()) {

        std::cerr << "相机序列号不能为空" << std::endl;
        return false;
    }

    //先初始化sdk
    if (!sdk::HikCamera::CameraSDKInit()) {
        std::cerr << "海康相机 SDK 初始化失败" << std::endl;
        return false;
    }

    // 保存序列号，确保传给底层 SDK 的 c_str() 长期有效。
    serial_number_ = serial_number;
    flip_ = flip;
    mirror_ = mirror;

    if (!camera_.CameraInit(
            serial_number_.c_str(),
            auto_white_balance,
            exposure_time_us,
            gain_factor,
            gamma)) {
        std::cerr << "海康相机初始化失败，序列号: "
                  << serial_number_ << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "海康相机初始化成功，序列号: "
              << serial_number_ << std::endl;
    return true;

}

cv::Mat HkCamera::getFrame()
{
    if (!initialized_) {
        std::cerr << "相机尚未初始化" << std::endl;
        return {};
    }

    // SDK 层返回 RGB 格式图像。
    cv::Mat rgb_frame = camera_.getFrame(
        flip_,
        mirror_);

    if (rgb_frame.empty()) {
        return {};
    }

    if (rgb_frame.type() != CV_8UC3) {
        std::cerr << "不支持的图像类型: "
                  << rgb_frame.type() << std::endl;
        return {};
    }

    cv::Mat bgr_frame;

    try {
        // 后续 OpenCV 和 YOLO 使用 BGR 图像。
        cv::cvtColor(
            rgb_frame,
            bgr_frame,
            cv::COLOR_RGB2BGR);
    } catch (const cv::Exception &exception) {
        std::cerr << "RGB 转 BGR 失败: "
                  << exception.what() << std::endl;
        return {};
    }

    return bgr_frame;
}