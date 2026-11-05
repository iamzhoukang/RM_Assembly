#pragma once

#include "rb26SDK/include/hik/hik.hpp"

#include <opencv2/opencv.hpp>
#include <string>

class HkCamera {
    private:
    sdk::HikCamera camera_;

    std::string serial_number_;

    bool initialized_ = false;
    bool flip_ = false;
    bool mirror_ = false;

    public:
        HkCamera() = default;
        ~HkCamera() = default;

        // 相机对象内部管理 SDK 句柄，禁止复制。
        HkCamera(const HkCamera&) = delete;
        HkCamera &operator=(const HkCamera&) = delete;

        //初始化MVS SDK ,打开相机并开始采集
        //程序启动只调用一次
        bool initCamera(
            const std::string& serial_number,
            bool auto_white_balance = true,
            int exposure_time_us = 5000,
            double gain = 0.7,
            double gamma = 1.0,
            bool flip = false,
            bool mirror = false
        );

        //先取一帧BGR
        //失败或者超时返回空cv::Mat
        cv::Mat getFrame();    
};