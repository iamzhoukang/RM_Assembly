#pragma once

#include "MvCameraControl.h"
#include "../sdk.h"

#include <cstddef>

namespace sdk
{

class HikCamera : public Camera
{
private:
    int nRet = MV_OK;
    void *camera_handle_ = nullptr;
    std::size_t error_num = 0;
    std::size_t device_count_ = 0;
    bool device_open_ = false;
    bool grabbing_ = false;

    MV_FRAME_OUT raw{};
    MV_CC_DEVICE_INFO_LIST device_list{};

public:
    static bool CameraSDKInit()
    {
        const int ret = MV_CC_Initialize();
        if (ret != MV_OK) {
            std::cerr << "Hik camera SDK initialization failed, error: 0x"
                      << std::hex << ret << std::dec << std::endl;
            return false;
        }

        return true;
    }

    /// 根据相机序列号选择设备。
    bool chooseCamera(
        MV_CC_DEVICE_INFO **device_info,
        const char *serial_number,
        std::size_t &camera_index);

    // 兼容旧的函数名。
    bool ChoiceCamrea(
        MV_CC_DEVICE_INFO **device_info,
        unsigned char *serial_number,
        std::size_t &camera_index)
    {
        return chooseCamera(
            device_info,
            reinterpret_cast<const char *>(serial_number),
            camera_index);
    }

    /// 初始化相机。
    bool CameraInit(
        const char *serial_number,
        bool auto_white_balance = false,
        int exposure_time_us = 2000,
        double gain_factor = 1.0,
        double gamma = 1.0) override;

    ~HikCamera() override;

    bool capture_stop();

    bool capture_start(
        double exposure_time_us,
        double gain,
        double gamma,
        bool auto_white_balance = true);

    /// 获取一帧已经解码、且由 cv::Mat 自己持有数据的图像。
    /// 彩色图像通道顺序为 RGB，发布前可再转换为 BGR。
    cv::Mat getFrame(bool flip = false, bool mirror = false);
};

} // namespace sdk
