#include "../../include/hik/hik.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>

namespace sdk
{

bool HikCamera::chooseCamera(
    MV_CC_DEVICE_INFO **device_info,
    const char *serial_number,
    std::size_t &camera_index)
{
    if (device_info == nullptr || serial_number == nullptr) {
        return false;
    }

    for (std::size_t i = 0; i < device_count_; ++i) {
        if (device_info[i] == nullptr) {
            continue;
        }

        const char *candidate_serial =
            reinterpret_cast<const char *>(
                device_info[i]->SpecialInfo.stUsb3VInfo.chSerialNumber);

        std::cout << "device " << i << " serial number: "
                  << candidate_serial << std::endl;

        if (std::strcmp(serial_number, candidate_serial) == 0) {
            camera_index = i;
            return true;
        }
    }

    return false;
}

bool HikCamera::CameraInit(
    const char *serial_number,
    bool auto_white_balance,
    int exposure_time_us,
    double gain_factor,
    double gamma)
{
    if (serial_number == nullptr || serial_number[0] == '\0') {
        std::cerr << "Hik camera serial number is empty" << std::endl;
        return false;
    }

    if (camera_handle_ != nullptr) {
        capture_stop();
    }

    cap_sn = serial_number;
    camera_brand = CameraBrand::Hik;

    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
    if (nRet != MV_OK) {
        std::cerr << "Hik camera enumeration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        return false;
    }

    device_count_ = device_list.nDeviceNum;
    if (device_count_ == 0) {
        std::cerr << "No Hik camera was found" << std::endl;
        return false;
    }

    std::size_t camera_index = 0;
    if (!chooseCamera(
            device_list.pDeviceInfo,
            serial_number,
            camera_index)) {
        std::cerr << "Hik camera was not found: "
                  << serial_number << std::endl;
        return false;
    }

    nRet = MV_CC_CreateHandle(
        &camera_handle_,
        device_list.pDeviceInfo[camera_index]);
    if (nRet != MV_OK) {
        std::cerr << "Hik camera handle creation failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        camera_handle_ = nullptr;
        return false;
    }

    nRet = MV_CC_OpenDevice(camera_handle_);
    if (nRet != MV_OK) {
        std::cerr << "Hik camera open failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        MV_CC_DestroyHandle(camera_handle_);
        camera_handle_ = nullptr;
        return false;
    }
    device_open_ = true;

    updateTimer();

    if (!capture_start(
            exposure_time_us,
            gain_factor,
            gamma,
            auto_white_balance)) {
        std::cerr << "Hik camera capture start failed" << std::endl;
        capture_stop();
        return false;
    }

    MVCC_INTVALUE_EX width{};
    MVCC_INTVALUE_EX height{};

    if (MV_CC_GetIntValueEx(
            camera_handle_, "Width", &width) == MV_OK) {
        sensor_width = width.nCurValue;
    }

    if (MV_CC_GetIntValueEx(
            camera_handle_, "Height", &height) == MV_OK) {
        sensor_height = height.nCurValue;
    }

    cap_init = true;
    return true;
}

HikCamera::~HikCamera()
{
    // capture_stop() 已经负责停止采集、关闭设备和销毁句柄，
    // 这里不能再次调用 MV_CC_CloseDevice，否则会重复关闭设备。
    capture_stop();
    MV_CC_Finalize();
}

bool HikCamera::capture_stop()
{
    bool success = true;

    if (camera_handle_ == nullptr) {
        grabbing_ = false;
        device_open_ = false;
        cap_init = false;
        return true;
    }

    if (grabbing_) {
        nRet = MV_CC_StopGrabbing(camera_handle_);
        if (nRet != MV_OK) {
            std::cerr << "MV_CC_StopGrabbing failed, error: 0x"
                      << std::hex << nRet << std::dec << std::endl;
            success = false;
        }
        grabbing_ = false;
    }

    if (device_open_) {
        nRet = MV_CC_CloseDevice(camera_handle_);
        if (nRet != MV_OK) {
            std::cerr << "MV_CC_CloseDevice failed, error: 0x"
                      << std::hex << nRet << std::dec << std::endl;
            success = false;
        }
        device_open_ = false;
    }

    nRet = MV_CC_DestroyHandle(camera_handle_);
    if (nRet != MV_OK) {
        std::cerr << "MV_CC_DestroyHandle failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        success = false;
    }

    camera_handle_ = nullptr;
    cap_init = false;
    return success;
}

bool HikCamera::capture_start(
    double exposure_time_us,
    double gain,
    double gamma,
    bool auto_white_balance)
{
    if (camera_handle_ == nullptr || !device_open_) {
        return false;
    }

    nRet = MV_CC_SetEnumValue(
        camera_handle_,
        "BalanceWhiteAuto",
        auto_white_balance
            ? MV_BALANCEWHITE_AUTO_CONTINUOUS
            : MV_BALANCEWHITE_AUTO_OFF);
    if (nRet != MV_OK) {
        std::cerr << "BalanceWhiteAuto configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetEnumValue(
        camera_handle_,
        "ExposureAuto",
        MV_EXPOSURE_AUTO_MODE_OFF);
    if (nRet != MV_OK) {
        std::cerr << "ExposureAuto configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetEnumValue(
        camera_handle_,
        "GainAuto",
        MV_GAIN_MODE_OFF);
    if (nRet != MV_OK) {
        std::cerr << "GainAuto configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetBoolValue(
        camera_handle_, "GammaEnable", true);
    if (nRet != MV_OK) {
        std::cerr << "GammaEnable configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetEnumValue(
        camera_handle_,
        "GammaSelector",
        MV_GAMMA_SELECTOR_USER);
    if (nRet != MV_OK) {
        std::cerr << "GammaSelector configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetFloatValue(
        camera_handle_, "Gamma", gamma);
    if (nRet != MV_OK) {
        std::cerr << "Gamma configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
    }

    nRet = MV_CC_SetFloatValue(
        camera_handle_, "ExposureTime", exposure_time_us);
    if (nRet != MV_OK) {
        std::cerr << "ExposureTime configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        return false;
    }

    MVCC_FLOATVALUE gain_range{};
    nRet = MV_CC_GetFloatValue(
        camera_handle_, "Gain", &gain_range);
    if (nRet != MV_OK) {
        std::cerr << "Gain range query failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        return false;
    }

    const double clamped_gain = std::clamp(gain, 0.0, 1.0);
    const double gain_value =
        gain_range.fMin +
        (gain_range.fMax - gain_range.fMin) * clamped_gain;

    nRet = MV_CC_SetFloatValue(
        camera_handle_, "Gain", gain_value);
    if (nRet != MV_OK) {
        std::cerr << "Gain configuration failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        return false;
    }

    nRet = MV_CC_StartGrabbing(camera_handle_);
    if (nRet != MV_OK) {
        std::cerr << "MV_CC_StartGrabbing failed, error: 0x"
                  << std::hex << nRet << std::dec << std::endl;
        return false;
    }

    grabbing_ = true;
    return true;
}

cv::Mat HikCamera::getFrame(bool flip, bool mirror)
{
    if (camera_handle_ == nullptr || !grabbing_) {
        return {};
    }

    MV_FRAME_OUT frame{};
    nRet = MV_CC_GetImageBuffer(camera_handle_, &frame, 100);
    if (nRet != MV_OK) {
        ++error_num;
        return {};
    }

    bool buffer_released = false;
    const auto release_buffer = [&]() {
        if (!buffer_released) {
            const int release_ret =
                MV_CC_FreeImageBuffer(camera_handle_, &frame);
            if (release_ret != MV_OK) {
                std::cerr << "MV_CC_FreeImageBuffer failed, error: 0x"
                          << std::hex << release_ret << std::dec
                          << std::endl;
            }
            buffer_released = true;
        }
    };

    cv::Mat image_rgb;

    try {
        const auto pixel_type = frame.stFrameInfo.enPixelType;
        const cv::Size size(
            static_cast<int>(frame.stFrameInfo.nWidth),
            static_cast<int>(frame.stFrameInfo.nHeight));

        // OpenCV 的 Bayer 常量命名与直觉相反：只有 "xxx2BGR" 这一组是真枚举，
        // "xxx2RGB" 只是少数几个的重定义。下表按「传感器 2x2 排列 -> 输出内存顺序 R,G,B」
        // 实测得到（OpenCV 4.5 系列没有四字母别名 xxxRGGB2RGB，4.9 才加入）。
        const static std::unordered_map<
            MvGvspPixelType,
            cv::ColorConversionCodes> bayer_to_rgb = {
            {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},  // GRBG
            {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},  // RGGB
            {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},  // GBRG
            {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR},  // BGGR
        };

        const auto bayer_it = bayer_to_rgb.find(pixel_type);
        if (bayer_it != bayer_to_rgb.end()) {
            cv::Mat raw_image(size, CV_8UC1, frame.pBufAddr);
            cv::cvtColor(raw_image, image_rgb, bayer_it->second);
        } else if (pixel_type == PixelType_Gvsp_RGB8_Packed) {
            // 必须 clone：FreeImageBuffer 后 SDK 缓冲区就失效。
            cv::Mat raw_image(size, CV_8UC3, frame.pBufAddr);
            image_rgb = raw_image.clone();
        } else if (pixel_type == PixelType_Gvsp_BGR8_Packed) {
            cv::Mat raw_image(size, CV_8UC3, frame.pBufAddr);
            cv::cvtColor(raw_image, image_rgb, cv::COLOR_BGR2RGB);
        } else if (pixel_type == PixelType_Gvsp_Mono8) {
            cv::Mat raw_image(size, CV_8UC1, frame.pBufAddr);
            cv::cvtColor(raw_image, image_rgb, cv::COLOR_GRAY2RGB);
        } else {
            std::cerr << "Unsupported Hik pixel type: 0x"
                      << std::hex << pixel_type << std::dec << std::endl;
            release_buffer();
            return {};
        }

        if (flip) {
            cv::flip(image_rgb, image_rgb, 0);
        }
        if (mirror) {
            cv::flip(image_rgb, image_rgb, 1);
        }
    } catch (const cv::Exception &exception) {
        std::cerr << "OpenCV frame conversion failed: "
                  << exception.what() << std::endl;
        release_buffer();
        return {};
    }

    // 在这里释放 SDK 缓冲区，返回的 image_rgb 已经完全独立拥有数据。
    release_buffer();

    updateTimer();
    cap_init = true;
    return image_rgb;
}

} // namespace sdk
