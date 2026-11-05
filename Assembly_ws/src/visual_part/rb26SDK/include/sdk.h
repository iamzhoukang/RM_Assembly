#pragma once

#include <chrono>
#include <cstdint>

#include <opencv2/opencv.hpp>

namespace sdk
{

using Clock = std::chrono::high_resolution_clock;
using ClockPoint = Clock::time_point;

class Camera
{
public:
    enum class CameraBrand
    {
        NullClass,
        Daheng,
        Hik,
    };

    virtual ~Camera() = default;

    CameraBrand camera_brand = CameraBrand::NullClass;

    const char *cap_sn = nullptr;
    int64_t sensor_width = -1;
    int64_t sensor_height = -1;
    ClockPoint time_point = Clock::now();
    double fps = 0.0;
    bool cap_init = false;

    virtual bool CameraInit(
        const char *serial_number,
        bool auto_white_balance,
        int exposure_time_us,
        double gain_factor,
        double gamma) = 0;

    double updateTimer()
    {
        const auto now = Clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - time_point).count();
        time_point = now;

        if (elapsed > 0.0) {
            fps = 1.0 / elapsed;
        }

        return fps;
    }

    // 兼容旧代码。
    double update_timer()
    {
        return updateTimer();
    }
};

} // namespace sdk
