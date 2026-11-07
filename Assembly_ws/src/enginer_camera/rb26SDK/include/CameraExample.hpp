#pragma once

#include "daheng/daheng.hpp"
#include "hik/hik.hpp"

#include <iostream>
#include <type_traits>

namespace sdk
{

template <class CameraType>
class CameraExample : public CameraType
{
    static_assert(
        std::is_same<CameraType, HikCamera>::value ||
        std::is_same<CameraType, DahengCamera>::value,
        "CameraType must be HikCamera or DahengCamera");

public:
    cv::Mat frame;

    ~CameraExample() override = default;

    void printResolution() const
    {
        std::cout << "resolution: width " << this->sensor_width
                  << ", height " << this->sensor_height << std::endl;
    }

    void printFps() const
    {
        std::cout << "fps: " << this->fps << std::endl;
    }

    cv::Mat getFrame(bool flip = false, bool mirror = false)
    {
        frame = CameraType::getFrame(flip, mirror);
        return frame;
    }

    // 兼容旧接口名称。
    void putResolution() const
    {
        printResolution();
    }

    void putFps() const
    {
        printFps();
    }
};

// 兼容旧的类名拼写。
template <class CameraType>
using CameraExmple = CameraExample<CameraType>;

} // namespace sdk
