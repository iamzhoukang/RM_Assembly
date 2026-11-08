#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

// ROS 2 节点、图像消息和标定文件读取
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <camera_calibration_parsers/parse.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

//相机SDK
#include "CameraExample.hpp"

class CameraNode : public rclcpp::Node 
{
    private:
        std::string frame_id;
        std::string serial_number_;
        bool camera_flip_ = false;
        bool camera_mirror_ = false;
        std::unique_ptr<sdk::CameraExample<sdk::HikCamera>> camera_;  //sdk初始化成功后构造,由sdk析构
        sensor_msgs::msg::CameraInfo camera_info_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
        rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
        rclcpp::TimerBase::SharedPtr timer_;

    void loadCalibration()
    {
        const auto default_path = ament_index_cpp::get_package_share_directory("enginer_camera") 
                                + "/config/calibration.yaml";
        const auto path = declare_parameter<std::string>("calibration_file", default_path);
        std::string camera_name;
        const bool loaded = camera_calibration_parsers::readCalibration(path, camera_name, camera_info_);
        if(!loaded){
            throw std::runtime_error("读取相机标定文件失败"+ path);
        }
    }

    void initCamera()
    {
        //sdk会保存序列号指针,字符串的生命周期必须大于sdk对象的生命周期
        serial_number_ = declare_parameter<std::string>("serial_number", "");
        const int exposure_us = declare_parameter<int>("exposure_time", 2000);
        const double gain = declare_parameter<double>("gain", 0.0);
        const double gamma = declare_parameter<double>("gamma", 1.0);
        const bool white_balance = declare_parameter<bool>("auto_white_balance", true);
        camera_flip_ = declare_parameter<bool>("camera_flip", false);
        camera_mirror_ = declare_parameter<bool>("camera_mirror", false);

        const bool sdk_ready = sdk::HikCamera::CameraSDKInit();
        if(!sdk_ready){
            throw std::runtime_error("相机SDK初始化失败");
            }

        camera_ = std::make_unique<sdk::CameraExample<sdk::HikCamera>>();
        const bool opened = camera_->CameraInit(
            serial_number_.c_str(), white_balance, exposure_us, gain, gamma
        );
        if(!opened){
            throw std::runtime_error("相机打开或者采集失败"+ serial_number_ );
        }
    }

    void publishFrame()
    {
        const cv::Mat rgb_frame = camera_->getFrame(camera_flip_, camera_mirror_);
        if(rgb_frame.empty()){
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "采集到空帧,请检查相机连接");
            return;
        }
        //getFrame只返回图像,记录下主机收到解码帧的时间戳,用于ros2消息头
        camera_info_.header.stamp = now();
        camera_info_.header.frame_id = frame_id;
        camera_info_.width = rgb_frame.cols;
        camera_info_.height = rgb_frame.rows;

        sensor_msgs::msg::Image image;
        const cv_bridge::CvImage cv_image(
            camera_info_.header, sensor_msgs::image_encodings::RGB8, rgb_frame
        );
        cv_image.toImageMsg(image);
        info_pub_->publish(camera_info_);
        image_pub_->publish(image);
    }

    public:
        CameraNode(): Node("camera_node")
        {
            frame_id = declare_parameter<std::string>("frame_id", "camera_optical");
            const int period_ms = declare_parameter<int>("period_ms", 33);
            loadCalibration();
            initCamera();

            const auto qos = rclcpp::SensorDataQoS();
            image_pub_ = create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
            info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", qos);
            timer_ = create_wall_timer(
                std::chrono::milliseconds(period_ms), 
                [this](){
                    publishFrame();
                }
            );
            RCLCPP_INFO(get_logger(), "相机%s已经开始采集", serial_number_.c_str());
        }

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    int result = EXIT_SUCCESS;
    try {
        rclcpp::spin(std::make_shared<CameraNode>());
    } catch (const std::exception & e) {
        RCLCPP_ERROR(rclcpp::get_logger("camera_node"), "Error: %s", e.what());
        result = EXIT_FAILURE;
    }
    rclcpp::shutdown();
    return result;
}
