//
// Created by yaione on 3/7/22.
//

#ifndef RM_STANDARD2022_DAHENG_H
#define RM_STANDARD2022_DAHENG_H

#include "GxIAPI.h"
#include "DxImageProc.h"

#include "../sdk.h"

using namespace std;
using namespace cv;

namespace sdk {
    class DahengCamera : public Camera{
    private:
        GX_STATUS status = GX_STATUS_SUCCESS;
        GX_DEV_HANDLE hDevice = nullptr;
        GX_FRAME_DATA frameData{};
        void *pRaw8Buffer = nullptr;
        void *pMirrorBuffer = nullptr;
        void *pRGBframeData = nullptr;
        void *pGammaLut = nullptr;
        int64_t PixelFormat = GX_PIXEL_FORMAT_BAYER_GR8;
        int64_t ColorFilter = GX_COLOR_FILTER_NONE;

    public:

        static bool CameraSDKInit(){
            
            if(GXInitLib()!= DX_OK) return false;

            return true;
        }


        /// @brief                    初始化相机
        /// @param serial_number      相机序列号
        /// @param auto_white_balance 自动白平衡
        /// @param exposure_time_us   曝光时间
        /// @param gain_factor        增益
        /// @param gamma              gamma
        /// @param aim                
        /// @return                   初始化是否成功，若成功返回true
        bool CameraInit(
            const char *serial_number,
            bool auto_white_balance = false,
            int exposure_time_us = 2000,
            double gain_factor = 1.0,
            double gamma = 1.0) override;

        ~DahengCamera() ;

        /// @brief 
        /// @param flip 是否进行垂直翻转。
        /// @param mirror 是否进行水平镜像。
        /// @return 
        Mat getFrame(bool flip = false, bool mirror = false);

        /// @brief 设置图像格式
        /// @param pImageBuf 图像数据地址（启用帧信息后，pImgBuf包含图像数据和帧信息数据）
        /// @param pImageRaw8Buf Raw8 位图像地址
        /// @param pImageRGBBuf RGB图像地址
        /// @param nImageWidth 图像宽
        /// @param nImageHeight 图像高
        /// @param nPixelFormat 数据格式
        /// @param nPixelColorFilter Bayer阵列的色彩滤波器类型（如RG, GB等），用于正确解马赛克。
        /// @param flip 是否进行垂直翻转。
        /// @param mirror 是否进行水平镜像。
        void ProcessData(void *pImageBuf, void *pImageRaw8Buf, void *pImageRGBBuf, int nImageWidth, int nImageHeight,
                         int nPixelFormat, int nPixelColorFilter, bool flip = false, bool mirror = false) ;
    };
}


#endif //RM_STANDARD2022_DAHENG_H
