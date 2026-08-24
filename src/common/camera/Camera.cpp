#include "common/camera/Camera.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

using namespace std::chrono;

// GigE相机构造函数
Camera::Camera(const std::string& deviceIp, const std::string& netIp) 
    : handle(nullptr)
    , status(DISCONNECTED)
    , cameraType(GIGE_CAMERA)
    , running(false)
    , deviceIp(deviceIp)
    , netIp(netIp)
    , deviceIndex(0)
    , grabbing(false)
    , needReconnect(false)
    , exposureTime(5000)
    , gain(16.0) {
    
    std::cout << "GigE Camera created with IP: " << deviceIp << std::endl;
}

// USB相机构造函数
Camera::Camera(int deviceIndex) 
    : handle(nullptr)
    , status(DISCONNECTED)
    , cameraType(USB_CAMERA)
    , running(false)
    , deviceIp("")
    , netIp("")
    , deviceIndex(deviceIndex)
    , grabbing(false)
    , needReconnect(false)
    , exposureTime(5000)
    , gain(16.0) {
    
    std::cout << "USB Camera created with device index: " << deviceIndex << std::endl;
}

bool Camera::start() {
    if (running.load()) {
        std::cout << "Camera is already running." << std::endl;
        return true;
    }

    auto now = steady_clock::now();
    lastReconnectTime.store(now);
    
    running.store(true);
    status.store(CONNECTING);
    
    reconnectThread = std::thread(&Camera::reconnectLoop, this);
    grabThread = std::thread(&Camera::grabLoop, this);
    
    std::cout << "Camera started." << std::endl;
    return true;
}

void Camera::stop() {
    if (!running.load()) return;
    
    std::cout << "Stopping camera..." << std::endl;
    running.store(false);
    grabbing.store(false);
    needReconnect.store(true);
    
    {
        std::lock_guard<std::mutex> lock(reconnectMutex);
        reconnectCV.notify_all();
    }
    
    if (reconnectThread.joinable()) reconnectThread.join();
    if (grabThread.joinable()) grabThread.join();
    
    disconnectDevice();
    std::cout << "Camera stopped." << std::endl;
}

Camera::~Camera() { stop(); }


void Camera::reconnectLoop() {
    std::cout << "Reconnect loop started." << std::endl;
    bool first_try = true;
    
    while (running.load()) {
        CameraStatus currentStatus = status.load();
        
        if (currentStatus == GRABBING && !needReconnect.load()) {
            std::unique_lock<std::mutex> lock(reconnectMutex);
            reconnectCV.wait_for(lock, std::chrono::seconds(1));
            continue;
        }
            
        if (!first_try) {
            auto now = steady_clock::now();
            auto lastReconnect = lastReconnectTime.load();
            auto timeSinceLastReconnect = duration_cast<seconds>(now - lastReconnect);
            
            if (timeSinceLastReconnect.count() < 3 && !needReconnect.load()) {
                std::unique_lock<std::mutex> lock(reconnectMutex);
                auto waitTime = seconds(3) - timeSinceLastReconnect;
                reconnectCV.wait_for(lock, waitTime);
                continue;
            }
        }
    
        first_try = false;
        status.store(CONNECTING);
        std::cout << "Attempting to connect camera..." << std::endl;
        
        if (connectDevice()) {
            status.store(CONNECTED);
            needReconnect.store(false);
            std::cout << "Camera connected successfully." << std::endl;
            updateReconnectTime();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            grabbing.store(true);
        } else {
            status.store(DISCONNECTED);
            std::cout << "Failed to connect camera. Will retry in 3 seconds." << std::endl;
            updateReconnectTime();
            std::unique_lock<std::mutex> lock(reconnectMutex);
            reconnectCV.wait_for(lock, std::chrono::seconds(3));
        }
    }
    std::cout << "Reconnect loop stopped." << std::endl;
}


void Camera::grabLoop() {
    std::cout << "Grab loop started." << std::endl;
    
    while (running.load()) {
        while (running.load() && !grabbing.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!running.load()) break;
        
        if (handle == nullptr) {
            needReconnect.store(true);
            grabbing.store(false);
            continue;
        }
        
        int nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet) {
            std::cerr << "Start grabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            needReconnect.store(true);
            grabbing.store(false);
            continue;
        }
        
        status.store(GRABBING);
        std::cout << "Grabbing started." << std::endl;
        
        MVCC_INTVALUE stParam;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet) {
            std::cerr << "Get PayloadSize fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            needReconnect.store(true);
            grabbing.store(false);
            continue;
        }
        
        unsigned int nPayloadSize = stParam.nCurValue;
        unsigned char* pData = new unsigned char[nPayloadSize];
        if (pData == nullptr) {
            std::cerr << "Allocate memory fail!" << std::endl;
            needReconnect.store(true);
            grabbing.store(false);
            continue;
        }
        
        MV_FRAME_OUT_INFO_EX stImageInfo;
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        
        auto lastSuccessTime = steady_clock::now();
        
        while (running.load() && grabbing.load()) {
            nRet = MV_CC_GetOneFrameTimeout(handle, pData, nPayloadSize, &stImageInfo, 1000);
            
            if (nRet == MV_OK) {
                if (stImageInfo.nFrameLen > 0) {
                    cv::Mat processedImage;
                    if (processImage(pData, stImageInfo, processedImage)) {
                        if (isImageChanged(processedImage)) {
                            lastValidImage = processedImage.clone();
                            lastSuccessTime = steady_clock::now();
                        }
                        // 写入成员变量，始终覆盖并设置新帧标志。
                        // hasNewFrame_ 与时间戳在同一临界区内发布：getLatestFrame
                        // 全程持锁检查标志，写入新帧期间获取新帧会被阻塞，
                        // 杜绝"读到旧帧时间戳 / 半写状态"的竞态。
                        {
                            std::lock_guard<std::mutex> lock(frameMutex_);
                            latestFrame_ = processedImage.clone();
                            frameTimestamp_ = steady_clock::now();
                            hasNewFrame_ = true;
                        }
                    }
                }
            } else {
                auto now = steady_clock::now();
                auto timeSinceLastSuccess = duration_cast<seconds>(now - lastSuccessTime);
                if (timeSinceLastSuccess.count() >= 3) {
                    std::cout << "No valid image for 3 seconds, triggering reconnect." << std::endl;
                    needReconnect.store(true);
                    grabbing.store(false);
                    break;
                }
            }
            
            if (needReconnect.load()) {
                grabbing.store(false);
                break;
            }
        }
        
        if (handle != nullptr) {
            MV_CC_StopGrabbing(handle);
            std::cout << "Grabbing stopped." << std::endl;
        }
        
        delete[] pData;
        
        if (needReconnect.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    std::cout << "Grab loop stopped." << std::endl;
}

// ── 获取最新帧与时间戳 ──
bool Camera::getLatestFrame(cv::Mat& frame, std::chrono::steady_clock::time_point& timestamp) {
    // 全程持锁：写入新帧期间本调用阻塞等待，不返回"无新帧"误判；
    // 标志与时间戳在同一临界区发布/读取，见取流循环写入侧注释。
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (!hasNewFrame_) return false;
    latestFrame_.copyTo(frame);
    timestamp = frameTimestamp_;
    hasNewFrame_ = false;
    return true;
}


bool Camera::connectDevice() {
    int nRet = MV_CC_Initialize();
    if (MV_OK != nRet) {
        std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    return (cameraType == GIGE_CAMERA) ? tryConnectGigE() : tryConnectUSB();
}

bool Camera::tryConnectGigE() {
    int nRet = MV_OK;
    
    MV_CC_DEVICE_INFO stDevInfo;
    MV_GIGE_DEVICE_INFO stGigEDev;
    memset(&stDevInfo, 0, sizeof(MV_CC_DEVICE_INFO));
    memset(&stGigEDev, 0, sizeof(MV_GIGE_DEVICE_INFO));
    
    parseIp(deviceIp, stGigEDev.nCurrentIp);
    parseIp(netIp, stGigEDev.nNetExport);
    
    stDevInfo.nTLayerType = MV_GIGE_DEVICE;
    stDevInfo.SpecialInfo.stGigEInfo = stGigEDev;
    
    nRet = MV_CC_CreateHandle(&handle, &stDevInfo);
    if (MV_OK != nRet) {
        std::cerr << "Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cerr << "Open device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }
    
    int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
    if (nPacketSize > 0) {
        nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
        if (MV_OK != nRet)
            std::cerr << "Set Packet Size fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
    }
    
    if (!initCameraCommonParams()) {
        std::cerr << "Failed to initialize camera parameters!" << std::endl;
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }
    return true;
}

bool Camera::tryConnectUSB() {
    int nRet = MV_OK;
    
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        std::cerr << "Enum USB devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    if (stDeviceList.nDeviceNum == 0) {
        std::cerr << "No USB camera found!" << std::endl;
        return false;
    }
    
    if (deviceIndex >= static_cast<int>(stDeviceList.nDeviceNum)) {
        std::cerr << "Device index out of range! Found " << stDeviceList.nDeviceNum << " devices." << std::endl;
        return false;
    }
    
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[deviceIndex]);
    if (MV_OK != nRet) {
        std::cerr << "Create Handle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet) {
        std::cerr << "Open device fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }
    
    if (!initCameraCommonParams()) {
        std::cerr << "Failed to initialize camera parameters!" << std::endl;
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
        return false;
    }
    return true;
}

void Camera::disconnectDevice() {
    if (handle == nullptr) return;
    
    if (grabbing.load()) {
        MV_CC_StopGrabbing(handle);
        grabbing.store(false);
    }
    
    MV_CC_CloseDevice(handle);
    MV_CC_DestroyHandle(handle);
    handle = nullptr;
    MV_CC_Finalize();
    
    std::cout << "Device disconnected." << std::endl;
}

bool Camera::isImageChanged(const cv::Mat& newImage) {
    if (lastValidImage.empty() && newImage.empty()) return false;
    if (lastValidImage.empty() || newImage.empty()) return true;
    if (lastValidImage.size() != newImage.size()) return true;
    if (newImage.rows == 0 && newImage.cols == 0) return false;

    cv::Mat diff;
    cv::absdiff(lastValidImage, newImage, diff);
    cv::Scalar sumDiff = cv::sum(diff);
    for (int i = 0; i < sumDiff.channels; ++i) {
        if (sumDiff[i] != 0) return true;
    }
    return false;
}

void Camera::parseIp(const std::string& ip, unsigned int& parsedIp) {
    int parts[4];
    sscanf(ip.c_str(), "%d.%d.%d.%d", &parts[0], &parts[1], &parts[2], &parts[3]);
    parsedIp = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
}

bool Camera::setExposureTime(float exposureTime_) {
    exposureTime = exposureTime_;
    if (handle == nullptr) return false;
    
    int nRet = MV_CC_SetFloatValue(handle, "ExposureTime", exposureTime);
    if (MV_OK != nRet) {
        std::cerr << "Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Exposure time set to " << exposureTime << "us" << std::endl;
    return true;
}

bool Camera::setGain(float gain_) {
    gain = gain_;
    if (handle == nullptr) return false;
    
    int nRet = MV_CC_SetFloatValue(handle, "Gain", gain);
    if (MV_OK != nRet) {
        std::cerr << "Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    std::cout << "Gain set to " << gain << std::endl;
    return true;
}

bool Camera::initCameraParams() {
    if (handle == nullptr) return false;
    
    int nRet;
    nRet = MV_CC_SetFloatValue(handle, "ExposureTime", exposureTime);
    if (MV_OK != nRet) {
        std::cerr << "Set ExposureTime fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    nRet = MV_CC_SetFloatValue(handle, "Gain", gain);
    if (MV_OK != nRet) {
        std::cerr << "Set Gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    return true;
}

bool Camera::initCameraCommonParams() {
    if (handle == nullptr) return false;
    
    int nRet;
    nRet = MV_CC_SetEnumValue(handle, "ExposureAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto exposure fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    nRet = MV_CC_SetEnumValue(handle, "GainAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto gain fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    nRet = MV_CC_SetEnumValue(handle, "BalanceWhiteAuto", 0);
    if (MV_OK != nRet) {
        std::cerr << "Disable auto white balance fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return false;
    }
    
    if (!initCameraParams()) {
        std::cerr << "Failed to initialize camera exposure and gain parameters!" << std::endl;
        return false;
    }
    return true;
}



bool Camera::processImage(unsigned char* pData, MV_FRAME_OUT_INFO_EX& stImageInfo, cv::Mat& outputImage) {
    switch (stImageInfo.enPixelType) {
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerBG8: {
            cv::Mat img(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            cv::Mat bgrImg;
            
            int conversionCode;
            switch (stImageInfo.enPixelType) {
                case PixelType_Gvsp_BayerGB8: conversionCode = cv::COLOR_BayerGB2BGR; break;
                case PixelType_Gvsp_BayerRG8: conversionCode = cv::COLOR_BayerRG2BGR; break;
                case PixelType_Gvsp_BayerGR8: conversionCode = cv::COLOR_BayerGR2BGR; break;
                case PixelType_Gvsp_BayerBG8: conversionCode = cv::COLOR_BayerBG2BGR; break;
                default:                     conversionCode = cv::COLOR_BayerGB2BGR; break;
            }
            
            cv::cvtColor(img, bgrImg, conversionCode);
            
            std::vector<cv::Mat> channels(3);
            cv::split(bgrImg, channels);
            cv::Mat temp = channels[0];
            channels[0] = channels[2];
            channels[2] = temp;
            cv::merge(channels, bgrImg);
            
            outputImage = bgrImg;
            return true;
        }
        
        case PixelType_Gvsp_RGB8_Packed:
        case PixelType_Gvsp_BGR8_Packed: {
            cv::Mat img(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
            if (stImageInfo.enPixelType == PixelType_Gvsp_RGB8_Packed)
                cv::cvtColor(img, outputImage, cv::COLOR_RGB2BGR);
            else
                outputImage = img;
            return true;
        }
        
        case PixelType_Gvsp_Mono8:
            outputImage = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
            return true;
            
        default:
            std::cerr << "Unsupported pixel format: " << stImageInfo.enPixelType << std::endl;
            return false;
    }
}

std::vector<std::string> Camera::enumUSBDevices() {
    std::vector<std::string> deviceList;
    int nRet = MV_OK;
    
    static bool sdkInitialized = false;
    if (!sdkInitialized) {
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet) {
            std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return deviceList;
        }
        sdkInitialized = true;
    }
    
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet) {
        std::cerr << "Enum USB devices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        return deviceList;
    }
    
    for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++) {
        MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
        if (pDeviceInfo->nTLayerType == MV_USB_DEVICE) {
            std::string deviceName = reinterpret_cast<char*>(pDeviceInfo->SpecialInfo.stUsb3VInfo.chModelName);
            std::string serialNumber = reinterpret_cast<char*>(pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            std::string deviceInfo = "Device " + std::to_string(i) + ": " + deviceName + " (SN: " + serialNumber + ")";
            deviceList.push_back(deviceInfo);
        }
    }
    
    return deviceList;
}
