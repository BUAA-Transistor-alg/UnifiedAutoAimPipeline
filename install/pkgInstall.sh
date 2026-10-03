#!/bin/sh

# 出错时立即退出，避免错误累积
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 临时目录
TEMP_DIR="$SCRIPT_DIR/pkg_temp"
# 确保目录存在
mkdir -p "$TEMP_DIR"

# --- 1. 下载并安装 GPG 密钥 (使用新方法) ---
KEY_URL="https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
KEY_FILE="$TEMP_DIR/intel-gpg-key.pub"
# 将密钥去armor后保存到 trusted.gpg.d 目录
wget -O "$KEY_FILE" "$KEY_URL"
sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/intel.gpg "$KEY_FILE"
# 可选：删除临时密钥文件
rm -f "$KEY_FILE"

# --- 2. 添加 APT 仓库源 ---
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list

# --- 3. 更新并安装 OpenVINO 及依赖 ---
sudo apt update

# 安装 OpenVINO 指定版本
sudo apt install -y openvino-2024.0.0

# 安装开发依赖
sudo apt install -y libopencv-dev libeigen3-dev libceres-dev libyaml-cpp-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libudev-dev pkg-config libdw-dev build-essential

# --- 4. 检测是否为 Intel 机器，若是则安装 Intel GPU 驱动 ---
if grep -q "GenuineIntel" /proc/cpuinfo; then
    echo "检测到 Intel 平台，开始安装 Intel GPU 驱动..."

    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
        sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

    sudo apt update

    sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo
    sudo apt-get install -y libze-dev intel-ocloc

    echo "Intel GPU 驱动安装完成。"
else
    echo "未检测到 Intel 平台，跳过 Intel GPU 驱动安装。"
fi

# --- 5. Python 依赖安装 ---
pip install ultralytics==8.4.60 numpy opencv-python matplotlib pyyaml \
    onnx onnxslim onnxruntime openvino

echo "全部安装完成。"