# python/ — Record 模式数据分析脚本

## 文件

- `analyze_record.py` — 解析 Record 模式录制的视频 + frame_info txt，
  对比棋盘格 PnP 欧拉角 yaw 与录制的 imu_euler_yaw，绘制同一张图上的两条曲线
  及误差曲线（结果图默认保存为 `<record_dir>/checkerboard_imu_yaw_comparison.png`）。
- `path_resolver.py` — `include/common/PathResolver.h` 的 Python 翻译版
  （独立文件，`analyze_record.py` 通过它定位项目根目录与 config/config.yaml）。

## 用法

```bash
# <record_dir> 为 Record 模式会话目录（record_YYYYmmdd_HHMMSS，内含 video_*.mkv 与 frame_info_*.txt）
# 默认使用 common.input_mode.camera_mode 的相机参数（实机相机录制）
python3 python/analyze_record.py <record_dir>

# 换用 video_mode 相机参数（无畸变标定 / 模拟相机录制）
python3 python/analyze_record.py <record_dir> --mode video

# 文本数据平移 3 帧（视频帧 k 对应文本行 k-3）；负数同样允许（如 --shift -2 → k+2）
python3 python/analyze_record.py <record_dir> --shift 3

# 额外交互显示 + 自定义输出路径
python3 python/analyze_record.py <record_dir> --show --output /path/to/out.png
```

## 行为说明

- 棋盘格参数与 `other_files/camera_calibration/dist.py` 相同：
  角点数 `(11, 8)`、方块边长 `0.03`；每帧尝试 `findChessboardCorners`。
- 成功取到角点后用 `solvePnP` 解算位姿，并按
  `CameraProjection::pnpRvecToEuler` 相同的约定取 cam 系欧拉角
  （`yaw = atan2(-R02, R22)`、`pitch = asin(-R12)`、`roll = atan2(R10, R11)`）。
- 未成功解析的帧使用前一帧数据（再往前推；直到第一帧都无有效值则为 0）。
- `--shift N`：文本数据帧平移量，视频帧 k 对应文本行 k-N（N 可为负数，默认 0），
  再以相同方法（min 公共长度）计算公共段；默认 0。
- `imu_euler_yaw`（frame_info 第 5 列）先解缠绕，再整体平移一个常数，
  使「视频成功解析的点」上两条曲线的平均值相等，然后绘制。
- 误差曲线 = 平移后 imu yaw − 棋盘格 yaw。
