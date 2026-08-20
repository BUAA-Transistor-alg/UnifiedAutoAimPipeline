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

# 帧平移量自动搜索（视频帧 k 对应文本行 k-shift），默认小数精度 0.1 帧；
# 可用 --precision 调整小数搜索精度（如 0.05）
python3 python/analyze_record.py <record_dir> --precision 0.1

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
- 帧平移量自动搜索：不再手动指定 `--shift`。在
  `|shift| <= min(视频帧数, 文本行数)/2` 范围内先整数粗搜，再在最佳整数附近按
  `--precision`（默认 0.1）细化，目标为使「成功解析帧上的误差 MSE」（先做与绘图一致的
  均值对齐）最小；小数 shift 用线性插值采样 IMU yaw 曲线（文本位置 k-shift），
  并抛弃插值窗口越出文本数据范围的端点帧。
- 平均 dt：取最终公共段（计算范围）对应文本行内所有帧 dt 的均值；
  结果图标题与汇总输出同时显示最优 shift 与 `shift × 平均 dt`（时间量，单位秒）。
- `imu_euler_yaw`（frame_info 第 5 列）先解缠绕，再整体平移一个常数，
  使「视频成功解析的点」上两条曲线的平均值相等，然后绘制。
- 误差曲线 = 平移后 imu yaw − 棋盘格 yaw。
