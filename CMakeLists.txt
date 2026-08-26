cmake_minimum_required(VERSION 3.10)
project(UnifiedAutoAim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 将可执行文件输出到项目根目录的 bin/ 文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ── backward-cpp 堆栈回溯 (需要调试符号 + dynamic 导出) ──
add_compile_options(-g)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# 定义项目根目录（供 PathResolver 使用）
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# ── 查找依赖 ──
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# yaml-cpp（config/config.yaml 解析）
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# FFmpeg（录制：MkvWriter 视频编码）
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

# 查找 udev（串口通信需要）
find_library(UDEV_LIBRARY udev)
if(NOT UDEV_LIBRARY)
    message(FATAL_ERROR "libudev not found")
endif()

# 查找海康 MVS SDK
find_library(MVCAMERA_LIB MvCameraControl PATHS /opt/MVS/lib/64 /usr/lib)

# ── backward-cpp: 查找 libdw (更好的 DWARF 堆栈解析) ──
find_library(DW_LIBRARY dw)
if(DW_LIBRARY)
    message(STATUS "found libdw: ${DW_LIBRARY}")
endif()

# 设置包含目录（头文件统一使用 include/ 根目录下的相对路径）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    /opt/MVS/include
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

# ── TorqueController 子模组：通信 + 融合 + MPC 一体化封装（RobotController）──
# 仅链接其 RobotController 动态库（robot_comm_c），不构建子模组内的演示程序
add_subdirectory(sub_module/TorqueController ${CMAKE_BINARY_DIR}/torque_controller EXCLUDE_FROM_ALL)

# ── 公共源文件（通用类：基础设施 / 输入模式 / tf 树 / 弹道 / 输出模式）──
set(COMMON_SOURCES
    src/common/PathResolver.cpp
    src/common/TaskPool.cpp
    src/common/FrameRateCounter.cpp
    src/common/RobotConfig.cpp
    src/common/BacklogAdaptiveDelay.cpp
    src/common/CameraProjection.cpp
    src/common/Infer/InferCore.cpp
    src/common/Infer/InferShmClient.cpp
    src/common/Infer/InferShmServer.cpp
    src/common/Infer/InferProcessManager.cpp
    src/common/Output/VisualizeOutput.cpp
    src/common/Output/GimbalOutput.cpp
    src/common/TransformTree/CoordinateTransform.cpp
    src/common/TransformTree/TransformNode.cpp
    src/common/TransformTree/TransformTreeManager.cpp
    src/common/TransformTree/RobotTfTree.cpp
    src/common/Ballistic/BallisticSolver.cpp
    src/common/Ballistic/GimbalSolver.cpp
    src/common/Ballistic/PredictedBallisticSolver.cpp
    src/common/Ballistic/SequencePredictor.cpp
    src/common/camera/Camera.cpp
    src/common/Input/CameraInputMode.cpp
    src/common/Input/VideoInputMode.cpp
    src/common/Input/InteractiveInputMode.cpp
    src/common/Record/MkvWriter.cpp
    src/common/Record/FrameRecorder.cpp
)

# ── Outpost 源文件 ──
set(OUTPOST_SOURCES
    src/Outpost/ESEKF.cpp
    src/Outpost/OpenvinoInfer.cpp
    src/Outpost/OutpostVisualizer.cpp
    src/Outpost/OutpostPipeline.cpp
)

# ── PowerRune 源文件 ──
set(POWER_RUNE_SOURCES
    src/PowerRune/YAxisFilter.cpp
    src/PowerRune/RollPredictor.cpp
    src/PowerRune/TargetPositionCalculator.cpp
    src/PowerRune/PowerRunePoseSolver.cpp
    src/PowerRune/YoloPoseInfer.cpp
    src/PowerRune/PowerRuneVisualizer.cpp
    src/PowerRune/PowerRunePipeline.cpp
)

# ── 主程序 ──
add_executable(unified_auto_aim
    src/main.cpp
    ${COMMON_SOURCES}
    ${OUTPOST_SOURCES}
    ${POWER_RUNE_SOURCES}
)

target_link_libraries(unified_auto_aim PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
    robot_comm_c          # TorqueController 子模组（RobotController 一体化封装）
    ${UDEV_LIBRARY}
    Threads::Threads
    ${MVCAMERA_LIB}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${CMAKE_DL_LIBS}       # libdl (backward-cpp 需要 dladdr)
)

# ── backward-cpp: 可选 libdw 链接 ──
if(DW_LIBRARY)
    target_link_libraries(unified_auto_aim PRIVATE ${DW_LIBRARY})
    target_compile_definitions(unified_auto_aim PRIVATE BACKWARD_HAS_DW=1)
endif()

# ── 推理进程（共享内存 + 信号量通信；各进程只编译一个模型，见 InferShm.h）──
# 启动方式（先于主程序）：
#   ./bin/outpost_infer_process    （读 config outpost.inference 段 + shm_key）
#   ./bin/power_rune_infer_process （读 config power_rune.inference 段 + shm_key）
set(INFER_PROCESS_LIBS
    ${OpenCV_LIBS}
    openvino::runtime
    Threads::Threads
    ${YAMLCPP_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

add_executable(outpost_infer_process
    src/Infer/outpost_infer_process_main.cpp
    src/common/RobotConfig.cpp
    src/common/PathResolver.cpp
    src/common/TaskPool.cpp
    src/common/Infer/InferCore.cpp
    src/common/Infer/InferShmClient.cpp
    src/common/Infer/InferShmServer.cpp
    src/Outpost/OpenvinoInfer.cpp
)
target_link_libraries(outpost_infer_process PRIVATE ${INFER_PROCESS_LIBS})

add_executable(power_rune_infer_process
    src/Infer/power_rune_infer_process_main.cpp
    src/common/RobotConfig.cpp
    src/common/PathResolver.cpp
    src/common/TaskPool.cpp
    src/common/Infer/InferCore.cpp
    src/common/Infer/InferShmClient.cpp
    src/common/Infer/InferShmServer.cpp
    src/PowerRune/YoloPoseInfer.cpp
)
target_link_libraries(power_rune_infer_process PRIVATE ${INFER_PROCESS_LIBS})
