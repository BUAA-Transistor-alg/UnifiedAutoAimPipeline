import numpy as np
import cv2
import glob

# 棋盘格参数
chessboard_size = (11, 8) # (12, 8)  # 棋盘格角点数目
square_size = 0.03 # 2.35  # 棋盘格每个小方块的边长（米）

# 准备棋盘格角点坐标
objpoints = []  # 世界坐标系中的三维点
imgpoints = []  # 图像坐标系中的二维点

objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp[:, :2] *= square_size  # 将棋盘格坐标缩放到米

# 读取棋盘格图像并查找角点
images = glob.glob('chessboard/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像：{fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到，添加世界坐标系中的三维点和图像坐标系中的二维点
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制检测到的角点以便查看
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(1)  # 显示图像一段时间，单位为毫秒

    else:
        print(f"未能在图像 {fname} 中找到棋盘格角点。")

# 关闭所有窗口
cv2.destroyAllWindows()

# 校验是否找到足够的棋盘格图像
if len(objpoints) < 1:
    print("未找到足够的棋盘格图像用于标定。")
    exit()

# 打印找到的棋盘格图像数量
print(f"找到了 {len(objpoints)} 张棋盘格图像用于标定。")

# 标定相机
ret, K, dist_coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出内参和畸变系数
print("内参矩阵 K:\n", K)
print("畸变系数 dist_coef:\n", dist_coef)

print("以下输出可直接复制入config.yaml")

print(f"""camera_matrix: # 相机内参矩阵
  - [{K[0][0]:.8e}, {K[0][1]:.8e}, {K[0][2]:.8e}]
  - [{K[1][0]:.8e}, {K[1][1]:.8e}, {K[1][2]:.8e}]
  - [{K[2][0]:.8e}, {K[2][1]:.8e}, {K[2][2]:.8e}]
dist_coeffs: [{dist_coef[0][0]:.8f}, {dist_coef[0][1]:.8f}, {dist_coef[0][2]:.8f}, {dist_coef[0][3]:.8f}, {dist_coef[0][4]:.8f}] # 畸变系数""")
