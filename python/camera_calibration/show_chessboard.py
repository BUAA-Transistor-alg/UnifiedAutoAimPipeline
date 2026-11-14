import cv2
import numpy as np

chessboard_img = np.zeros((1080,1920,3))
chessboard_img[:] = 255
square_side_length = 100
margin_x = 200
margin_y = 50

for y_square in range(9):
    for x_square in range(13):
        if (y_square + x_square) % 2 == 0:
            chessboard_img[margin_y+y_square*square_side_length : margin_y+(y_square+1)*square_side_length,
                           margin_x+x_square*square_side_length : margin_x+(x_square+1)*square_side_length,
                           :] = 0

cv2.imshow("", chessboard_img)
cv2.waitKey(0)