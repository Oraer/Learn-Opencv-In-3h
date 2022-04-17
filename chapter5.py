import cv2
import numpy as np


# 空间变换，
img = cv2.imread("Resources/cards.jpg")

width, height = 250, 350          # 纸牌比例是2.5:3.5
pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)   # 使用getPerspectiveTransform()得到转换矩阵
imgOutput = cv2.warpPerspective(img, matrix, (width, height)) #使用warpPerspective()进行透视变换

cv2.imshow("Img", img)
cv2.imshow("Output", imgOutput)
cv2.waitKey(0)