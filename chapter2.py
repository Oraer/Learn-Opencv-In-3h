import cv2
import numpy as np

# 简单的入门体验，
# 灰度图 高斯模糊 canny边缘检测 膨胀 腐蚀 操作

img = cv2.imread("Resources/lena.png")
kernel = np.ones((5,5), dtype=np.uint8)

# 灰度图
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯模糊
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)   # kernal 必须为奇数3*3 5*5 7*7
# canny边缘检测
imgCanny = cv2.Canny(img, 150, 200)
# 膨胀
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)   # 迭代次数决定了厚度
# 腐蚀
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Dilation", imgDialation)
cv2.imshow("Erosion", imgEroded)
cv2.waitKey(0)
