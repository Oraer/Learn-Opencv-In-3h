import cv2
import numpy as np

# 在图像上绘制形状 画线 矩形 圆 ，并且在图像上放置文本

img = np.zeros((512, 512, 3), dtype=np.int8)
# print(img)
# img[:] = 255,0,0

# 画出直线
# cv2.line(img, (0,0), (480,480), (0,255,0), 3)
cv2.line(img, (0,0), (img.shape[1],img.shape[0]), (0,255,0), 3)  # numpy先高后宽，cv2中先宽后高

# 画出矩形
cv2.rectangle(img, (0,0), (100,100), (0,0,255), 2)

# 画出圆形
cv2.circle(img, (400,50), 30, (255,0,0), 5, cv2.FILLED)   # cv2.FILLED 填充图形

# 放置文本
cv2.putText(img, "opencv", (300,200), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,150,0),1)

cv2.imshow("img", img)
cv2.waitKey(0)
# cv2.destroyAllWindows()