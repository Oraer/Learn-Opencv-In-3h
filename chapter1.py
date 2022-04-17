import cv2


# 图像的read和show，视频文件的读取，摄像头的调用

# print("Package Imported")

# img = cv2.imread("Resources/lena.png")
#
# cv2.imshow('Output', img)
# cv2.waitKey(0)

# cap = cv2.VideoCapture("./Resources/test_video.mp4")
cap = cv2.VideoCapture(0)
cap.set(3, 640)     # width id 3
cap.set(4, 480)     # height id 4
cap.set(10, 150)    # 亮度 id 10

while True:
    ret, img = cap.read()
    cv2.imshow("video", img)
    # & 0xFF的按位与操作只取cv2.waitKey(1)返回值最后八位,
    # 因为有些系统cv2.waitKey(1)的返回值不止八位 ord(‘q’)表示q的ASCII值
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
