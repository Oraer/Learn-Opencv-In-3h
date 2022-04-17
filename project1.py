import cv2
import numpy as np


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)     # width id 3
cap.set(4, frameHeight)     # height id 4
cap.set(10, 150)    # 亮度 id 10

# hsv阈值获取不理想
myColors = [
           [0, 155, 120, 88, 255, 250]]

mycolorValues = [[128, 0, 0],          # BGR
                 [0, 0, 255]]

myPoints = []    # x, y, colorId

def findColor(img, myColors, mycolorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    index = 0         # count
    newPiontsList = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        cv2.imshow(str(color[0]), mask)
        xx, yy = getContours(mask)
        cv2.circle(imgResult, (xx, yy), 10, mycolorValues[index], cv2.FILLED)
        if xx!=0 and yy!=0:
            newPiontsList.append([xx, yy, index])
        index += 1
        return newPiontsList

def getContours(img):
    x, y, w, h = 0, 0, 0, 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 图片，检索方法，cv2.RETR_EXTERNAL 检索极端外部轮廓, 不需要近似
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:           # 忽略噪声影响 形状面积大于500
            # 画出轮廓
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 5)  # contourIdx=-1 代表全部contour
            # 计算曲线长度，长度可以帮助我们得到近似边缘的拐角
            # perimeter 周长
            peri = cv2.arcLength(cnt, True)   # curve closed 边缘 True
            # 近似拐角点  进行多边形逼近，得到多边形的角点
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)   # 拐角点

            x, y, w, h = cv2.boundingRect(approx)            # 获取图形的宽和高以及起始点的坐标
    return x+w//2, y


def drawOnCanvas(myPoints, mycolorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, mycolorValues[point[2]], cv2.FILLED)




while True:
    ret, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, mycolorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, mycolorValues)
    cv2.imshow("Result", imgResult)
    # & 0xFF的按位与操作只取cv2.waitKey(1)返回值最后八位,
    # 因为有些系统cv2.waitKey(1)的返回值不止八位 ord(‘q’)表示q的ASCII值
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
