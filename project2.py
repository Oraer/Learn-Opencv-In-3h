import cv2
import numpy as np

##########################
imgWidth = 480
imgHeight = 640
##########################


cap = cv2.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, 150)


def preProcessing(img):
    """
    对图像进行预处理
    :param img:
    :return:
    """

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDila = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDila, kernel, iterations=1)  # imgErode

    return imgThres


def getContours(img):
    """
    对预处理后的图像，获取文档轮廓
    :param img:
    :return:
    """

    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # 图片，检索方法，cv2.RETR_EXTERNAL 检索极端外部轮廓, 不需要近似
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # 忽略噪声影响 形状面积大于500
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)  # contourIdx=-1 代表全部contour
            # 计算曲线长度，长度可以帮助我们得到近似边缘的拐角
            # perimeter 周长
            peri = cv2.arcLength(cnt, True)  # curve closed 边缘 True
            # print(peri)
            # 近似拐角点  进行多边形逼近，得到多边形的角点
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 拐角点
            # print(len(approx))              # 拐角点个数 len(approx)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    # print(biggest)
    return biggest


def reOrder(myPoints):
    """
    对得到的四个角点，进行重排序，[[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]]
    :param myPoints: 四个角点
    :return:
    """

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)

    # 求和，得到最小和最大的
    add = myPoints.sum(axis=1)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    # 相减计算差值 ，后减前
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew
    # print("myPointNew", myPointsNew)


def getWrap(img, biggest):
    """
    利用得到的角点，进行透视变换
    :param img:
    :param biggest:
    :return:
    """

    biggest = reOrder(biggest)
    # 简单理解，这地方返回的四个点，不是按照顺序返回的，需要重定位，知道那个在左右，那个在上下
    pts1 = np.float32(biggest)
    # [0,0]加起来是四个中最小的，[width,height]加起来是四个中最大的，[width,0]相减（后减前）是负数，[0,height]相减是正数
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 使用getPerspectiveTransform()得到转换矩阵
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))  # 使用warpPerspective()进行透视变换

    # 因为结果图还是存在边框多余，裁剪结果图，去掉四边10个像素，
    imgCropped = imgOutput[20:imgOutput.shape[0] - 10, 20:imgOutput.shape[1] - 10]
    imgCropped = cv2.resize(imgCropped, (imgWidth, imgHeight))
    return imgCropped


# 图片的堆叠，合并
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    # & 输出一个 rows * cols 的矩阵（imgArray）
    # print(rows,cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[][] 是什么意思呢？
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    # & 而shape[1]就是width，shape[0]是height，shape[2]是
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    # & 例如，我们可以展示一下是什么含义
    # cv2.imshow("img", imgArray[0][1])

    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


while True:
    # success, img = cap.read()
    img = cv2.imread("./Resources/1.jpg")
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgContour = img.copy()
    imgThres = preProcessing(img)        # 预处理
    biggest = getContours(imgThres)      # 获取contours轮廓，四个角点
    # print(biggest)

    # 堆叠图片
    # 如果摄像头么欸有检测到具有四个角的文档，则不会对图片进行透视变换处理
    if biggest.size != 0:
        # 透视变换
        imgWrapped = getWrap(img, biggest)
        imgArray = ([img, imgThres],
                    [imgContour, imgWrapped])
    else:
        imgArray = ([img, imgThres],
                    [img, img])
    imgStack = stackImages(0.6, imgArray)

    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
