import cv2

##############################################
imgWidth = 640
imgHeight = 480
minArea = 500
myColor = (255, 0, 0)
nplateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
##############################################

# cap = cv2.VideoCapture(0)
# cap.set(3, imgWidth)     # width id 3
# cap.set(4, imgHeight)     # height id 4
# cap.set(10, 150)    # 亮度 id 10
count = 0



while True:
    # ret, img = cap.read()
    # cv2.imshow("video", img)
    img = cv2.imread("Resources/p1.jpg")



    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = nplateCascade.detectMultiScale(imgGray, 1.1, 4)  # scalefactor 比例因子 minNeighbors 最小邻居

    for (x,y,w,h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plates", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, myColor, 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)




    cv2.imshow("Result", img)
    # & 0xFF的按位与操作只取cv2.waitKey(1)返回值最后八位,
    # 因为有些系统cv2.waitKey(1)的返回值不止八位 ord(‘q’)表示q的ASCII值
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("./myScanned/NoPlate_" + str(count) + ".jpg", imgRoi)
        print("./myScanned/NoPlate_" + str(count) + ".jpg" + "  " + "saved")
        cv2.rectangle(img, (0, img.shape[0]//2 - 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scanned Save", (150, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
