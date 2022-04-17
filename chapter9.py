import cv2

# 检测人脸

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
# img = cv2.imread('Resources/lena.png')

# 调用摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)     # width id 3
cap.set(4, 480)     # height id 4
cap.set(10, 150)    # 亮度 id 10
while True:
    ret, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)  # scalefactor 比例因子 minNeighbors 最小邻居

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
