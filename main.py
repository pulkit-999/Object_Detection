import cv2
import time
import imutils

camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
time.sleep(1)

originalframe=None
area=500


while True:
    _,img=camera.read()
    text="Normal"
    img=imutils.resize(img,width=500)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg=cv2.GaussianBlur(grayImg,(21,21),0)
    if originalframe is None:
        originalframe=gaussianImg
        continue
    imgdif=cv2.absdiff(originalframe,gaussianImg)
    threshimg=cv2.threshold(imgdif,25,255,cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg,None,iterations=2)
    cnts = cv2.findContours(threshimg.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,201,0),3)
        text="MOVING OBJECT DETECTED"
        print(text)
        cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.imshow("Camera Feed",img)
        key=cv2.waitKey(1)&0xFF
        if key==ord("q"):
            break
camera.release()
cv2.destroyAllWindows()
