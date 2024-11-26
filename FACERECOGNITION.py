import numpy as np
import cv2
#from djitellopy import tello

# import time

#me = tello.Tello()
#me.connect()
#print(me.get_battery())

#me.streamon()
#me.takeoff()
#me.send_rc_control(0,0,25,0)
#time.sleep(2.2)

w, h = 360,240
fbRange =[6200,6800]
pid = [0.4,0.4,0]
pError = 0


def findFace(img):
    # Load the face detection model
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    if faceCascade.empty():
        print("Error loading Haar cascade file.")
        return img  # Return the original image if cascade file is not loaded

    # Convert the image to grayscale for face detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangles on detected faces
        cx = x + w // 2 #Get Centre Points
        cy = y + h // 2 #Get Centre Points
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx,cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) !=0:
        i= myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0,0],0]

def trackFace(info, w, pid, pError):
    area = info[1]
    x,y = info[0]
    error = x - w//2
    fb = 0
    speed = pid[0]*error +pid[1]*(error-pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb=-20
    elif area < fbRange[0] and area !=0:
        fb = 20

    print(speed, fb)

    if x ==0:
        speed = 0
        error = 0
    #me.send_rc_control(0, fb, 0, speed)
    return error


# Initialize video capture (use 0 for the default camera)
cap = cv2.VideoCapture(0)  # Use 0 if there's only one camera

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
       print("Failed to capture frame.")
       break
    #img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    pError = trackFace( info, w, pid, pError)
    print("Centre" , info[0], "Area", info[1])# Process the frame to detect faces
    cv2.imshow("Output", img)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #me.land()
        break

