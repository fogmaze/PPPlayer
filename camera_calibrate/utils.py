from pupil_apriltags import Detector, Detection
import cv2
import time
import pickle



def takePicture():
    cap = cv2.VideoCapture(0)
    a = False
    while True :
        ret, frame = cap.read()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) & 0xff

            if a :
                cv2.imwrite('B.jpg', frame)
                break

            if key == ord('w') :
                cameraMatrix = pickle.load(open('calibration1', 'rb'))
                print(calculateCameraPosition(cameraMatrix, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True

def calculateCameraPosition(cameraMatrix, frame, tagSize=12.9) :
    detector = Detector()
    results = detector.detect(frame, estimate_tag_pose=True, camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]), tag_size=tagSize)
    if len(results) == 1:
        res:Detection = results[0]
        return res.pose_t[0][0], res.pose_t[1][0], res.pose_t[2][0]
    else:
        return None

def getCameraPosition_realTime(cameraMatrix) :
    cap = cv2.VideoCapture(0)
    a = False
    while True :
        ret, frame = cap.read()
        time.sleep(1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = calculateCameraPosition(cameraMatrix, image)

        if res is not None :
            print(res)
        

def getBallPixelSize(distance, cameraMatrix) :
    BALL_REAL_SIZE = 3.8
    return BALL_REAL_SIZE * cameraMatrix[0][0] / distance

#print(getBallPixelSize(100), getBallPixelSize(450))

if __name__ == "__main__" :
    cameraMatrix = pickle.load(open('calibration1', 'rb'))
    print(getBallPixelSize(500, cameraMatrix), getBallPixelSize(100, cameraMatrix))
    #calculateCameraPosition(cameraMatrix, cv2.imread('test.jpg'))
    #print(cameraMatrix)
    #getCameraPosition_realTime(cameraMatrix)
    pass