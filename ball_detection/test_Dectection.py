import cv2

def main():
    pass

def startDetect(source):
    cap = cv2.VideoCapture(source)
    while True:
        ret, img = cap.read()
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detectOneFrame(veryFirstFrame, frame):
    pass

if __name__ == "__main__":
    main()