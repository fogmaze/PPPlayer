import cv2

cap = cv2.VideoCapture(1)
while True :
    # show 
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(10) & 0xff
    if key == ord('q') :
        break
    elif key == ord('w') :
        cv2.imwrite('B.jpg', frame)
        break
cap.release()
cv2.destroyAllWindows()
