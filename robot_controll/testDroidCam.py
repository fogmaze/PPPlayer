import cv2
import time

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter("time.mp4", fourcc, 30, (640, 480))
times = []

iteration = 0
while True:
    t = time.time()
    print(t)
    times.append(t)
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.putText(frame, ("{}".format(iteration)), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    out_video.write(frame)
    iteration += 1
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

with open("times.txt", "w") as f:
    for t in times:
        f.write(str(iteration) + " " + str(t)+"\n")
