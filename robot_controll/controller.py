import socket
import time

class Robot :
    def __init__(self, ip, h = 5678) :
        self.ip = ip
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not self.ip == "" :
            self.socket.connect((self.ip, h))
        self.previousHitTime = 0

    def hit(self) :
        if self.ip == "" :
            return
        if time.time() - self.previousHitTime < 0.1 :
            return
        self.previousHitTime = time.time()
        self.socket.sendall(b'hit')
    
    def move(self, y, z) :
        if self.ip == "" :
            return
        self.socket.sendall(("m " + str(y) + " " + str(z)).encode())

ro = [
[1.041, 0.135],
[1.094, 0.133],
[1.241, 0.131],
[1.115, 0.130],
[0.985, 0.146],
[1.027, 0.140],
[0.912, 0.132],
[0.816, 0.142],
[0.786, 0.133],
[0.857, 0.135],
[0.886, 0.182],
[0.895, 0.189],
[0.889, 0.231],
[0.868, 0.224],
[0.837, 0.224],
[0.866, 0.213],
[0.852, 0.198],
[0.831, 0.192],
[0.834, 0.196],
[0.814, 0.186],
[0.801, 0.166],
[0.789, 0.158]
]
if __name__ == "__main__" :
    robot = Robot("192.168.137.221", 6678)
    for d in ro :
        robot.move(d[0]-0.6, d[1]+0.1)
        time.sleep(4/30)
    exit()
    robot = Robot("192.168.137.89")
    while True :
        a, b = input().split()
        robot.move(float(a), float(b))


