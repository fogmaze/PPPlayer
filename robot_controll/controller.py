import socket
import time

class Robot :
    def __init__(self, ip) :
        self.ip = ip
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not self.ip == "" :
            self.socket.connect((self.ip, 5678))
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

if __name__ == "__main__" :
    robot = Robot("172.20.20.228")
    robot.move(0.5, 0.5 )
    time.sleep(1)
    robot.move(-0.5, 0.5 )


