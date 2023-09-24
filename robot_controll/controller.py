import socket
import time

class Robot :
    def __init__(self, ip) :
        self.ip = ip
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, 5678))
        self.previousHitTime = 0

    def hit(self) :
        if time.time() - self.previousHitTime < 0.1 :
            return
        self.previousHitTime = time.time()
        self.socket.sendall(b'hit')
    
    def move(self, y, z) :
        self.socket.sendall(("move " + str(y) + " " + str(z)).encode())