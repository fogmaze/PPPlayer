import socket
import cv2
import numpy as np

class CameraReceiver:
    def __init__(self, ip):
        self.ip = ip
        self.port = 7439
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, self.port))
        
    def read(self) :
        img_str = b''
        size = self.socket.recv(4)
        size = int.from_bytes(size, 'big')
        img_str = self.socket.recv(size)
        exact_size = len(img_str)
        # if not start with b'\xff\xd8' continue
        if img_str[:2] != b'\xff\xd8':
            print('not start with b\'\\xff\\xd8\'')
            while True :
                data = self.socket.recv(1024)
                img_str += data
                if len(data) < 1024 :
                    break
            return
        print('size : ', size, len(img_str))
        while exact_size < size :
            data = self.socket.recv(1024)
            img_str += data
            exact_size += len(data)
        nparr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        

    def close(self):
        self.socket.close()

if __name__ == "__main__" :
    receiver = CameraReceiver('192.168.66.30')
    while True :
        receiver.read()