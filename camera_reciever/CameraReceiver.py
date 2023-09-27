import time
import socket
import cv2
import numpy as np

class CameraReceiver:
    def __init__(self, ip):
        self.ip = ip
        self.port = 7439
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.socket.connect((self.ip, self.port))
        
    def connect(self) :
        self.socket.connect((self.ip, self.port))
        
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
            return self.read()
        while exact_size < size :
            data = self.socket.recv(1024)
            img_str += data
            exact_size += len(data)
        nparr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return True, img
        
    def __del__(self):
        self.close()

    def close(self):
        self.socket.close()

if __name__ == "__main__" :
    receiver = CameraReceiver('192.168.66.30')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))
    while True :
        img = receiver.read()
        if img is False :
            continue
        cv2.imshow('img', img)
        writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    receiver.close()
    writer.release()