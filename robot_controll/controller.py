import socket
import time
import math
import csv


_ang_data_rev = {}
_ang_data = {}
_key_rev = []
_key = []
_data_rows = 0
with open("servo_angle.csv") as f:
    reader = csv.reader(f)
    first = next(reader)
    for i in first :
        _ang_data_rev[i] = 0
        _key_rev.append(i)
    for row in reader :
        _data_rows += 1
        for i, d in enumerate(row):
            _ang_data_rev[_key_rev[i]] += float(d)
        
    for k in _key_rev :
        _ang_data_rev[k] /= _data_rows
    
    for k in _key_rev :
        _ang_data[_ang_data_rev[k]] = float(k)
        _key.append(_ang_data_rev[k])

def transRad2Deg(rad) :
    degCor = 90+rad*180/math.pi
    if degCor in _ang_data :
        return _ang_data[degCor]
    for i in range(len(_key)-1) :
        if _key[i] < degCor < _key[i+1] :
            return _ang_data[_key[i]] + (_ang_data[_key[i+1]]-_ang_data[_key[i]])/(_key[i+1]-_key[i])*(degCor-_key[i])
    print("out of range")

class Robot :
    def __init__(self, ip, h = 5678) :
        self.ip = ip
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not self.ip == "" :
            self.socket.connect((self.ip, h))
            pass
        self.previousHitTime = 0
        self.arm_length = 0.465
        self.arm_height = 0.46
        self.now_rad = 0.5 * math.pi

    def stop(self) :
        self.socket.sendall(b"q")

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
        if abs((z-self.arm_height)/self.arm_length) > 1 :
            print("out of range")
            return
        rad1 = math.asin((z-self.arm_height)/self.arm_length)
        rad2 = math.pi - rad1

        arm_borad1 = math.cos(rad1) * self.arm_length
        arm_borad2 = -arm_borad1

        base_pos1 = round((y-arm_borad1)/0.0008)
        base_pos2 = round((y-arm_borad2)/0.0008)

        if -756 < base_pos1 < 756 and -0.25 * math.pi < rad1 < 1.25*math.pi and -756 < base_pos2 < 756 and -0.25*math.pi < rad2 < 1.25*math.pi :
            if abs(rad1 - self.now_rad) < abs(rad2 - self.now_rad) :
                rad = rad1
                base_pos = base_pos1
            else :
                rad = rad2
                base_pos = base_pos2
        elif -756 < base_pos1 < 756 and -0.25*math.pi < rad1 < 1.25*math.pi :
            rad = rad1
            base_pos = base_pos1
        elif -756 < base_pos2 < 756 and -0.25*math.pi < rad2 < 1.25*math.pi :
            rad = rad2
            base_pos = base_pos2
        else :
            print(rad1, base_pos1, arm_borad1, rad2, base_pos2)
            print("not reachable")
            return
                        
        deg = transRad2Deg(rad)
        print(rad1, base_pos1, arm_borad1, rad2, base_pos2)
        print("move to", deg, base_pos)
        self.now_rad = rad
        self.socket.sendall(("m " + str(deg) + " " + str(base_pos)).encode())

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
    robot = Robot("192.168.137.254", 5678)
    for d in ro :
        robot.move(d[0]-0.6, d[1]+0.1)
        time.sleep(4/30)
    robot.stop()
    exit()
    robot = Robot("192.168.137.89")
    while True :
        a, b = input().split()
        robot.move(float(a), float(b))


