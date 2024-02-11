import socket
import struct
import time
import math
import csv


_ang_data_rev = {}
_ang_data = {}
_key_rev = []
_key = []
_data_rows = 0
with open("robot_control/servo_angle.csv") as f:
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
        self.arm_length = 0.48
        self.arm_height = 0.46
        self.now_rad = 0.5 * math.pi

    def stop(self) :
        if self.ip == "" :
            return
        self.socket.sendall(b"q    ")

    def hit(self) :
        if self.ip == "" :
            return
        if time.time() - self.previousHitTime < 0.1 :
            return
        self.previousHitTime = time.time()
        self.socket.sendall(b'h    ')

    def testCommunicateTime(self) :
        if self.ip == "" :
            return
        start = time.time()
        self.socket.sendall(struct.pack("!f", start))
        recv = self.socket.recv(1024)
        end = time.time()
        print("time:", end-start)
        print("time diff:", struct.unpack("!f", recv)[0]-(start))
    
    def move(self, y, z) :
        if abs((z-self.arm_height)/self.arm_length) > 1 :
            print("out of range")
            return None, None
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
            print(rad1, rad2, base_pos1, base_pos2)
            return None, None
                        
        deg = transRad2Deg(rad)
        value = int(500 + (deg*2000)/270)
        #print(rad1, base_pos1, arm_borad1, rad2, base_pos2)
        print("move to", deg, base_pos)
        self.now_rad = rad
        buf = b"m"
        buf += value.to_bytes(2, byteorder="big")
        base_pos += 1000
        buf += base_pos.to_bytes(2, byteorder="big")
        
        if self.ip == "" :
            return deg, base_pos-1000
        self.socket.sendall(buf)
        return deg, base_pos-1000

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
    robot = Robot("10.42.0.82", 6678)
    #robot = Robot("")
    #robot.testCommunicateTime()

    while True :
        inp = input()
        if inp == "h" :
            robot.hit()
        else :
            a, b = inp.split()
            print(robot.move(float(a), float(b)))

    exit()
    for d in ro :
        robot.move(d[0], d[1])
        time.sleep(1/300)
    robot.stop()
    exit()
    robot = Robot("192.168.137.89")
    while True :
        a, b = input().split()
        robot.move(float(a), float(b))


