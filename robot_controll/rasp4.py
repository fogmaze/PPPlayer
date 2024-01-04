import RPi.GPIO as gpio
import struct
import select
import time
import multiprocessing as mp
import threading
import socket
import math
import serial
import csv

gpio.setmode(gpio.BOARD)
gpio.setwarnings(False)

LogData = []

def _stepper_process(pipe: mp.Pipe, DIR=38, ENA=36, PUL=40, acceleration=5, speed=1200) :
    gpio.setup(PUL, gpio.OUT)
    gpio.setup(ENA, gpio.OUT)
    gpio.setup(DIR, gpio.OUT)
    gpio.output(ENA, gpio.LOW)
    
    ALL_LENGTH = 1512/4
    recent_position = -756
    destination = 0
    isRunning = True
    stepper_log = []

    velocity = 0
    while True :
        if pipe.poll():
            destination = pipe.recv()
            stepper_log.append([time.time(), "goto", destination])
            print(destination)
            if destination == "stop" :
                break
                print("stop received")
        if destination == recent_position :
            gpio.output(ENA, gpio.HIGH)
            stepper_log.append([time.time(), "st", recent_position])
            continue
        gpio.output(ENA, gpio.LOW)
            
        #whether to accelerate or decelerate
        distance = destination - recent_position
            
        if velocity == 0 :
            if distance > 0 :
                velocity += acceleration
            else :
                velocity -= acceleration  
        elif velocity < 0 :
            if distance < 0 :
                if abs(velocity/acceleration) >= abs(distance) :
                    velocity += acceleration
                elif abs(velocity) < speed :
                    velocity -= acceleration    
            else :
                velocity += acceleration 
        elif velocity > 0 :
            if distance > 0 :
                if abs(velocity/acceleration) >= abs(distance) :
                    velocity -= acceleration
                elif velocity < speed :
                    velocity += acceleration
            else :
                velocity -= acceleration
                    
        if velocity == 0 :
            continue
        elif velocity > 0 :
            gpio.output(DIR, gpio.HIGH)
            recent_position += 1
        elif velocity < 0 :
            gpio.output(DIR, gpio.LOW)
            recent_position -= 1
        pulse_time = 1/abs(velocity)
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
        stepper_log.append((time.time(), "now at", recent_position))
    print("pipe stopped")
    with open("log_stepper", "w") as f :
        for l in LogData:
            f.write(str(l) + "\n")
        


class Stepper() :
    def __init__(self, DIR=38, ENA=40, PUL=36, acceleration=4, speed=1000) :
        self.pipe, pipe = mp.Pipe()
        self.process = mp.Process(target=_stepper_process, args=(pipe, DIR, ENA, PUL, acceleration, speed))
        self.process.start()
        self.isRunning = True
        
    def move(self, position) :
        self.pipe.send(position)
        LogData.append([time.time(), "mv", position])
    
    def stop(self) :
        time.sleep(3)
        self.pipe.send("stop")
        print("wait for sub process to stop")
        self.process.join()
        self.isRunning = False
    
    def __del__(self) :
        if self.isRunning :
            self.stop()
            
gpio.setup(37, gpio.OUT)     
gpio.setup(33, gpio.OUT)    
gpio.setup(36, gpio.OUT)
gpio.setup(35, gpio.OUT)   
gpio.setup(38, gpio.OUT)
gpio.setup(40, gpio.OUT)
gpio.output(37, gpio.HIGH)
def hit_ball(DIR=35, ENA=37, PUL=33) :
    gpio.output(ENA, gpio.LOW)
    speeda = 0
    speedb = 0
    step = 100
    gpio.output(DIR, gpio.LOW)
    for i in range(step) : 
        if i > step/2 :
            speeda -= 8
        else :
            speeda += 8
        pulse_time = 1/speeda
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)
    time.sleep(0.3)
    gpio.output(DIR, gpio.HIGH)
    for i in range(step) :
        if i > step/2 :
            speedb -= 5
        else :
            speedb += 5
        pulse_time = 1/speedb
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
    gpio.output(ENA, gpio.HIGH)


def getIP() :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def main() :
        stepper = Stepper()
        servo = serial.Serial(port='/dev/ttyUSB0', baudrate=9600)
        servo.write(int(500 + (140*2000)/270).to_bytes(2, byteorder="big"))

        ip = getIP()
        print(ip)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((ip, 5678))
        serversocket.listen()
        clientsocket = None
        (clientsocket, address) = serversocket.accept()
        print("connected")
        while True :
            ret = clientsocket.recv(100)
            LogData.append([time.time(), "rc", (ret, len(ret))])

            #i = 0
            #ret = clientsocket.recv(5)
            #while len(select.select([clientsocket], [], [], 0.0)[0]) != 0:
                #ret = clientsocket.recv(5)
                #i += 1
            #LogData.append([time.time(), "rc", (ret, len(ret), i)])


            if len(ret) >= 5 and len(ret) % 5 == 0:
                ret = ret[-5:]
                key = ret[0:1].decode("ascii")
                if key == "m":
                    deg = ret[1:3]
                    pos = int.from_bytes(ret[3:5], byteorder="big")
                    pos -= 1000
                    servo.write(deg)
                    stepper.move(pos)
                
                if key == "h" :
                    hit_ball()
                if key == "q" :
                    print("stop signal received")
                    break
            else :
                print(len(ret)) if len(ret) != 0 else None
        stepper.stop()
        serversocket.close()
                    
    #except Exception as e:
        #raise e
        #stepper.stop()
        #serversocket.close()
        
    
if __name__ == "__main__" :
    
    main()
    with open("log", "w") as f :
        for l in LogData:
            f.write(str(l) + "\n")
    for l in LogData:
        print(l)
    exit()
    servo = serial.Serial(port='/dev/ttyUSB0', baudrate=9600)
    while True :
        deg = int(input())
        servo.write(int(500 + (deg*2000)/270).to_bytes(2, byteorder="big"))
        #print(str(500 + (a*2000)/270).encode(encoding="gbk"))
        #print(servo.readline())
    exit()
    
    ip = getIP()
    print(ip)
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((ip, 5678))
    serversocket.listen()
    clientsocket = None
    (clientsocket, address) = serversocket.accept()
    recv = clientsocket.recv(1024)
    clientsocket.sendall(struct.pack("!f", time.time()))
    print("time diff", struct.unpack("!f", recv)[0] - time.time())
    stepper = Stepper()
    while True :
        a = int(input())

        stepper.move(a)
    exit()
    servo = serial.Serial(port='/dev/ttyAMA0', baudrate=9600)
    while True :
        a = int(input())
        servo.write(str(500 + (a*2000)/270).encode(encoding="gbk"))
        #print(servo.read(6))
    exit()
    
    hit_ball()
    exit()
    hit_ball()
    exit()
    time.sleep(9)
    servo.write(str(500 + (220*2000)/270).encode(encoding="gbk"))
    exit()
    gpio.setup(PWM, gpio.OUT)
    pwm_motor = gpio.PWM(32, 50)
    pwm_motor.start(7.5)
    while True :
        a = int(input())
        servo(pwm_motor, a)
    exit()
    main()
    exit()
    gpio.setup(PWM, gpio.OUT)
    pwm_motor = gpio.PWM(32, 50)
    pwm_motor.start(7.5)
    while True :
        a = int(input())
        servo(pwm_motor, a)
    exit()
    hit_ball()
    
    exit()
    
    servo = serial.Serial(port='/dev/ttyAMA0', baudrate=9600)
    servo.write(b"2000") 
    servo.write(str(500 + (150*2000)/300).encode(encoding="gbk"))
    exit()
    gpio.setup(PWM, gpio.OUT)
    pwm_motor = gpio.PWM(32, 50)
    pwm_motor.start(7.5)
    while True :
        a = int(input())
        servo(pwm_motor, a)
    exit()
    #hit_ball()
    servo = Servo()
    while True :
        a = int(input())
        if a < 0 :
            hit_ball(38, 40)
        servo.move(a)
    
    exit()
    hit_ball(200, 38, 40)
    exit()

exit()
s = Stepper()
s.run()

exit()

    

exit()
p = mp.Process(target=f2)

p.start()
for i in range(10) :
    stepper_acceleration_absolute(1400, 900, 10)
    stepper_acceleration_absolute(0, 900, 10)
    
p.terminate()
    
  
servo(pwm_motor, 20)
time.sleep(2)
exit()
while True :
    a = int(input())
    stepper_acceleration_absolute(a, 1000, 10)


def f1() :

    j = 0
    while True :
        for i in range(250, 20, -10) :
            lift(pwm_motor, i)
            time.sleep(0.1)
        for i in range(20, 250, 10) :
            lift(pwm_motor, i)
            time.sleep(0.1)
        j+=1
        if j == 4:
            break
