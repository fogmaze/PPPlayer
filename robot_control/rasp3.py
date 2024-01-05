import RPi.gpio as GPIO
import time
import multiprocessing as mp
import threading
import socket
import math
import serial
import csv

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


def _stepper_process(pipe: mp.Pipe, DIR=38, ENA=36, PUL=40, acceleration=50, speed=1200) :
    GPIO.setup(PUL, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.output(ENA, GPIO.LOW)
    initial_speed = 100
    sq_initial_speed = initial_speed*initial_speed
    threshold = initial_speed * 1.2
    
    recent_position = -756
    destination = 0
    last_pulse_time = 0

    velocity = 0
    while True :
        if pipe.poll():
            destination = pipe.recv()
            print(destination)
            if destination == "stop" :
                break
                print("stop received")
        GPIO.output(ENA, GPIO.LOW)
        if recent_position == destination :
            GPIO.output(ENA, GPIO.HIGH)
            velocity = 0
            last_pulse_time = 0
            continue
            
        #whether to accelerate or decelerate
        distance = destination - recent_position
        isSlowing = (velocity*velocity - sq_initial_speed) / 2 / acceleration >= abs(distance)
            
        if velocity == 0 :
            if distance > 0 :
                velocity = initial_speed
            elif distance < 0 :
                velocity = -initial_speed
        elif velocity < 0 :
            if distance < 0 :
                if isSlowing :
                    velocity += acceleration * last_pulse_time
                elif abs(velocity) < speed :
                    velocity -= acceleration * last_pulse_time
            else :
                velocity += acceleration * last_pulse_time
        elif velocity > 0 :
            if distance > 0 :
                if isSlowing :
                    velocity -= acceleration * last_pulse_time
                elif velocity < speed :
                    velocity += acceleration * last_pulse_time
            else :
                velocity -= acceleration * last_pulse_time
                    
        if velocity == 0 :
            continue
        elif velocity > 0 :
            GPIO.output(DIR, GPIO.HIGH)
            recent_position += 1
        elif velocity < 0 :
            GPIO.output(DIR, GPIO.LOW)
            recent_position -= 1
        print(recent_position, velocity)
        pulse_time = 1/abs(velocity)
        GPIO.output(PUL, GPIO.HIGH)
        time.sleep(pulse_time/2)
        GPIO.output(PUL, GPIO.LOW)
        time.sleep(pulse_time/2)
        last_pulse_time = pulse_time
        if isSlowing and abs(velocity) < threshold :
            velocity = 0
            last_pulse_time = 0

    print("pipe stopped")
        


class Stepper() :
    def __init__(self, DIR=38, ENA=40, PUL=36, acceleration=400, speed=1000) :
        self.pipe, pipe = mp.Pipe()
        self.process = mp.Process(target=_stepper_process, args=(pipe, DIR, ENA, PUL, acceleration, speed))
        self.process.start()
        self.isRunning = True
        
    def move(self, position) :
        self.pipe.send(position)
    
    def stop(self) :
        time.sleep(3)
        self.pipe.send("stop")
        print("wait for sub process to stop")
        self.process.join()
        self.isRunning = False
    
    def __del__(self) :
        if self.isRunning :
            self.stop()
            
GPIO.setup(37, GPIO.OUT)     
GPIO.setup(33, GPIO.OUT)    
GPIO.setup(36, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)   
GPIO.setup(38, GPIO.OUT)
GPIO.setup(40, GPIO.OUT)
GPIO.output(37, GPIO.HIGH)
def hit_ball(DIR=35, ENA=37, PUL=33) :
    GPIO.output(ENA, GPIO.LOW)
    speeda = 0
    speedb = 0
    step = 100
    GPIO.output(DIR, GPIO.LOW)
    for i in range(step) : 
        if i > step/2 :
            speeda -= 8
        else :
            speeda += 8
        pulse_time = 1/speeda
        GPIO.output(PUL, GPIO.LOW)
        time.sleep(pulse_time/2)
        GPIO.output(PUL, GPIO.HIGH)
        time.sleep(pulse_time/2)
    time.sleep(0.3)
    GPIO.output(DIR, GPIO.HIGH)
    for i in range(step) :
        if i > step/2 :
            speedb -= 5
        else :
            speedb += 5
        pulse_time = 1/speedb
        GPIO.output(PUL, GPIO.HIGH)
        time.sleep(pulse_time/2)
        GPIO.output(PUL, GPIO.LOW)
        time.sleep(pulse_time/2)
    GPIO.output(ENA, GPIO.HIGH)


def getIP() :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def main() :
        ip = getIP()
        print(ip)
        stepper = Stepper()
        servo = None# serial.Serial(port='/dev/ttyUSB0', baudrate=9600)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((ip, 5678))
        serversocket.listen()
        clientsocket = None
        servo.write(int(500 + (140*2000)/270).to_bytes(2, byteorder="big"))
        
        (clientsocket, address) = serversocket.accept()
        print("connected")
        while True :
            ret = clientsocket.recv(5)
            key = ret[0:1].decode("ascii")
            
            if len(ret) == 5:
                if key == "m":
                    deg = ret[1:3]
                    pos = int.from_bytes(ret[3:5], byteorder="big")
                    pos -= 1000
                    servo.write(deg)
                    stepper.move(pos)
                    print(int.from_bytes(deg, byteorder="big"), pos)
                
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
    
    stepper = Stepper()
    while True :
        a = int(input())
        if a == -2 :
            stepper.stop()
            exit()
        stepper.move(a)
    stepper.stop()
    exit()

    main()
    exit()
    servo = serial.Serial(port='/dev/ttyUSB0', baudrate=9600)
    while True :
        deg = int(input())
        servo.write(int(500 + (deg*2000)/270).to_bytes(2, byteorder="big"))
        #print(str(500 + (a*2000)/270).encode(encoding="gbk"))
        #print(servo.readline())
    exit()
    
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
    