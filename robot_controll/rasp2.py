import RPi.GPIO as gpio
import time
import multiprocessing as mp
import threading
import socket
import math
import serial
import csv
#import busio
#import board
#from adafruit_pca9685 import PCA9685
#from adafruit_motor import servo 

gpio.setmode(gpio.BOARD)
gpio.setwarnings(False)

DIR=38
ENA=36
PUL=40

PWM=32



def stepper(direction, step, speed) :
    pulse_time = 1/speed
    if direction  == "l":
        gpio.output(DIR, gpio.LOW)
    elif direction == "r":
        gpio.output(DIR, gpio.HIGH)
    for i in range(step) :
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)        
        
def stepper_acceleration(direction, step, speed, acceleration=100) :
    velocity = 100
    if direction  == "l":
        gpio.output(DIR, gpio.LOW)
    elif direction == "r":
        gpio.output(DIR, gpio.HIGH)
    for i in range(1, step+1):
        if i % 1 == 0 and i < (speed/acceleration)*1 :
            velocity += acceleration
        elif (step - i)%1 == 0 and (step-i) < (speed/acceleration)*1 :
            velocity -= acceleration
        if velocity == 0 :
            break
        pulse_time = 1/velocity
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)

def stepper_acceleration_absolute(position, speed, acceleration=100) :
    global stepper_recent_position
    velocity = 100
    step = position - stepper_recent_position 
    stepper_recent_position = position
    if step > 0 :
        gpio.output(DIR, gpio.LOW)
    elif step < 0 :
        step = -step
        gpio.output(DIR, gpio.HIGH)
    for i in range(1, step+1):
        if i % 1 == 0 and i < (speed/acceleration)*1 :
            velocity += acceleration
        elif (step - i)%1 == 0 and (step-i) < (speed/acceleration)*1 :
            velocity -= acceleration
        if velocity == 0 :
            break
        pulse_time = 1/velocity
        gpio.output(PUL, gpio.LOW)
        time.sleep(pulse_time/2)
        gpio.output(PUL, gpio.HIGH)
        time.sleep(pulse_time/2)



def _stepper_process(pipe: mp.Pipe, DIR=38, ENA=36, PUL=40, acceleration=8, speed=1200) :
    gpio.setup(PUL, gpio.OUT)
    gpio.setup(ENA, gpio.OUT)
    gpio.setup(DIR, gpio.OUT)
    gpio.output(ENA, gpio.LOW)
    
    ALL_LENGTH = 1512/4
    recent_position = -756
    destination = 0
    isRunning = True

    velocity = 0
    while True :
        if pipe.poll() :
            destination = pipe.recv()
            if destination == "stop" :
                isRunning = False
                destination = -ALL_LENGTH/2
        if destination == recent_position :
            if not isRunning :
                break
            gpio.output(ENA, gpio.HIGH)
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
        


class Stepper() :
    def __init__(self, DIR=38, ENA=40, PUL=36, acceleration=8, speed=1200) :
        self.pipe, pipe = mp.Pipe()
        self.process = mp.Process(target=_stepper_process, args=(pipe, DIR, ENA, PUL, acceleration, speed))
        self.process.start()
        self.isRunning = True
        
    def move(self, position) :
        self.pipe.send(position)
    
    def stop(self) :
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
    try:
        ip = getIP()
        print(ip)
        stepper = Stepper()
        servo = serial.Serial(port='/dev/ttyAMA0', baudrate=9600)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((ip, 6678))
        serversocket.listen()
        clientsocket = None

        while True :
            (clientsocket, address) = serversocket.accept()
            while True :
                ret = str(clientsocket.recv(1024), encoding='utf-8')
                cmd = ret.split()
                if len(cmd) == 0:
                    pass
                elif cmd[0] == "q" :
                    break
                elif cmd[0] == "m" :
                    deg  = float(cmd[1])
                    base_pos = int(cmd[2])
                    print("move to", deg, base_pos)
                    #servo.write(str(500 + (deg*2000)/270).encode(encoding="gbk"))
                    stepper.move(base_pos)
                elif cmd[0] == "hit" :
                    hit_ball()
    except Exception as e:
        stepper.stop()
        serversocket.close()
        raise e
        
                 
def count() :
    c = 0
    for i in range(100) :
        
        if i % 2 == 0 :
            a = time.time()
            stepper_acceleration_absolute(1512, 1000, 10)
            b = time.time()
        else :
            a = time.time()
            stepper_acceleration_absolute(0, 1000, 10)
            b = time.time()
        c += (b-a)
    return c/100
    
if __name__ == "__main__" :
    main()
    exit()
    
    servo = serial.Serial(port='/dev/ttyAMA0', baudrate=9600)
    while True :
        a = int(input())
        servo.write(str(500 + (a*2000)/270).encode(encoding="gbk"))
        #print(servo.read(6))
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