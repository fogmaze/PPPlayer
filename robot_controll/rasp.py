import RPi.GPIO as gpio
import time
import multiprocessing as mp
import threading
import socket
import math
import serial
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



servo_recent_deg = 135
stepper_recent_position = 0

                 
            
     
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



def servo_acc(pwm_pin, deg) :
    direction = 2
    global servo_recent_deg
    pulse_time = 0.1
    if servo_recent_deg > deg :
        direction = 1
    elif servo_recent_deg < deg :
        direction = 0
    servo_recent_deg = deg
    for i in range(servo_recent_deg, deg+1) :
        pwm_pin.ChangeDutyCycle(2.5 + 10*(i/300))
        
        if direction == 1 and i < 105 :
            pulse_time = 0.2
        elif direction == 0 and i > 165 :
            pulse_time = 0.2
         
        time.sleep(pulse_time)
       
def servo(pwm_pin, deg) :
        pwm_pin.ChangeDutyCycle(2.5 + 10*(deg/300))

class Servo_arduino(threading.Thread) :
    def __init__(self, PWM=32) :
        threading.Thread.__init__(self)
        gpio.setup(PWM, gpio.OUT)
        self.PWM = PWM
        self.pwm_motor = gpio.PWM(PWM, 50)
        self.pwm_motor.start(7.5)
        self.recent_deg = 150
        self.destination = 150
        self.pulse_time = 4/300
        self.lock = threading.Lock()
        self.isRunning = True
        self.start()

    def move(self, deg) :
        self.lock.acquire()
        self.destination = deg
        self.lock.release()

    def stop(self):
        self.lock.acquire()
        self.isRunning = False
        self.lock.release()

    def run(self) :
        while True :
            self.lock.acquire()
            if not self.isRunning :
                self.destination = 150
            if self.recent_deg == self.destination :
                self.lock.release()
                if not self.isRunning :
                    break
                continue
            if self.recent_deg > self.destination :
                self.pwm_motor.ChangeDutyCycle(2.5 + 10*((self.recent_deg-1)/300))
                self.recent_deg -= 1
            else :
                self.pwm_motor.ChangeDutyCycle(2.5 + 10*((self.recent_deg+1)/300))
                self.recent_deg += 1
            self.lock.release()
            time.sleep(self.pulse_time)
            
            
class Servo(threading.Thread) :
    def __init__(self) :
        threading.Thread.__init__(self)
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.s = servo.Servo(self.pca.channels[15], actuation_range=270, min_pulse=500, max_pulse=2500)    
        self.a = 135
        self.s.angle = 135
        self.ra = 135
        self.start()
        
    def move(self, a) :
        self.a = a
       
    def run(self) :
        while True :
            if abs(self.ra-self.a) < 10 :
                self.s.angle = self.a
                self.ra = self.a
            elif self.ra > self.a :
                self.ra-=10
                self.s.angle = self.ra
            elif self.ra < self.a :
                self.ra+=10
                self.s.angle = self.ra
            time.sleep(3/270)
            
            
class Stepper(threading.Thread) :
    def __init__(self, DIR=38, ENA=40, PUL=36):
        threading.Thread.__init__(self)
        self.ALL_LENGTH = 1512/4
        self.recent_position = -756
        self.destination = -756
        self.DIR = DIR
        self.ENA = ENA
        self.PUL = PUL
        gpio.setup(self.PUL, gpio.OUT)
        gpio.setup(self.ENA, gpio.OUT)
        gpio.setup(self.DIR, gpio.OUT)
        self.lock = threading.Lock()
        self.isRunning = True
        self.start()
        gpio.output(self.ENA, gpio.LOW)
        
    def move(self, position) :
        self.lock.acquire()
        self.destination = position
        self.lock.release()
    
    def stop(self):
        self.lock.acquire()
        self.isRunning = False
        self.lock.release()
    
    def run(self, speed=1500, acceleration=10) :
        velocity = 0
        while True :
            self.lock.acquire()
            if not self.isRunning :
                self.destination = -self.ALL_LENGTH/2
            des = self.destination
            if self.destination == self.recent_position :
                self.lock.release()
                if not self.isRunning :
                    break
                gpio.output(self.ENA, gpio.HIGH)
                continue
            gpio.output(self.ENA, gpio.LOW)
            
            #whether to accelerate or decelerate
            distance = self.destination - self.recent_position
            self.lock.release()
            
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
                self.recent_position += 1
            elif velocity < 0 :
                gpio.output(DIR, gpio.LOW)
                self.recent_position -= 1
            pulse_time = 1/abs(velocity)
            gpio.output(self.PUL, gpio.HIGH)
            time.sleep(pulse_time/2)
            gpio.output(self.PUL, gpio.LOW)
            time.sleep(pulse_time/2)
            
    def __del__(self) :
        self.stop()
        self.join()
        
            
            
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


def getIP() :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def main() :
    try:
        ip = getIP()
        print(ip)
        stepper = Stepper()
        #servo = Servo()
        servo = serial.Serial(port='/dev/ttyAMA0', baudrate=9600)
        #servo.move(150)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((ip, 5678))
        serversocket.listen()
        arm_length = 0.465
        arm_height = 0.46
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
                    print(cmd)
                    deg = math.asin((float(cmd[2])-arm_height)/arm_length) / math.pi * 180
                    print(deg)
                    arm_broad = abs(math.cos(deg) * arm_length)
                    if len(cmd) == 0:
                        pass
                    elif float(cmd[1]) > 0 :
                        deg = 240-deg
                        print("move ta",deg, round((float(cmd[1])-arm_broad)/0.0008))
                        servo.write(str(500 + (deg*2000)/270).encode(encoding="gbk"))
                        stepper.move(round((float(cmd[1])-arm_broad)/0.0008))
                    else :
                        print("move to", 240-(180-deg), round((float(cmd[1])-arm_broad)/0.0008))
                        servo(ser,240-(180-deg))
                        stepper.move(round((float(cmd[1])+arm_broad)/0.0008))
                elif cmd[0] == "hit" :
                    hit_ball()
    except Exception as e:
        stepper.__del__()
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