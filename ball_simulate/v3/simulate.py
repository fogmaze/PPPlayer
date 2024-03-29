import time
import cv2
import numpy as np
import shutil
import argparse
import pybullet as p
import tqdm
import os
import sys
sys.path.append(os.getcwd())
import core.Equation3d as equ
import core.display as display
import core.Constants as c
import math
import random
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import multiprocessing
import ball_simulate.v3.dataFileOperator as dfo


class Work:
    timestamp:float
    index:int
    def action(self,ball_pos):
        pass

class CameraWork(Work):
    def __init__(self, timestamp, camera_pos, index):
        self.timestamp = timestamp
        self.camera_pos = camera_pos
        self.index = index
        
    def action(self, ball_pos_ideal):
        lineCamBall = equ.LineEquation3d(self.camera_pos, ball_pos_ideal + randomBallPosError())
        self.rad_xy = math.atan(lineCamBall.line_xy.a)
        self.rad_xz = math.atan(lineCamBall.line_xz.a)
        self.lineCamBall = lineCamBall

class BallWork(Work):
    def __init__(self, timestamp, index):
        self.timestamp = timestamp
        self.index = index

    def action(self, ball_pos):
        self.ball_pos:equ.Point3d = ball_pos

class Work_display:
    timestamp:float
    index:int
    def action(self, ax, ball_pos):
        pass

class CameraWork_display(Work):
    def __init__(self, timestamp, camera_pos, index):
        self.timestamp = timestamp
        self.camera_pos = camera_pos
        self.index = index
        
    def action(self, ax, ball_pos_ideal):
        lineCamBall = equ.LineEquation3d(self.camera_pos, ball_pos_ideal + randomBallPosError())
        self.rad_xy = math.atan(lineCamBall.line_xy.a)
        self.rad_xz = math.atan(lineCamBall.line_xz.a)
        self.lineCamBall = lineCamBall
        color = ['orange', "green"]
        display.drawLine3d(ax, lineCamBall, color[self.index[0]])
        

class BallWork_display(Work):
    def __init__(self, timestamp, index):
        self.timestamp = timestamp
        self.index = index

    def action(self, ax, ball_pos):
        self.ball_pos:equ.Point3d = ball_pos
        display.plotPoint(ax, ball_pos)

def randomCameraPos():
    x = 0
    y = 0
    z = 0
    # if pos not in ball_area, return random pos
    while x < c.BALL_AREA_HALF_LENGTH and x > -c.BALL_AREA_HALF_LENGTH and y < c.BALL_AREA_HALF_WIDTH and y > -c.BALL_AREA_HALF_WIDTH and z < c.BALL_AREA_HEIGHT:
        x = random.uniform(-c.CAMERA_AREA_HALF_LENGTH, c.CAMERA_AREA_HALF_LENGTH)
        y = random.uniform(-c.CAMERA_AREA_HALF_WIDTH, c.CAMERA_AREA_HALF_WIDTH)
        z = random.uniform(0, c.CAMERA_AREA_HEIGHT)
    return equ.Point3d(x, y, z)

def randomCameraPosError():
    x = random.gauss(0, c.CAMERA_POSITION_ERROR_STD)
    y = random.gauss(0, c.CAMERA_POSITION_ERROR_STD)
    z = random.gauss(0, c.CAMERA_POSITION_ERROR_STD)
    return equ.Point3d(x, y, z)

def randomBallPos():
    x = random.uniform(-c.BALL_AREA_HALF_LENGTH, c.BALL_AREA_HALF_LENGTH)
    y = random.uniform(-c.BALL_AREA_HALF_WIDTH, c.BALL_AREA_HALF_WIDTH)
    z = random.uniform(0, c.BALL_AREA_HEIGHT)
    return equ.Point3d(x, y, z)

def randomBallPosError():
    x = random.gauss(0, c.BALL_POSITION_ERROR_STD)
    y = random.gauss(0,c. BALL_POSITION_ERROR_STD)
    z = random.gauss(0, c.BALL_POSITION_ERROR_STD)
    return equ.Point3d(x, y, z)

def randomInpIdxs() -> List[int]:
    ignore_area_len = round(random.gauss(c.INPUT_IGNORE_AREA_MEAN, c.INPUT_IGNORE_AREA_STD))
    all = [1] * c.SIMULATE_INPUT_LEN
    for i in range(ignore_area_len) :
        beg = random.randrange(0, c.SIMULATE_INPUT_LEN)
        for j in range(beg, beg + round(random.gauss(c.INPUT_IGNORE_WIDTH_MEAN, c.INPUT_IGNORE_WIDTH_STD))):
            if j < c.SIMULATE_INPUT_LEN:
                all[j] = 0
    res = [i for i in range(c.SIMULATE_INPUT_LEN) if all[i] == 1]
    if len(res) < 3 :
        return randomInpIdxs()
    return res

def configRoom(ax:Axes):
    lim = c.CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def drawLine3d(axe:plt.Axes,line:equ.LineEquation3d):
    points = [line.getPoint({'x':-c.BALL_AREA_HALF_LENGTH}),line.getPoint({'x':c.BALL_AREA_HALF_LENGTH})]
    X = [points[0][0],points[1][0]]
    Y = [points[0][1],points[1][1]]
    Z = [points[0][2],points[1][2]]
    axe.plot(X,Y,Z)

def createRoom()->Axes:
    ax = plt.axes(projection='3d')
    configRoom(ax)
    return ax

def cleenRoom(axe:plt.Axes):
    axe.cla()
    configRoom(ax=axe)

def plotData(ax,inp:Tuple[List[CameraWork]],ans:List[BallWork]):
    for input_data in inp:
        for work in input_data:
            drawLine3d(ax,work.lineCamBall)
    # plot ball points
    for pos in ans:
        ax.scatter(pos.ball_pos.x,pos.ball_pos.y,pos.ball_pos.z)

def simulate_display(GUI = False, dataLength = 10):
    SINGLE_SIMULATE_SAMPLE_LEN = 5
    if GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setPhysicsEngineParameter(restitutionVelocityThreshold=0)

    planeId = p.createCollisionShape(p.GEOM_PLANE)
    plane = p.createMultiBody(0, planeId)

    radius = 0.04
    sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    sphere = p.createMultiBody(27, sphereId, basePosition=startPos, baseOrientation=startOrientation)

    linearVelocity = [0,0, random.uniform(0, 5)]
    angularVelocity = [0, 0, 0]
    p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)

    #linearDamping = 6 * math.pi * radius * 0.0185
    #angularDamping = 0.1
    #p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)


    restitution = 1
    p.changeDynamics(sphere, -1, restitution=restitution)
    p.changeDynamics(plane, -1, restitution=restitution)
    p.setRealTimeSimulation(0)
    p.setTimeStep(c.stepTime)

    p.setGravity(0, 0, -c.G)
    fig, ax = display.createFigRoom()

    for i in tqdm.tqdm(range(int(dataLength/SINGLE_SIMULATE_SAMPLE_LEN))) :
        video = cv2.VideoWriter("exp/{}.mp4".format(i), cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
        cam1_pos = randomCameraPos()
        cam2_pos = randomCameraPos()

        cam1_error = randomCameraPosError()
        cam2_error = randomCameraPosError()

        ball_pos = randomBallPos()

        camera_systematic_error = random.gauss(0, c.SHUTTER_SYSTEMATIC_ERROR_STD)

        #set ball pos
        p.resetBasePositionAndOrientation(sphere, [ball_pos.x, ball_pos.y, ball_pos.z], startOrientation)
        linearVelocity = [random.uniform(-5,5), random.uniform(-5,5), random.uniform(0, 3)]
        angularVelocity = [0, 0, 0]
        p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)


        works:List[Work_display] = []
        cam1_data:List[CameraWork] = []
        cam2_data:List[CameraWork] = []
        ans_data:List[BallWork] = []
        for j in range(c.SIMULATE_INPUT_LEN):
            works.append(CameraWork_display(abs(j/c.FPS + random.gauss(0,c.SHUTTER_RANDOM_ERROR_STD)), cam1_pos + cam1_error, (0,j)))
            works.append(CameraWork_display(abs(j/c.FPS + random.gauss(0,c.SHUTTER_RANDOM_ERROR_STD) + camera_systematic_error), cam2_pos + cam2_error, (1, j)))
        for j in range(c.SIMULATE_TEST_LEN):
            works.append(BallWork_display(j*c.CURVE_SHOWING_GAP, (2, j)))
        works = sorted(works, key = lambda x: x.timestamp)
        nowTimeStamp = 0
        nowWorkIndex = 0
        i = 0

        display.cleanRoom(ax)
        while len(works) > nowWorkIndex:
            if works[nowWorkIndex].timestamp > nowTimeStamp:
            #if True:
                p.stepSimulation()
                nowTimeStamp += c.stepTime
                if GUI:
                    time.sleep(c.stepTime)
                if i % 20 == 0:
                    fig.canvas.draw()
                    img = np.fromstring(ax.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img = img.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img)
                i += 1
                continue

            #get ball pos
            ball_pos_list = p.getBasePositionAndOrientation(sphere)[0]
            ball_pos = equ.Point3d(ball_pos_list[0], ball_pos_list[1], ball_pos_list[2])

            #do work
            works[nowWorkIndex].action(ax, ball_pos)
            if works[nowWorkIndex].index[0] == 0:
                cam1_data.append(works[nowWorkIndex])
            elif works[nowWorkIndex].index[0] == 1:
                cam2_data.append(works[nowWorkIndex])
            elif works[nowWorkIndex].index[0] == 2:
                ans_data.append(works[nowWorkIndex])

            nowWorkIndex += 1
        video.release()



def simulate(GUI = False, dataLength = 10, outputFileName = "train.bin", mode = "better3") :
    if GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setPhysicsEngineParameter(restitutionVelocityThreshold=0)

    planeId = p.createCollisionShape(p.GEOM_PLANE)
    plane = p.createMultiBody(0, planeId)

    radius = 0.04
    sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    sphere = p.createMultiBody(27, sphereId, basePosition=startPos, baseOrientation=startOrientation)

    linearVelocity = [0,0, random.uniform(0, 5)]
    angularVelocity = [0, 0, 0]
    p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)

    #linearDamping = 6 * math.pi * radius * 0.0185
    #angularDamping = 0.1
    #p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)


    restitution = 1
    p.changeDynamics(sphere, -1, restitution=restitution)
    p.changeDynamics(plane, -1, restitution=restitution)
    p.setRealTimeSimulation(0)
    p.setTimeStep(c.stepTime)

    p.setGravity(0, 0, -c.G)

    dataset = dfo.BallDataSet_put(outputFileName, dataLength, mode)
    for i in tqdm.tqdm(range(dataLength)) :
        cam1_end = random.randint(3, c.SIMULATE_INPUT_LEN)
        cam2_end = min(random.randint(cam1_end-2, cam1_end+2), c.SIMULATE_INPUT_LEN)

        cam1_pos = randomCameraPos()
        cam2_pos = randomCameraPos()

        cam1_error = randomCameraPosError()
        cam2_error = randomCameraPosError()

        ball_pos = randomBallPos()

        camera_systematic_error = random.gauss(0, c.SHUTTER_SYSTEMATIC_ERROR_STD)

        #set ball pos
        p.resetBasePositionAndOrientation(sphere, [ball_pos.x, ball_pos.y, ball_pos.z], startOrientation)
        linearVelocity = [random.uniform(-5,5), random.uniform(-5,5), random.uniform(0, 3)]
        angularVelocity = [0, 0, 0]
        p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)


        works:List[Work] = []
        cam1_data:List[CameraWork] = []
        cam2_data:List[CameraWork] = []
        ans_data:List[BallWork] = []
        for j in range(c.SIMULATE_INPUT_LEN):
            works.append(CameraWork(abs(j/c.FPS + random.gauss(0,c.SHUTTER_RANDOM_ERROR_STD)), cam1_pos + cam1_error, (0,j)))
            works.append(CameraWork(abs(j/c.FPS + random.gauss(0,c.SHUTTER_RANDOM_ERROR_STD) + camera_systematic_error), cam2_pos + cam2_error, (1, j)))
        for j in range(c.SIMULATE_TEST_LEN):
            works.append(BallWork(j * c.CURVE_SHOWING_GAP + cam1_end / c.FPS, (2, j)))
        works = sorted(works, key = lambda x: x.timestamp)
        nowTimeStamp = 0
        nowWorkIndex = 0
        while len(works) > nowWorkIndex:
            if works[nowWorkIndex].timestamp > nowTimeStamp:
            #if True:
                p.stepSimulation()
                nowTimeStamp += c.stepTime
                if GUI:
                    time.sleep(c.stepTime)
                continue

            #get ball pos
            ball_pos_list = p.getBasePositionAndOrientation(sphere)[0]
            ball_pos = equ.Point3d(ball_pos_list[0], ball_pos_list[1], ball_pos_list[2])

            #do work
            works[nowWorkIndex].action(ball_pos)

            if works[nowWorkIndex].index[0] == 0:
                cam1_data.append(works[nowWorkIndex])
            elif works[nowWorkIndex].index[0] == 1:
                cam2_data.append(works[nowWorkIndex])
            elif works[nowWorkIndex].index[0] == 2:
                ans_data.append(works[nowWorkIndex])

            nowWorkIndex += 1
        
        if GUI:
            ax = createRoom()
            configRoom(ax)
            plotData(ax,(cam1_data,cam2_data),ans_data)
            plt.show()

        cam1_ign = randomInpIdxs()
        cam2_ign = randomInpIdxs()
        if not cam1_ign[0] == 0 :
            cam1_ign.insert(0,0)
        # save data
        dataStruct = dataset.dataStructClass()
        dataStruct.inputs[0].camera_x = cam1_pos.x
        dataStruct.inputs[0].camera_y = cam1_pos.y
        dataStruct.inputs[0].camera_z = cam1_pos.z
        dataStruct.inputs[1].camera_x = cam2_pos.x
        dataStruct.inputs[1].camera_y = cam2_pos.y
        dataStruct.inputs[1].camera_z = cam2_pos.z
        dataStruct.inputs[0].seq_len = cam1_end
        dataStruct.inputs[1].seq_len = cam2_end

        ind = 0
        for k in cam1_ign:
            dataStruct.inputs[0].line_rad_xy[ind] = cam1_data[k].rad_xy
            dataStruct.inputs[0].line_rad_xz[ind] = cam1_data[k].rad_xz
            ind += 1
            if ind >= cam1_end:
                break
            if random.random() < c.INPUT_RANDOM_ERROR_RATE :
                dataStruct.inputs[0].line_rad_xy[ind] = random.uniform(-math.pi, math.pi)
                dataStruct.inputs[0].line_rad_xz[ind] = random.uniform(-math.pi, math.pi)
                ind += 1
            if ind >= cam1_end:
                break
        dataStruct.inputs[0].seq_len = ind

        ind = 0
        for k in cam2_ign:
            dataStruct.inputs[1].line_rad_xy[ind] = cam2_data[k].rad_xy
            dataStruct.inputs[1].line_rad_xz[ind] = cam2_data[k].rad_xz
            ind += 1
            if ind >= cam1_end:
                break
            if random.random() < c.INPUT_RANDOM_ERROR_RATE :
                dataStruct.inputs[0].line_rad_xy[ind] = random.uniform(-math.pi, math.pi)
                dataStruct.inputs[0].line_rad_xz[ind] = random.uniform(-math.pi, math.pi)
                ind += 1
            if ind >= cam2_end:
                break
        dataStruct.inputs[1].seq_len = ind

        for k in range(len(ans_data)):
            dataStruct.curvePoints[k].x = ans_data[k].ball_pos.x
            dataStruct.curvePoints[k].y = ans_data[k].ball_pos.y
            dataStruct.curvePoints[k].z = ans_data[k].ball_pos.z

        c.normer.norm(dataStruct)
        dataset.putData(i, dataStruct)


def work_simulate(outputFileName, fromIdx):
    pass

def simulate_fast(dataLength = 10, num_workers = 1, outputFileName = "train.bin"):
    queue = multiprocessing.Queue()
    workers = []
    eachLen = dataLength // num_workers
    lessLen = dataLength % num_workers
    for i in range(num_workers-1):
        workers.append(multiprocessing.Process(target=work_simulate, args=(queue, eachLen)))
        workers[i].start()

    workers.append(multiprocessing.Process(target=work_simulate, args=(queue, eachLen + lessLen)))
    workers[len(workers)-1].start()
    saver = multiprocessing.Process(target=None, args=(queue, outputFileName, dataLength))
    saver.start()

    while saver.is_alive() :
        try :
            saver.join(timeout=1)
            time.sleep(1)
        except KeyboardInterrupt:
            # kill all workers
            for worker in workers:
                worker.terminate()
            saver.terminate()

    print("simulate done")

if __name__ == "__main__":
    #print(calculateMeanStd("train.bin"))
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--GUI", default=False, action="store_true")
    argparser.add_argument("-l", default=1000, type=int)
    argparser.add_argument("-n", default="test")
    argparser.add_argument("--mode", default="better3", type=str)

    #fetch params
    args = argparser.parse_args()
    if args.mode != "default" :
        if args.mode == "normalBR3" :
            c.set2NormalBR3()
        elif args.mode == "better3" :
            c.set2Better3()

    if not os.path.isdir("ball_simulate/v3/dataset"):
        os.mkdir("ball_simulate/v3/dataset")

    simulate(GUI=args.GUI, dataLength=args.l, outputFileName="ball_simulate/v3/dataset/{}.train.bin".format(args.n), mode=args.mode)
    simulate(GUI=args.GUI, dataLength=10000, outputFileName="ball_simulate/v3/dataset/{}.valid.bin".format(args.n), mode=args.mode)
    pass

