import time
import argparse
import pybullet as p
import tqdm
import os
import sys
sys.path.append(os.getcwd())
import core.Equation3d as equ
from core.Constants import *
import math
import random
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import multiprocessing
import ball_simulate_v2.dataFileOperator as dfo


stepTime = 1./600.
G = 9.8
FPS = 30

SHUTTER_RANDOM_ERROR_STD = 0.005
SHUTTER_SYSTEMATIC_ERROR_STD = 0.01


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
        
    def action(self, ball_pos):
        lineCamBall = equ.LineEquation3d(self.camera_pos, ball_pos)
        self.rad_xy = math.atan(lineCamBall.line_xy.a)
        self.rad_xz = math.atan(lineCamBall.line_xz.a)
        self.lineCamBall = lineCamBall

class BallWork(Work):
    def __init__(self, timestamp, index):
        self.timestamp = timestamp
        self.index = index

    def action(self, ball_pos):
        self.ball_pos:equ.Point3d = ball_pos

def randomCameraPos():
    x = 0
    y = 0
    z = 0
    # if pos not in ball_area, return random pos
    while x < BALL_AREA_HALF_LENGTH and x > -BALL_AREA_HALF_LENGTH and y < BALL_AREA_HALF_WIDTH and y > -BALL_AREA_HALF_WIDTH and z < BALL_AREA_HEIGHT:
        x = random.uniform(-CAMERA_AREA_HALF_LENGTH, CAMERA_AREA_HALF_LENGTH)
        y = random.uniform(-CAMERA_AREA_HALF_WIDTH, CAMERA_AREA_HALF_WIDTH)
        z = random.uniform(0, CAMERA_AREA_HEIGHT)
    return equ.Point3d(x, y, z)

def randomBallPos():
    x = random.uniform(-BALL_AREA_HALF_LENGTH, BALL_AREA_HALF_LENGTH)
    y = random.uniform(-BALL_AREA_HALF_WIDTH, BALL_AREA_HALF_WIDTH)
    z = random.uniform(0, BALL_AREA_HEIGHT)
    return equ.Point3d(x, y, z)

def configRoom(ax:Axes):
    lim = CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def drawLine3d(axe:plt.Axes,line:equ.LineEquation3d):
    points = [line.getPoint({'x':-BALL_AREA_HALF_LENGTH}),line.getPoint({'x':BALL_AREA_HALF_LENGTH})]
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


def simulate(GUI = False, dataLength = 10, outputFileName = "train.bin"):
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

    linearDamping = 6 * math.pi * radius * 0.0185
    angularDamping = 0.1
    p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)


    restitution = 1
    p.changeDynamics(sphere, -1, restitution=restitution)
    p.changeDynamics(plane, -1, restitution=restitution)
    p.setRealTimeSimulation(0)
    p.setTimeStep(stepTime)

    p.setGravity(0, 0, -G)

    dataset = dfo.BallDataSet_sync(outputFileName, dataLength)
    for i in tqdm.tqdm(range(int(dataLength/SINGLE_SIMULATE_SAMPLE_LEN))) :
        cam1_pos = randomCameraPos()
        cam2_pos = randomCameraPos()

        ball_pos = randomBallPos()

        #set ball pos
        p.resetBasePositionAndOrientation(sphere, [ball_pos.x, ball_pos.y, ball_pos.z], startOrientation)
        linearVelocity = [random.uniform(-5,5), random.uniform(-5,5), random.uniform(0, 3)]
        angularVelocity = [0, 0, 0]
        p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)


        works:List[Work] = []
        cam1_data:List[CameraWork] = []
        cam2_data:List[CameraWork] = []
        ans_data:List[BallWork] = []
        camera_systematic_error = random.normalvariate(0, SHUTTER_SYSTEMATIC_ERROR_STD)
        for j in range(SIMULATE_INPUT_LEN):
            works.append(CameraWork(abs(j/FPS + random.normalvariate(0,SHUTTER_RANDOM_ERROR_STD)), cam1_pos, (0,j)))
            works.append(CameraWork(abs(j/FPS + random.normalvariate(0,SHUTTER_RANDOM_ERROR_STD) + camera_systematic_error), cam2_pos, (1, j)))
        for j in range(SIMULATE_TEST_LEN):
            works.append(BallWork(j*CURVE_SHOWING_GAP, (2, j)))
        works = sorted(works, key = lambda x: x.timestamp)
        nowTimeStamp = 0
        nowWorkIndex = 0
        while len(works) > nowWorkIndex:
            if works[nowWorkIndex].timestamp > nowTimeStamp:
            #if True:
                p.stepSimulation()
                nowTimeStamp += stepTime
                if GUI:
                    time.sleep(stepTime)
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

        # save data
        for j in range(SINGLE_SIMULATE_SAMPLE_LEN) :
            dataStruct = dfo.DataStruct()
            cam1_end = random.randint(3, len(cam1_data))
            cam2_end = min(random.randint(cam1_end-2, cam1_end+2), len(cam2_data))
            dataStruct.inputs[0].camera_x = cam1_data[0].camera_pos.x
            dataStruct.inputs[0].camera_y = cam1_data[0].camera_pos.y
            dataStruct.inputs[0].camera_z = cam1_data[0].camera_pos.z
            dataStruct.inputs[1].camera_x = cam2_data[0].camera_pos.x
            dataStruct.inputs[1].camera_y = cam2_data[0].camera_pos.y
            dataStruct.inputs[1].camera_z = cam2_data[0].camera_pos.z
            dataStruct.inputs[0].seq_len = cam1_end
            dataStruct.inputs[1].seq_len = cam2_end

            for k in range(cam1_end):
                dataStruct.inputs[0].line_rad_xy[k] = cam1_data[k].rad_xy
                dataStruct.inputs[0].line_rad_xz[k] = cam1_data[k].rad_xz
                dataStruct.inputs[0].timestamps[k] = cam1_data[k].timestamp
            
            for k in range(cam2_end):
                dataStruct.inputs[1].line_rad_xy[k] = cam2_data[k].rad_xy
                dataStruct.inputs[1].line_rad_xz[k] = cam2_data[k].rad_xz
                dataStruct.inputs[1].timestamps[k] = cam2_data[k].timestamp
            
            for k in range(len(ans_data)):
                dataStruct.curvePoints[k].x = ans_data[k].ball_pos.x
                dataStruct.curvePoints[k].y = ans_data[k].ball_pos.y
                dataStruct.curvePoints[k].z = ans_data[k].ball_pos.z
                dataStruct.curveTimestamps[k] = ans_data[k].timestamp

            normer.norm(dataStruct)
            dataset.putData(i*SINGLE_SIMULATE_SAMPLE_LEN+j, dataStruct)
    dataset.saveToFile()

def work_simulate(queue:multiprocessing.Queue, dataLength):
    SINGLE_SIMULATE_SAMPLE_LEN = 5
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

    linearDamping = 6 * math.pi * radius * 0.0185
    angularDamping = 0.1
    p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)


    restitution = 1
    p.changeDynamics(sphere, -1, restitution=restitution)
    p.changeDynamics(plane, -1, restitution=restitution)
    p.changeDynamics(plane, -1, lateralFriction=0.6)
    p.changeDynamics(plane, -1, spinningFriction=0.6)
    p.changeDynamics(plane, -1, rollingFriction=0.6)
    
    p.changeDynamics(sphere, -1, lateralFriction=0.6)
    p.changeDynamics(sphere, -1, spinningFriction=0.6)
    p.changeDynamics(sphere, -1, rollingFriction=0.6)
    p.setRealTimeSimulation(0)
    p.setTimeStep(stepTime)

    p.setGravity(0, 0, -G)

    for i in range(int(dataLength//SINGLE_SIMULATE_SAMPLE_LEN)) :
        cam1_pos = randomCameraPos()
        cam2_pos = randomCameraPos()

        ball_pos = randomBallPos()

        #set ball pos
        p.resetBasePositionAndOrientation(sphere, [ball_pos.x, ball_pos.y, ball_pos.z], startOrientation)
        linearVelocity = [random.uniform(-5,5), random.uniform(-5,5), random.uniform(0, 3)]
        angularVelocity = [0, 0, 0]
        p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)


        works:List[Work] = []
        cam1_data:List[CameraWork] = []
        cam2_data:List[CameraWork] = []
        ans_data:List[BallWork] = []
        camera_systematic_error = random.normalvariate(0, SHUTTER_SYSTEMATIC_ERROR_STD)
        for j in range(SIMULATE_INPUT_LEN):
            works.append(CameraWork(abs(j/FPS + random.normalvariate(0,SHUTTER_RANDOM_ERROR_STD)), cam1_pos, (0,j)))
            works.append(CameraWork(abs(j/FPS + random.normalvariate(0,SHUTTER_RANDOM_ERROR_STD) + camera_systematic_error), cam2_pos, (1, j)))
        for j in range(SIMULATE_TEST_LEN):
            works.append(BallWork(j*CURVE_SHOWING_GAP, (2, j)))
        works = sorted(works, key = lambda x: x.timestamp)
        nowTimeStamp = 0

        nowTimeStamp = 0
        nowWorkIndex = 0
        while len(works) > nowWorkIndex:
            if works[nowWorkIndex].timestamp > nowTimeStamp:
            #if True:
                p.stepSimulation()
                nowTimeStamp += stepTime
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

        # save data
        num = SINGLE_SIMULATE_SAMPLE_LEN + dataLength % SINGLE_SIMULATE_SAMPLE_LEN if i == int(dataLength//SINGLE_SIMULATE_SAMPLE_LEN) - 1 else SINGLE_SIMULATE_SAMPLE_LEN
        bat = []
        for j in range(num) :
            dataStruct = dfo.DataStruct()
            cam1_end = random.randint(3, len(cam1_data))
            cam2_end = min(random.randint(cam1_end-2, cam1_end+2), len(cam2_data))
            dataStruct.inputs[0].camera_x = cam1_data[0].camera_pos.x
            dataStruct.inputs[0].camera_y = cam1_data[0].camera_pos.y
            dataStruct.inputs[0].camera_z = cam1_data[0].camera_pos.z
            dataStruct.inputs[1].camera_x = cam2_data[0].camera_pos.x
            dataStruct.inputs[1].camera_y = cam2_data[0].camera_pos.y
            dataStruct.inputs[1].camera_z = cam2_data[0].camera_pos.z
            dataStruct.inputs[0].seq_len = cam1_end
            dataStruct.inputs[1].seq_len = cam2_end

            for k in range(cam1_end):
                dataStruct.inputs[0].line_rad_xy[k] = cam1_data[k].rad_xy
                dataStruct.inputs[0].line_rad_xz[k] = cam1_data[k].rad_xz
                dataStruct.inputs[0].timestamps[k] = cam1_data[k].timestamp
            
            for k in range(cam2_end):
                dataStruct.inputs[1].line_rad_xy[k] = cam2_data[k].rad_xy
                dataStruct.inputs[1].line_rad_xz[k] = cam2_data[k].rad_xz
                dataStruct.inputs[1].timestamps[k] = cam2_data[k].timestamp
            
            for k in range(len(ans_data)):
                dataStruct.curvePoints[k].x = ans_data[k].ball_pos.x
                dataStruct.curvePoints[k].y = ans_data[k].ball_pos.y
                dataStruct.curvePoints[k].z = ans_data[k].ball_pos.z
                dataStruct.curveTimestamps[k] = ans_data[k].timestamp

            normer.norm(dataStruct)
            bat.append(dataStruct)
        queue.put(bat)

def work_putData(queue:multiprocessing.Queue, fileName, dataLength) :
    dataSet = dfo.BallDataSet_sync(fileName=fileName, dataLength=dataLength)
    with tqdm.tqdm(total=dataLength) as pbar:
        ind = 0
        while ind != dataLength :
            datas = queue.get()
            for data in datas:
                dataSet.putData(ind, data)
                ind += 1
            pbar.update(len(datas))
        dataSet.saveToFile()
        print("save to file : ", fileName)

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
    saver = multiprocessing.Process(target=work_putData, args=(queue, outputFileName, dataLength))
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
    argparser.add_argument("-l", default=10, type=int)
    argparser.add_argument("-n", default="train.bin")
    argparser.add_argument("--fast", default=False, action="store_true")

    #fetch params
    args = argparser.parse_args()
    if args.fast:
        simulate_fast(dataLength=args.l, num_workers=4, outputFileName="ball_simulate_v2/dataset/{}.train.bin".format(args.n))
        simulate_fast(dataLength=10000, num_workers=4, outputFileName="ball_simulate_v2/dataset/{}.valid.bin".format(args.n))
        exit()

    simulate(GUI=args.GUI, dataLength=args.l, outputFileName="ball_simulate_v2/dataset/{}.train.bin".format(args.n))
    simulate(GUI=args.GUI, dataLength=10000, outputFileName="ball_simulate_v2/dataset/{}.valid.bin".format(args.n))
    pass

