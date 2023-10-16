import tqdm
import numpy as np
from typing import Tuple
import csv
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import torch
import cv2
import sys
import os
sys.path.append(os.getcwd())
import core.Constants as Constants
import core.Equation3d as equ
import ball_predection.predict as pred

def prepareModelInput(ll:list, rl:list, device="cuda:0") :
    l = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    r = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    l[:len(ll)] = torch.tensor(ll, device=device)
    r[:len(rl)] = torch.tensor(rl, device=device)
    l = l.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    r = r.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    l_len = torch.tensor([len(ll)]).view(1,1).to(device)
    r_len = torch.tensor([len(rl)]).view(1,1).to(device)
    return l, l_len, r, r_len

def visualizeDetection_video(root, fps=30) :
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(os.path.join(root, 'visualize.mp4')) :
        os.remove(os.path.join(root, 'visualize.mp4'))
    outputVideo = cv2.VideoWriter(os.path.join(root, 'visualize.mp4'), forcc, fps, (640*2, 480))
    origVideo = cv2.VideoCapture(os.path.join(root, 'all_tagged.mp4'))

    lines = pred.LineCollector_hor()
    fig, axe = createFigRoom()

    white = np.zeros((480, 640, 3), np.uint8)
    white[:] = (255, 255, 255)

    with open(os.path.join(root, 'detection.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        cam_datas = list(reader)
        # cast to float
        cam_datas = [[float(b) for b in a] for a in cam_datas]
    
    result = None
    for i in tqdm.tqdm(range(round(cam_datas[-1][0]))) :
        ret, frame = origVideo.read()
        if i == cam_datas[0][0] :
            cleanRoom(axe)
            line_data = cam_datas[0][6:]
            isHit = not lines.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
            if isHit :
                pass
            else :
                l1, = displayLines(axe, lines, color='b', label=None)
                axe.scatter(line_data[0], line_data[1], line_data[2], c='r', label=None)
                plt.legend([l1], ['cam'])
                fig.canvas.draw()
                img = np.fromstring(axe.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(axe.figure.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                result = img
            cam_datas.pop(0)

        if result is None :
            result = white

        fin = cv2.hconcat([frame, result])
        outputVideo.write(fin)
    outputVideo.release()


def visualizeDetection(root, fps=30) :
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(os.path.join(root, 'visualize.mp4')) :
        os.remove(os.path.join(root, 'visualize.mp4'))
    outputVideo = cv2.VideoWriter(os.path.join(root, 'visualize.mp4'), forcc, fps, (640, 480))  

    lines = pred.LineCollector_hor()
    fig, axe = createFigRoom()

    with open(os.path.join(root, 'detection.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        cam_datas = list(reader)
        # cast to float
        cam_datas = [[float(b) for b in a] for a in cam_datas]
    
    for cam_data in tqdm.tqdm(cam_datas) :
        cleanRoom(axe)
        line_data = cam_data[6:]
        isHit = not lines.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
        if isHit :
            pass
        else :
            l1, = displayLines(axe, lines, color='b', label=None)
            plt.legend([l1], ['cam'])
            fig.canvas.draw()
            img = np.fromstring(axe.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(axe.figure.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            outputVideo.write(img)
    outputVideo.release()


def visualizePrediction(root, fps=30) :
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(os.path.join(root, 'visualize.mp4')) :
        os.remove(os.path.join(root, 'visualize.mp4'))
    outputVideo = cv2.VideoWriter(os.path.join(root, 'visualize.mp4'), fourcc, fps, (640, 480))

    lines1 = pred.LineCollector_hor()
    lines2 = pred.LineCollector_hor()

    fig, axe = createFigRoom()

    with open(os.path.join(root, 'pred.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        pred_datas = [[float(b) for b in a] for a in reader]

    with open(os.path.join(root, 'cam1/detection.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        cam1_datas = list(reader)
        cam1_dict = {}
        for data in cam1_datas :
            cam1_dict[int(data[0])] = [float(a) for a in data[1:]]

    with open(os.path.join(root, 'cam2/detection.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        cam2_datas = list(reader)
        cam2_dict = {}
        for data in cam2_datas :
            cam2_dict[int(data[0])] = [float(a) for a in data[1:]]

    for pred_data in tqdm.tqdm(pred_datas) :
        cleanRoom(axe)
        if pred_data[0] == 1 :
            line_data = cam1_dict[pred_data[1]][5:]
            isHit = not lines1.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
            which = 1
        else :
            line_data = cam2_dict[pred_data[1]][5:]
            isHit = not lines2.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
            which = 2
        if isHit :
            pass
        else :
            out:torch.Tensor = torch.tensor(pred_data[6:]).view(-1,3)
            o = plotOutput(axe, out, color='r', label=None)
            leg = [(o, 'output')]
            if len(lines1.lines) > 0 :
                l1, = displayLines(axe, lines1, color='b', label=None)
                leg.append((l1, 'cam1'))
            if len(lines2.lines) > 0 :
                l2, = displayLines(axe, lines2, color='g', label=None)
                leg.append((l2, 'cam2'))
            plt.legend(*zip(*leg))
            fig.canvas.draw()
            img = np.fromstring(axe.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(axe.figure.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            outputVideo.write(img)
    outputVideo.release()

def displayLines(axe, lines, color="b", label=None) :
    r = None
    for line in lines.lines :
        l = equ.LineEquation3d(None, None)
        cp = equ.Point3d(line[0], line[1], line[2])
        l.setByPointOblique(cp, line[3], line[4])
        r = drawLine3d(axe, l, color=color, label=label)
    return r

def configRoom(ax:Axes) -> Axes:
    lim = Constants.CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.invert_xaxis()
    ax.view_init(50, 70)
    W = 2.74/2
    H = 1.525/2
    Z = np.array([
        [-W, -H, 0],
        [W, -H, 0],
        [W, H, 0],
        [-W, H, 0],
    ])
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])
    verts = [
        [Z[0],Z[1],Z[2],Z[3]],
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
    return ax

def drawLine3d(axe:plt.Axes,line:equ.LineEquation3d, color="r", label=None):
    points = [line.getPoint({'x':-Constants.BALL_AREA_HALF_LENGTH*1}),line.getPoint({'x':Constants.BALL_AREA_HALF_LENGTH*1})]
    X = [points[0][0],points[1][0]]
    Y = [points[0][1],points[1][1]]
    Z = [points[0][2],points[1][2]]
    return axe.plot(X,Y,Z, color=color, label=label)

def createFigRoom()->Tuple[Figure, Axes]:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    configRoom(ax)
    return fig, ax

def cleanRoom(axe:plt.Axes):
    axe.cla()
    configRoom(ax=axe)

def plotOutput(ax, out, color = 'r', label=None):
    o = out.view(-1,3)
    obj = None
    for p in o:
        obj = ax.scatter(p[0].item(),p[1].item(),p[2].item(),c=color, label=label)
    return obj



if __name__ == "__main__" :
    Constants.set2NormalB()
    visualizeDetection_video("ball_detection/result/c1_40")
    #visualizePrediction("ball_detection/result/dual_default_105")