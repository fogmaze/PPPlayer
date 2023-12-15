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
    for i in (range(round(cam_datas[-1][0]))) :
        ret, frame = origVideo.read()
        if cam_datas[0][1] == 1 :
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
        else :
            cam_datas.pop(0)

        if result is None :
            result = white

        fin = cv2.hconcat([frame, result])
        cv2.imshow('frame', fin)
        cv2.waitKey(1)
        outputVideo.write(fin)
    outputVideo.release()


def visualizeDetection(root, fps=30) :
    forcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(os.path.join(root, 'visualize.mp4')) :
        os.remove(os.path.join(root, 'visualize.mp4'))
    #outputVideo = cv2.VideoWriter(os.path.join(root, 'visualize.mp4'), forcc, fps, (640, 480))  

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

            #outputVideo.write(img)
            cv2.imshow('frame', img)
    #outputVideo.release()

def visualizePrediction_video(root, fps=30, lagg=10) :
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(os.path.join(root, 'visualize_video.mp4')) :
        os.remove(os.path.join(root, 'visualize_video.mp4'))
    outputVideo = cv2.VideoWriter(os.path.join(root, 'visualize_video.mp4'), fourcc, fps, (640*2, 480*2))
    cam1 = cv2.VideoCapture(os.path.join(root, 'cam1/all_tagged.mp4'))
    cam2 = cv2.VideoCapture(os.path.join(root, 'cam2/all_tagged.mp4'))

    lines1 = pred.LineCollector_hor()
    lines2 = pred.LineCollector_hor()

    fig, axe = createFigRoom()

    with open(os.path.join(root, 'pred.csv'), 'r') as f :
        reader = csv.reader(f)
        title = next(reader)
        pred_datas = [[float(b) for b in a] for a in reader]
        cam1_pred =  []
        cam2_pred =  []
        for pred_data in pred_datas :
            if pred_data[0] == 1:
                cam1_pred.append(pred_data)
            elif pred_data[0] == 2 :
                cam2_pred.append(pred_data)
            else :
                raise ValueError("pred_data[0] is not 1 or 2")

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

    white = np.zeros((480, 640, 3), np.uint8)
    white[:] = (255, 255, 255)
    ruframe = white
    rdframe = white

    s1,s2 = 0, 0

    for i in range(abs(lagg)) :
        if lagg < 0 :
            cam1.read()
            s1 += 1
        else :
            cam2.read()
            s2 += 1
    
    ind2 = s2
    for ind1 in tqdm.tqdm(range(s1, round(pred_datas[-1][1]))) :
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        cv2.putText(frame1, "cam1: {}".format(ind1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame2, "cam2: {}".format(ind2), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if not ret1 or not ret2 :
            break
        lframe = cv2.vconcat([frame1, frame2])
        isHit = True
        if cam1_pred[0][1] == ind1:
            line_data = cam1_dict[cam1_pred[0][1]][5:]
            isHit = not lines1.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
            which = 1
            pred_data = cam1_pred.pop(0)
        if cam2_pred[0][1] == ind2:
            line_data = cam2_dict[cam2_pred[0][1]][5:]
            isHit = not lines2.put(line_data[0], line_data[1], line_data[2], line_data[3], line_data[4])
            which = 2
            pred_data = cam2_pred.pop(0)
        if not isHit :
            cleanRoom(axe, (0, 90))
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
            ruframe = np.fromstring(axe.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            ruframe = ruframe.reshape(axe.figure.canvas.get_width_height()[::-1] + (3,))
            ruframe = cv2.cvtColor(ruframe, cv2.COLOR_RGB2BGR)

            cleanRoom(axe, (90, 90))
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
            rdframe = np.fromstring(axe.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            rdframe = rdframe.reshape(axe.figure.canvas.get_width_height()[::-1] + (3,))
            rdframe = cv2.cvtColor(rdframe, cv2.COLOR_RGB2BGR)
            if len(cam1_pred) == 0 or len(cam2_pred) == 0 :
                break
        ind2 += 1
        
        rframe = cv2.vconcat([ruframe, rdframe])
        fin = cv2.hconcat([lframe, rframe])
        cv2.imshow('frame', fin)
        cv2.waitKey(1)
        outputVideo.write(fin)

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

def configRoom(ax:Axes, ang=(50, 70)) -> Axes:
    lim = Constants.CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.invert_xaxis()
    ax.view_init(ang[0], ang[1])
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

def cleanRoom(axe:plt.Axes, ang=(50, 70)):
    axe.cla()
    configRoom(ax=axe, ang=ang)

def plotOutput(ax, out, color = 'r', label=None):
    o = out.view(-1,3)
    obj = None
    for p in o:
        obj = ax.scatter(p[0].item(),p[1].item(),p[2].item(),c=color, label=label)
    return obj



if __name__ == "__main__" :
    Constants.set2NormalB()
    #visualizeDetection_video("ball_detection/result/test")
    visualizePrediction_video("results/1215", lagg=6)
    exit()

    video = cv2.VideoCapture("results/1215/cam1/all_tagged.mp4")
    video_out = cv2.VideoWriter("results/1215/cam1/all_tagged_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    ind = 0
    while True :
        ret, frame = video.read()
        if not ret :
            break
        cv2.putText(frame, "{}".format(ind), (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        video_out.write(frame)
        ind += 1
    video.release()
    exit()
    #visualizePrediction("ball_detection/result/dual_default_105")