import time
import cv2
import numpy as np
from typing import Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import sys
sys.path.append(os.getcwd())
import ball_simulate_v2.dataFileOperator as dfo
import ball_simulate_v2.models as models
import ball_simulate_v2.train as train
import ball_predection.predict as predict
import core.Constants as c
import torch
import matplotlib.pyplot as plt
import robot_controll.controller as con


def sim_prediction_move(s=474) :
    r = con.Robot("")

    c.set2NormalB()
    ds = dfo.BallDataSet_sync("ball_simulate_v2/dataset/normalB.train.bin")
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0))
    m = models.ISEFWINNER_MEDIUM()
    m.cuda()
    m.load_state_dict(torch.load("ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt"))

    for i in range(s) :
        next(dl)
    X1, X1_len, X2, X2_len, T, Y = next(dl)
    Yl = Y.clone()
    c.normer.unnorm_ans_tensor(Yl)
    Yl = Yl.cpu()
    print(X1_len, X2_len)
    ti = 0
    for i in range(4, X2_len) :
        for j in range(i, i+2) :
            m.reset_hidden_cell(1)
            out = m(X1, torch.tensor([i], device="cuda:0"), X2, torch.tensor([j], device="cuda:0"), T)
            c.normer.unnorm_ans_tensor(out)
            X1_u = X1.clone()
            X2_u = X2.clone()
            c.normer.unorm_input_tensor(X1_u)
            c.normer.unorm_input_tensor(X2_u)
            hp, t = predict.getHitPointInformation(out)
            hp = hp.tolist()
            t = t.item()
            if hp is not None :
                #print(hp[1], hp[2], t)
                print("{:.3f} {:.3f} {:.3f}".format(hp[1], hp[2], t-ti))
                r.move(hp[1], hp[2])
            time.sleep(1/60)
            ti += 1/60
 
def sim_prediction() :
    video = cv2.VideoWriter("nor.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    c.set2NormalB()
    ds = dfo.BallDataSet_sync("ball_simulate_v2/dataset/normalB.train.bin")
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0))
    m = models.ISEFWINNER_MEDIUM()
    m.cuda()
    m.load_state_dict(torch.load("ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt"))

    for i in range(474) :
        next(dl)
    X1, X1_len, X2, X2_len, T, Y = next(dl)
    Yl = Y.clone()
    c.normer.unnorm_ans_tensor(Yl)
    Yl = Yl.cpu()
    print(X1_len, X2_len)
    fig, ax = createFigRoom()
    ti = 0
    for i in range(4, X2_len) :
        for j in range(i, i+2) :
            train.cleenRoom(ax)
            m.reset_hidden_cell(1)
            out = m(X1, torch.tensor([i], device="cuda:0"), X2, torch.tensor([j], device="cuda:0"), T)
            c.normer.unnorm_ans_tensor(out)
            X1_u = X1.clone()
            X2_u = X2.clone()
            c.normer.unorm_input_tensor(X1_u)
            c.normer.unorm_input_tensor(X2_u)
            hp, t = predict.getHitPointInformation(out)
            print(hp, t)
            pre = train.plotOutput(ax, out, color="green")
            l1, = train.drawLineSeq(ax, X1_u.cpu(), torch.tensor([i]), color="blue")
            l2, = train.drawLineSeq(ax, X2_u.cpu(), torch.tensor([j]), color="orange")
            hp_o = None
            if hp is not None :
                hp_o = train.plotOutput(ax, hp, color="black")
            ball = None
            #for i in range(c.SIMULATE_TEST_LEN) :
                #pass
                #if i * c.CURVE_SHOWING_GAP <= ti <= i * c.CURVE_SHOWING_GAP + c.CURVE_SHOWING_GAP:
                    #break
            ball = train.plotOutput(ax, Yl, color="red")
            hp_str = "None" if hp is None else "x: %.2f, y: %.2f, z: %.2f" % (hp[0], hp[1], hp[2])
            t_str = "None" if t is None else "%2f" % float(t - ti)
            ax.text(-4, -6, 0.2, "hit point: " + hp_str + "\ntime: " + t_str)
            plt.legend([l1, l2, pre, hp_o, ball], ["X1", "X2", "predict", "hit point", "real ball trajectory"])
            fig.canvas.draw()
            img = np.fromstring(ax.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)
        ti += 1/30
    video.release()

def createFigRoom()->Tuple[Figure, Axes]:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    train.configRoom(ax)
    return fig, ax

def find_seed() :
    ax = train.createRoom()
    c.set2NormalB()
    ds = dfo.BallDataSet_sync("ball_simulate_v2/dataset/normalB.train.bin")
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    ite = iter(dl)
    i = 0
    for i in range(354) :
        next(ite)
        i += 1
    for X1, X1_len, X2, X2_len, T, Y in ite:
        train.cleenRoom(ax)
        c.normer.unnorm_ans_tensor(Y)
        hp, t = predict.getHitPointInformation(Y)
        print(hp, t, i)
        train.plotOutput(ax, Y, color="green")
        plt.pause(0.01)
        input()
        for j in range(5) :
            next(ite)
            i += 1

        i += 1

sim_prediction()

#sim_prediction_move(354)
sim_prediction_move(474)
exit()

find_seed()
exit()

sim_prediction_move()
exit()

        
