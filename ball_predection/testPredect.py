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

def sim_prediction() :
    c.set2NormalB()
    ds = dfo.BallDataSet_sync("ball_simulate_v2/dataset/normalB.train.bin")
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0))
    m = models.ISEFWINNER_MEDIUM()
    m.cuda()
    m.load_state_dict(torch.load("ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt"))

    for i in range(102) :
        next(dl)
    X1, X1_len, X2, X2_len, T, Y = next(dl)
    Yl =c.normer.unnorm_ans_tensor(Y)
    hp, t = predict.getHitPointInformation(Yl)
    print(X1_len, X2_len)
    print(hp, t)
    print("---")
    ax = train.createRoom()
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
            l1 = train.drawLineSeq(ax, X1_u.cpu(), torch.tensor([i]), color="blue")
            l2 = train.drawLineSeq(ax, X2_u.cpu(), torch.tensor([j]), color="red")
            plt.legend([l1, l2, pre], ["X1", "X2", "predict"])
            plt.pause(0.01)
            input()

def find_seed() :
    ax = train.createRoom()
    c.set2NormalB()
    ds = dfo.BallDataSet_sync("ball_simulate_v2/dataset/normalB.train.bin")
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    i = 0
    for X1, X1_len, X2, X2_len, T, Y in dl :
        train.cleenRoom(ax)
        c.normer.unnorm_ans_tensor(Y)
        hp, t = predict.getHitPointInformation(Y)
        print(hp, t, i)
        train.plotOutput(ax, Y, color="green")
        plt.pause(0.01)
        input()

        i += 1



        
sim_prediction()
exit()
