import numpy as np
import random
import sys
import os
sys.path.append(os.getcwd())
import ball_simulate.v4.dataFileOperator as dfo
import ball_simulate.v4.models as models
from ball_simulate.v4.models import MODEL_MAP
from argparse import ArgumentParser
import logging
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
import ball_simulate.v4.Constants as c
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import core.Equation3d as equ
from typing import List,Tuple,Dict
import tqdm
import csv


def train(
        epochs = 100, 
        batch_size =16,
        scheduler_step_size=None, 
        LR = 0.001, 
        momentum=0.01, 
        dataset = "", 
        opt="adam",
        model_name = "small", 
        name="default", 
        weight = None, 
        device = "cuda:0", 
        num_workers=2, 
        mode="normalBR", 
        train_ratio=0.8, 
        valid_ratio=0.1, 
        test_ratio=0.1, 
        lossFN = "mse"
    ):
    torch.multiprocessing.set_start_method('spawn')
    #model_save_dir = time.strftime("./ball_simulate/v2/model_saves/" + name + "%Y-%m-%d_%H-%M-%S-"+ model_name +"/",time.localtime())
    model_save_dir = "./ball_simulate/v4/model_saves/" + name + "/"
    if os.path.isdir(model_save_dir):
        old_new_name = "./ball_simulate/v4/model_saves/" + name + "_" + str(random.randint(0,1000)) + "/"
        while os.path.isdir(old_new_name):
            old_new_name = "./ball_simulate/v4/model_saves/" + name + "_" + str(random.randint(0,1000)) + "/"
        os.rename(model_save_dir, old_new_name)
        print("model save dir exists, change the old one into " + old_new_name)
    os.makedirs(model_save_dir)
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(model_save_dir, 'training.log'))
    format_log = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(format_log)
    file_handler.setFormatter(format_log)
    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

    if opt == "adam":
        training_params = 'optimizer: adam, epochs:{}, batch_size:{}, LR:{}, dataset:{}, model_name:{}, weight:{}, device:{}'.format(epochs, batch_size, LR, dataset, model_name, weight, device)
    elif opt == "sgdm":
        training_params = 'optimizer: sgdm, epochs:{}, batch_size:{}, LR:{}, momentum:{}, dataset:{}, model_name:{}, weight:{}, device:{}'.format(epochs, batch_size, LR, momentum, dataset, model_name, weight, device)
    train_logger.info('start training with args: {}'.format(training_params))

    if (MODEL_MAP.get(model_name) == None):
        raise Exception("model name not found")
    model:models.ISEFWINNER_BASE = MODEL_MAP[model_name](device=device)

    if weight:
        try:
            train_logger.info('loading: ' + weight) 
            model.load_state_dict(torch.load(weight))
        except:
            train_logger.error('cannot load model ')

    if lossFN == "mse":
        criterion = nn.MSELoss().to(device=device)
    elif lossFN == "distance":
        criterion = models.DistanceLoss().to(device=device)
    model.to(device=device)
    #optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    if opt == "adam":
        optimizer = torch.optim.RAdam(model.parameters(), lr = LR)
    elif opt == "sgdm":
        optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=momentum)
        print("sgdm!!")
    if scheduler_step_size == None:
        scheduler = None
    else :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,scheduler_step_size,0.1)

    ball_datas = dfo.BallDataSet_sync(os.path.join("./ball_simulate/v4/dataset/", dataset + ".train.bin"), device=device, mode=mode)
    train_len = int(len(ball_datas) * train_ratio)
    valid_len = int(len(ball_datas) * valid_ratio)
    test_len = len(ball_datas) - train_len - valid_len
    train_datas, valid_datas, test_datas = random_split(ball_datas, [train_len, valid_len, test_len])
    dataloader_train = DataLoader(dataset=train_datas, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_valid = DataLoader(dataset=valid_datas, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_test = DataLoader(dataset=test_datas, batch_size=batch_size, num_workers=num_workers, shuffle=True)


    train_loss = 0
    valid_loss = 0

    train_loss_history = []
    valid_loss_history = []
    
    min_validloss = 0
    for e in range(epochs):
        try:
            torch.cuda.empty_cache()

            model.train()
            n = 0
            trainloss_sum = 0
            trainingloss = 0
            validloss_sum = 0
            validationloss = 0
            #data_sample = splitTrainData_batch([createTrainData() for _ in range(batch_size)],normalized=True)
            for data in tqdm.tqdm(dataloader_train):
                train_loss = model.fit_one_iteration(data, optimizer, criterion)
                trainloss_sum += train_loss
                if n % 1000 == 999:
                    trainingloss = trainloss_sum / 1000
                    trainloss_sum = 0
                n += 1

            torch.cuda.empty_cache()
            
            n = 0
            model.eval()
            for data  in tqdm.tqdm(dataloader_valid):
                valid_loss = model.validate_one_iteration(data, criterion)
                validloss_sum += valid_loss
                n += 1
            
            validationloss = validloss_sum / n
            train_loss_history.append(trainingloss)
            valid_loss_history.append(validationloss)

            print("==========================[epoch:" + str(e) + "]==============================")
            train_logger.info("epoch:{}\tlr:{:e}\ttraining loss:{:0.10f}\tvalidation loss:{:0.10f}".format(e,(optimizer.param_groups[0]['lr']), trainingloss, validationloss))

            print("save model")
            if not os.path.isdir(model_save_dir):
                os.makedirs(model_save_dir)
            dirsavename = model_save_dir + "epoch_" + str(e) + "/"
            os.makedirs(dirsavename)
            torch.save(model.state_dict(), dirsavename + "weight.pt")
            try:
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output1.png", seed=1)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output2.png", seed=100)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output3.png", seed=200)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output4.png", seed=300)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output5.png", seed=400)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output6.png", seed=500)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output7.png", seed=600)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output8.png", seed=700)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output9.png", seed=800)
                saveVisualizeModelOutput(model, valid_datas, dirsavename + "output10.png", seed=900)
            except Exception as e:
                print(e)
                pass
            model.reset_hidden_cell(batch_size=batch_size)

            if scheduler != None :
                scheduler.step()
        except KeyboardInterrupt :
            c_exit = input("exit?[Y/n]")
            if c_exit == "Y" or c_exit == "y" or c_exit == chr(13) :
                break

    plt.cla()
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    plt.legend(['train_loss', 'valid_loss'], loc='upper left')
    plt.savefig(model_save_dir + "loss.png")
    # save data to csv file
    with open(model_save_dir + "loss.csv", "w") as f:
        writer = csv.writer(f)
        #write params
        writer.writerow([training_params])
        # write the header
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        # write the data
        for i in range(len(train_loss_history)):
            writer.writerow([i, train_loss_history[i], valid_loss_history[i]])
    train_logger.info("training finished")
    train_logger.removeHandler(file_handler)
    train_logger.removeHandler(console_handler)


def exportModel(model_name:str, weight:str):
    model = MODEL_MAP[model_name](device='cpu')
    print("exporting: " + weight)
    model.load_state_dict(torch.load(weight))
    model.eval()
    model_script = torch.jit.script(model)
    model_script.save('model_final.pth')


def validModel(model_name, weight, dataset, batch_size=64) :
    model = MODEL_MAP[model_name](device='cpu')
    print("validating: " + weight)
    model.load_state_dict(torch.load(weight))
    model.eval()
    criterion = nn.MSELoss()
    ball_datas = dfo.BallDataSet_sync(os.path.join("./ball_simulate/v2/dataset/", dataset + ".valid.bin"), device='cpu')
    loader = DataLoader(ball_datas, batch_size=batch_size)
    loss_sum = 0
    i = 0
    for r, r_len, l, l_len, t, ans in tqdm.tqdm(loader):
        model.reset_hidden_cell(batch_size=t.shape[0])
        out = model(r, r_len, l, l_len, t)
        loss = criterion(out, ans)
        loss_sum += loss.item()
        i += 1
    print("loss: " + str(loss_sum / i))
    return loss_sum / i

def configRoom(ax:Axes):
    lim = c.CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def drawLine3d(axe:plt.Axes,line:equ.LineEquation3d, color="r", label=None):
    points = [line.getPoint({'x':-c.BALL_AREA_HALF_LENGTH*10}),line.getPoint({'x':c.BALL_AREA_HALF_LENGTH*10})]
    X = [points[0][0],points[1][0]]
    Y = [points[0][1],points[1][1]]
    Z = [points[0][2],points[1][2]]
    return axe.plot(X,Y,Z, color=color, label=label)

def createRoom()->Axes:
    ax = plt.axes(projection='3d')
    configRoom(ax)
    return ax

def cleenRoom(axe:plt.Axes):
    axe.cla()
    configRoom(ax=axe)

def plotOutput(ax, out, color = 'r', label=None):
    o = out.view(-1,3)
    obj = None
    for p in o:
        obj = ax.scatter(p[0].item(),p[1].item(),p[2].item(),c=color, label=label)
    return obj

def saveVisualizeTrainData(dataset_name, mode, seed=3) :
    dataset = dfo.BallDataSet_sync(os.path.join("./ball_simulate/v4/dataset/", dataset_name + ".train.bin"), mode=mode)
    r, r_len, l, l_len, ans = dataset[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    ans = ans.view(1, -1, 3)
    ans = ans.view(-1, 3).cpu()
    r = r.view(-1, 5).cpu()
    l = l.view(-1, 5).cpu()
    r_len = r_len.view(1).cpu()
    l_len = l_len.view(1).cpu()


    c.normer.unnorm_ans_tensor(ans)
    c.normer.unorm_input_tensor(r)
    c.normer.unorm_input_tensor(l)

    ax = createRoom()
    line_ans = plotOutput(ax, ans, color='b')

    seq_r, = drawLineSeq(ax, r, r_len, color='g')
    seq_l, = drawLineSeq(ax, l, l_len, color='y')

    # add legend
    ax.legend([line_ans, seq_r, seq_l], ["trajectory", "right", "left"])

    #plt.savefig(imgFileName)
    plt.show(block=True)
    

def saveVisualizeModelOutput(model:models.ISEFWINNER_BASE, dataset, imgFileName, seed = 3):
    model.eval()
    criterion = models.DistanceLoss()
    model.reset_hidden_cell(batch_size=1)

    r, r_len, l, l_len, ans = dataset[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    ans = ans.view(1, -1, 3)
    out = model(r, r_len, l, l_len, model.training_t.view(1, -1)).view(-1, 3).cpu()
    ans = ans.view(-1, 3).cpu()
    r = r.view(-1, 5).cpu()
    l = l.view(-1, 5).cpu()
    r_len = r_len.view(1).cpu()
    l_len = l_len.view(1).cpu()

    loss = criterion(out, ans).item()

    c.normer.unnorm_ans_tensor(ans)
    c.normer.unnorm_ans_tensor(out)
    c.normer.unorm_input_tensor(r)
    c.normer.unorm_input_tensor(l)

    ax = createRoom()
    line_out = plotOutput(ax, out, color='r')
    line_ans = plotOutput(ax, ans, color='b')

    seq_r, = drawLineSeq(ax, r, r_len, color='g')
    seq_l, = drawLineSeq(ax, l, l_len, color='y')

    # add legend
    ax.legend([line_out, line_ans, seq_r, seq_l], ["output", "answer", "right", "left"])
    plt.gcf().text(0.02, 0.02,s="Dist loss: " + str(loss), size=15)

    plt.savefig(imgFileName)
    plt.close()

def drawLineSeq(axe:plt.Axes, seq:torch.Tensor, seq_len:torch.Tensor, color="r") :
    lines_t = seq.view(-1, 5)
    obj = None
    try :        
        for i in range(seq_len.view(1)[0]) :
            line = equ.LineEquation3d(None, None)
            line.setByPointOblique(equ.Point3d(lines_t[i][0], lines_t[i][1], lines_t[i][2]), lines_t[i][3], lines_t[i][4])
            obj = drawLine3d(axe, line, color=color)
    except :
        pass
    return obj

def visualizeModelOutput(model_name, weight, seed = 3):
    model = MODEL_MAP[model_name](device='cpu')

    model.load_state_dict(torch.load(weight))
    model.eval()

    criterion = nn.MSELoss()
    ball_datas = dfo.BallDataSet("ball_simulate/v2/dataset/medium_pred.valid.bin", device='cpu')

    r, r_len, l, l_len, t, ans = ball_datas[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    t = t.view(1, -1)
    out = model(r, r_len, l, l_len, t).view(-1, 3)
    ans = ans.view(1, -1, 3)
    ans = ans.view(-1, 3)
    r = r.view(-1, 5)
    l = l.view(-1, 5)

    loss = criterion(out, ans).item()

    c.normer.unnorm_ans_tensor(ans)
    c.normer.unnorm_ans_tensor(out)
    c.normer.unorm_input_tensor(r)
    c.normer.unorm_input_tensor(l)

    ax = createRoom()
    line_out = plotOutput(ax, out, color='r')
    line_ans = plotOutput(ax, ans, color='b')

    seq_r, = drawLineSeq(ax, r, r_len, color='g')
    seq_l, = drawLineSeq(ax, l, l_len, color='y')

    # add legend
    ax.legend([line_out, line_ans, seq_r, seq_l], ["output", "answer", "right", "left"])
    plt.gcf().text(0.02, 0.02,s="MSE loss: " + str(loss), size=15)

    plt.show()
    
def redrawTrainResult(dirname, model_name, dataset):
    model = MODEL_MAP[model_name](device='cpu')
    ball_datas = dfo.BallDataSet("ball_simulate/v2/dataset/" + dataset + ".valid.bin", device='cpu')
    for i in tqdm.tqdm(range(0, 30)) :
        model.load_state_dict(torch.load(dirname + "epoch_" + str(i) + "/weight.pt"))
        saveVisualizeModelOutput(model, ball_datas, dirname + "epoch_" + str(i) + "/output1.png", seed=1)
        saveVisualizeModelOutput(model, ball_datas, dirname + "epoch_" + str(i) + "/output2.png", seed=100)
        saveVisualizeModelOutput(model, ball_datas, dirname + "epoch_" + str(i) + "/output3.png", seed=200)
        saveVisualizeModelOutput(model, ball_datas, dirname + "epoch_" + str(i) + "/output4.png", seed=300)
        saveVisualizeModelOutput(model, ball_datas, dirname + "epoch_" + str(i) + "/output5.png", seed=400)


def cross():
    with open("ball_simulate/v2/cross.csv", "w") as f:
        writer = csv.writer(f)
        for d in ('fit', 'ne', 'predict') :
            if d == 'fit':
                c.set2Fitting()
            elif d == 'ne':
                c.set2NoError()
            elif d == 'predict':
                c.set2Predict()
            res = []
            for w in ('fit', 'ne', 'predict') :
                res.append(validModel("medium", "ball_simulate/v2/model_saves/" + w  + "/epoch_29/weight.pt", d + "_medium"))
            writer.writerow(res)

def LRRTest(name, bs = 64, lr_range = (0.001, 0.5), opt = "adam") :
    c.set2Normal()
    for lr in  np.arange(lr_range[0], lr_range[1], (lr_range[1]-lr_range[0])/8 ):
        train(epochs=10, batch_size=bs, LR=lr, dataset="normal_medium", opt=opt, model_name="medium", name=name + "_lr_test_" + str(lr), num_workers=0)

if __name__ == "__main__" :
    argparser = ArgumentParser()
    argparser.add_argument('-lr', default=0.001, type=float)
    argparser.add_argument('-b', default=64, type=int)
    argparser.add_argument('-e', default=30, type=int)
    argparser.add_argument('-m', default="medium", type=str)
    argparser.add_argument('-d', default="test_short4", type=str)
    argparser.add_argument('-n', default="test", type=str)
    argparser.add_argument('--mode', default="short4", type=str)
    argparser.add_argument('--num_workers', default=0, type=int)
    argparser.add_argument('-o', default="adam")
    argparser.add_argument('-l', default="mse", type=str)
    argparser.add_argument('-w', default=None, type=str)
    argparser.add_argument('-mom', default=0.01, type=float)
    argparser.add_argument('-s', default=0, type=int)
    argparser.add_argument('--test', dest='test', action='store_true', default=False)
    argparser.add_argument('--LRRTest', dest='LRRTest', action='store_true', default=False)
    argparser.add_argument('--export-model', dest='export', action='store_true', default=False)

    args = argparser.parse_args()
    # if ball_simulate/v2/dataset not exested, then create

    if args.LRRTest :
        LRRTest(bs=args.b, name=args.n, opt=args.o)
        exit(0)

    if args.mode != "default":
        if args.mode == "normalBR4" :
            c.set2NormalBR4()
        elif args.mode == "better4" :
            c.set2Better4()
        elif args.mode == "short4" :
            c.set2Short4()
        else :
            raise Exception("mode not found")

    if not os.path.exists("ball_simulate/v4/dataset"):
        os.mkdir("ball_simulate/v4/dataset")
    if args.export:
        exit(0)
    if args.test:
        validModel(args.m, args.w, args.d)
        exit(0)
        
    train(
        scheduler_step_size=None if args.s == 0 else args.s,
        LR=args.lr,
        momentum=args.mom,
        batch_size=args.b,
        epochs=args.e, 
        dataset=args.d, 
        model_name=args.m, 
        weight=args.w, 
        name=args.n, 
        num_workers=args.num_workers, 
        opt=args.o, 
        mode=args.mode,
        lossFN=args.l
    )
    pass