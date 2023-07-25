import random
import sys
import os
sys.path.append(os.getcwd())
import ball_simulate_v2.dataFileOperator as dfo
from argparse import ArgumentParser
import logging
import time
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import core.Constants as c
import core
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import core.Equation3d as equ
from typing import List,Tuple
import tqdm
import csv



class ISEFWINNER_BASE(nn.Module):
    #input:  [cam_x, cam_y, cam_z, rad_xy, rad_xz] * 2 , [time]
    #output: [x, y, z]
    device = "cuda:0"
    input_size:int = 5
    output_size:int = 3
    mlp1_out:int
    mlp2_out:int
    lstm_out:int
    lstm_num_layers:int
    mlp1:nn.Sequential
    lstm:nn.LSTM
    llstm_hidden_cell:tuple
    rlstm_hidden_cell:tuple
    mlp2:nn.Sequential

    @torch.jit.export
    def reset_hidden_cell(self, batch_size:int):
        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_out, device=self.device), torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_out, device=self.device), torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))

    #input shape:(batch_size, seq_len, input_size)
    def forward(self, X1:torch.Tensor, X1_len:torch.Tensor, X2:torch.Tensor, X2_len:torch.Tensor, T:torch.Tensor):
            
        x1_batch_size = len(X1)
        x2_batch_size = len(X2)

        X1 = self.mlp1(X1.view(-1,self.input_size)).view(x1_batch_size, -1, self.mlp1_out)
        X2 = self.mlp1(X2.view(-1,self.input_size)).view(x2_batch_size, -1, self.mlp1_out)
        #shape: (batch_size, seq_len, input_size)

        X1 = X1.transpose(0, 1)
        X2 = X2.transpose(0, 1)
        #shape: (seq_len, batch_size, input_size)

        X1_seq, self.llstm_hidden_cell = self.lstm(X1, self.llstm_hidden_cell)
        X2_seq, self.rlstm_hidden_cell = self.lstm(X2, self.rlstm_hidden_cell)
        #shape: (seq_len, batch_size, input_size)

        X1_len_ind = X1_len - 1
        X2_len_ind = X2_len - 1

        X1_ind = X1_len_ind.view(1, x1_batch_size, 1).expand(1, x1_batch_size, self.lstm_out)
        X2_ind = X2_len_ind.view(1, x2_batch_size, 1).expand(1, x2_batch_size, self.lstm_out)

        X1 = X1_seq.gather(0, X1_ind).view(1, x1_batch_size, self.lstm_out)
        X2 = X2_seq.gather(0, X2_ind).view(1, x2_batch_size, self.lstm_out)
        
        #shape: (1, batch_size, input_size)
        X1 = X1.transpose(0, 1)
        X2 = X2.transpose(0, 1)
        #shape: (batch_size, 1, input_size)

        X = torch.cat((X1, X2), 2)

        #shape of T: batch_size, out_seq_len
        X = X.repeat(1, T.shape[1], 1)
        #shape: (batch_size, out_seq_len, input_size)

        X = torch.cat((X, T.view(x1_batch_size, T.shape[1], 1)), 2)
        #shape: (batch_size, out_seq_len, input_size + 1)

        res = self.mlp2(X.view(-1, self.lstm_out * 2 + 1)).view(x1_batch_size, T.shape[1], self.output_size)

        # out shape : (batch_size, seq_len, output_size)
        return res

class ISEFWINNER_SMALL(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device

        mlp1_l1_out = 50
        mlp1_l2_out = 40
        mlp1_l3_out = 40
        mlp1_l4_out = 15

        mlp2_l1_out = 45
        mlp2_l2_out = 30
        mlp2_l3_out = 20

        self.mlp1_out = mlp1_l4_out
        self.mlp2_out = mlp2_l3_out
        self.lstm_out = 40
        self.lstm_num_layers = 4

        batch_size = 1
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_size,mlp1_l1_out),
            nn.ReLU(),
            nn.Linear(mlp1_l1_out,mlp1_l2_out),
            nn.Tanh(),
            nn.Linear(mlp1_l2_out,mlp1_l3_out),
            nn.Tanh(),
            nn.Linear(mlp1_l3_out,mlp1_l4_out),
        )
        self.lstm = nn.LSTM(input_size = mlp1_l4_out,hidden_size= self.lstm_out,num_layers = self.lstm_num_layers)

        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        
        self.mlp2 = nn.Sequential(
            nn.Linear(self.lstm_out * 2 + 1, mlp2_l1_out),
            nn.ReLU(),
            nn.Linear(mlp2_l1_out, mlp2_l2_out),
            nn.Tanh(),
            nn.Linear(mlp2_l2_out, mlp2_l3_out),
            nn.Tanh(),
            nn.Linear(mlp2_l3_out, self.output_size)
        )

class ISEFWINNER_MEDIUM(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device

        mlp1_l1_out = 100
        mlp1_l2_out = 80
        mlp1_l3_out = 80
        mlp1_l4_out = 50

        mlp2_l1_out = 90
        mlp2_l2_out = 60
        mlp2_l3_out = 50
        mlp2_l4_out = 30

        self.mlp1_out = mlp1_l4_out
        self.mlp2_out = mlp2_l4_out
        self.lstm_out = 60
        self.lstm_num_layers = 6

        batch_size = 1
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_size,mlp1_l1_out),
            nn.ReLU(),
            nn.Linear(mlp1_l1_out,mlp1_l2_out),
            nn.Tanh(),
            nn.Linear(mlp1_l2_out,mlp1_l3_out),
            nn.Tanh(),
            nn.Linear(mlp1_l3_out,mlp1_l4_out),
        )
        self.lstm = nn.LSTM(input_size = mlp1_l4_out,hidden_size= self.lstm_out,num_layers = self.lstm_num_layers)

        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        
        self.mlp2 = nn.Sequential(
            nn.Linear(self.lstm_out * 2 + 1, mlp2_l1_out),
            nn.ReLU(),
            nn.Linear(mlp2_l1_out, mlp2_l2_out),
            nn.Tanh(),
            nn.Linear(mlp2_l2_out, mlp2_l3_out),
            nn.Tanh(),
            nn.Linear(mlp2_l3_out, mlp2_l4_out),
            nn.Tanh(),
            nn.Linear(mlp2_l4_out, self.output_size)
        )

class ISEFWINNER_LARGE(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device

        mlp1_l1_out = 200
        mlp1_l2_out = 150
        mlp1_l3_out = 150
        mlp1_l4_out = 90
        mlp1_l5_out = 70

        mlp2_l1_out = 130
        mlp2_l2_out = 120
        mlp2_l3_out = 100
        mlp2_l4_out = 60
        mlp2_l5_out = 60


        self.mlp1_out = mlp1_l5_out
        self.mlp2_out = self.output_size
        self.lstm_out = 60
        self.lstm_num_layers = 8

        batch_size = 1
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_size,mlp1_l1_out),
            nn.ReLU(),
            nn.Linear(mlp1_l1_out, mlp1_l2_out),
            nn.Tanh(),
            nn.Linear(mlp1_l2_out, mlp1_l3_out),
            nn.Tanh(),
            nn.Linear(mlp1_l3_out, mlp1_l4_out),
            nn.Tanh(),
            nn.Linear(mlp1_l4_out, mlp1_l5_out)
        )
        self.lstm = nn.LSTM(input_size = self.mlp1_out, hidden_size=self.lstm_out, num_layers = self.lstm_num_layers)

        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        
        self.mlp2 = nn.Sequential(
            nn.Linear(self.lstm_out * 2 + 1, mlp2_l1_out),
            nn.ReLU(),
            nn.Linear(mlp2_l1_out, mlp2_l2_out),
            nn.Tanh(),
            nn.Linear(mlp2_l2_out, mlp2_l3_out),
            nn.Tanh(),
            nn.Linear(mlp2_l3_out, mlp2_l4_out),
            nn.Tanh(),
            nn.Linear(mlp2_l4_out, mlp2_l5_out),
            nn.Tanh(),
            nn.Linear(mlp2_l5_out, self.output_size)
        )

def train(epochs = 100, batch_size =16,scheduler_step_size=7, LR = 0.0001, dataset = "",model_name = "small", name="default", weight = None, device = "cuda:0", num_workers=2):
    model_save_dir = time.strftime("./ball_simulate_v2/model_saves/" + name + "%Y-%m-%d_%H-%M-%S-"+ model_name +"/",time.localtime())
    os.makedirs(model_save_dir)
    torch.multiprocessing.set_start_method('spawn')
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

    training_params = 'epochs:{}, batch_size:{}, scheduler_step_size:{}, LR:{}, dataset:{}, model_name:{}, weight:{}, device:{}'.format(epochs,batch_size,scheduler_step_size,LR,dataset,model_name,weight,device)
    train_logger.info('start training with args: epochs:{}, batch_size:{}, scheduler_step_size:{}, LR:{}, dataset:{}, model_name:{}, weight:{}, device:{}'.format(epochs,batch_size,scheduler_step_size,LR,dataset,model_name,weight,device))
    
    if (MODEL_MAP.get(model_name) == None):
        raise Exception("model name not found")
    model = MODEL_MAP[model_name](device=device)

    if weight:
        try:
            train_logger.info('loading: ' + weight) 
            model.load_state_dict(torch.load(weight))
        except:
            train_logger.error('cannot load model ')

    criterion = nn.MSELoss().cuda()
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,scheduler_step_size,0.1)

    ball_datas_train = dfo.BallDataSet_sync(os.path.join("./ball_simulate_v2/dataset/", dataset + ".train.bin"), device=device)
    dataloader_train = DataLoader(dataset=ball_datas_train, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    ball_datas_valid = dfo.BallDataSet_sync(os.path.join("./ball_simulate_v2/dataset/", dataset + ".valid.bin"), device=device)
    dataloader_valid = DataLoader(dataset=ball_datas_valid, batch_size=batch_size)

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
            validloss_sum = 0
            
            #data_sample = splitTrainData_batch([createTrainData() for _ in range(batch_size)],normalized=True)
            for r, r_len, l, l_len, t, ans in tqdm.tqdm(dataloader_train):
                model.reset_hidden_cell(batch_size=r.shape[0])
                optimizer.zero_grad()
                out = model(r, r_len, l, l_len, t)
                train_loss = criterion(out, ans)
                train_loss.backward()
                trainloss_sum += train_loss.item()
                optimizer.step()
                n += 1

            torch.cuda.empty_cache()
            
            model.eval()
            for r, r_len, l, l_len, t, ans in tqdm.tqdm(dataloader_valid):
                model.reset_hidden_cell(batch_size=ans.shape[0])
                out = model(r, r_len, l, l_len, t)
                valid_loss = criterion(out, ans)
                validloss_sum += valid_loss.item()
            
            real_trainingloss = trainloss_sum / len(dataloader_train.dataset) * batch_size
            real_validationloss = validloss_sum / len(dataloader_valid.dataset) * batch_size
            train_loss_history.append(real_trainingloss)
            valid_loss_history.append(real_validationloss)

            print("==========================[epoch:" + str(e) + "]==============================")
            train_logger.info("epoch:{}\tlr:{:e}\ttraining loss:{:0.10f}\tvalidation loss:{:0.10f}".format(e,(optimizer.param_groups[0]['lr']),real_trainingloss,real_validationloss))

            print("save model")
            if not os.path.isdir(model_save_dir):
                os.makedirs(model_save_dir)
            dirsavename = model_save_dir + "epoch_" + str(e) + "/"
            os.makedirs(dirsavename)
            torch.save(model.state_dict(), dirsavename + "weight.pt")
            saveVisualizeModelOutput(model, ball_datas_valid, dirsavename + "output1.png", seed=1)
            saveVisualizeModelOutput(model, ball_datas_valid, dirsavename + "output2.png", seed=100)
            saveVisualizeModelOutput(model, ball_datas_valid, dirsavename + "output3.png", seed=200)
            saveVisualizeModelOutput(model, ball_datas_valid, dirsavename + "output4.png", seed=300)
            saveVisualizeModelOutput(model, ball_datas_valid, dirsavename + "output5.png", seed=400)
            model.reset_hidden_cell(batch_size=batch_size)

            scheduler.step()
        except KeyboardInterrupt:
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


def exportModel(model_name:str, weight:str):
    model = MODEL_MAP[model_name](device='cpu')
    print("exporting: " + weight)
    model.load_state_dict(torch.load(weight))
    model.eval()
    model_script = torch.jit.script(model)
    model_script.save('model_final.pth')


def validModel(model_name, weight) :
    model = MODEL_MAP[model_name](device='cuda:0')
    print("validating: " + weight)
    model.cuda()
    model.load_state_dict(torch.load(weight))
    model.eval()
    criterion = nn.MSELoss().cuda()
    ball_datas = dfo.BallDataSet("ball_simulate_v2/medium.valid.bin",device='cuda:0')
    loader = DataLoader(ball_datas,1)
    loss_sum = 0
    for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(loader):
        model.reset_hidden_cell(batch_size=1)
        out = model(X1,x1_len,X2,x2_len)
        loss = criterion(out, labels.view(out.shape[0], -1))
        loss_sum += loss.item()
    print("loss: " + str(loss_sum / len(ball_datas)))

def configRoom(ax:Axes):
    lim = c.CAMERA_AREA_HALF_LENGTH
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(0,lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def drawLine3d(axe:plt.Axes,line:equ.LineEquation3d, color="r", label=None):
    points = [line.getPoint({'x':-c.BALL_AREA_HALF_LENGTH}),line.getPoint({'x':c.BALL_AREA_HALF_LENGTH})]
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

def saveVisualizeModelOutput(model:ISEFWINNER_BASE, dataset, imgFileName, seed = 3):
    model.eval()
    criterion = nn.MSELoss()
    model.reset_hidden_cell(batch_size=1)

    r, r_len, l, l_len, t, ans = dataset[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    t = t.view(1, -1)
    ans = ans.view(1, -1, 3)
    out = model(r, r_len, l, l_len, t).view(-1, 3)
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

    plt.savefig(imgFileName)
    plt.close()

def drawLineSeq(axe:plt.Axes, seq:torch.Tensor, seq_len:torch.Tensor, color="r") :
    lines_t = seq.view(-1, 5)
    obj = None
    for i in range(seq_len.view(1)[0]) :
        line = equ.LineEquation3d(None, None)
        line.setByPointOblique(equ.Point3d(lines_t[i][0], lines_t[i][1], lines_t[i][2]), lines_t[i][3], lines_t[i][4])
        obj = drawLine3d(axe, line, color=color)
    return obj


def visualizeModelOutput(model_name, weight, seed = 3):
    model = MODEL_MAP[model_name](device='cpu')

    model.load_state_dict(torch.load(weight))
    model.eval()

    criterion = nn.MSELoss()
    ball_datas = dfo.BallDataSet("ball_simulate_v2/dataset/medium_pred.valid.bin", device='cpu')

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
    
MODEL_MAP = {
    "small":ISEFWINNER_SMALL,
    "medium":ISEFWINNER_MEDIUM,
    "large":ISEFWINNER_LARGE
}


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-lr', default=0.001, type=float)
    argparser.add_argument('-b', default=64, type=int)
    argparser.add_argument('-e', default=30, type=int)
    argparser.add_argument('-m', default="medium", type=str)
    argparser.add_argument('-d', default="medium_fit", type=str)
    argparser.add_argument('-s', default=8, type=int)
    argparser.add_argument('-w', default=None, type=str)
    argparser.add_argument('-n', default="default", type=str)
    argparser.add_argument('--num_workers', default=0, type=int)
    argparser.add_argument('--export-model', dest='export', action='store_true', default=False)
    argparser.add_argument('--test', dest='test', action='store_true', default=False)
    argparser.add_argument('--mode', default="fit", type=str)

    args = argparser.parse_args()
    # if ball_simulate_v2/dataset not exested, then create
    if not os.path.exists("ball_simulate_v2/dataset"):
        os.mkdir("ball_simulate_v2/dataset")
    if args.export:
        exit(0)
    if args.test:
        exit(0)

    if args.mode != "default":
        if args.mode == "fit" :
            c.set2Fitting()
            dfo.loadLib()
        elif args.mode == "ne" :
            c.set2NoError()
            dfo.loadLib()
        elif args.mode == "predict" :
            c.set2Predict()
            dfo.loadLib()
        else :
            raise Exception("mode error")
    train(scheduler_step_size=args.s, LR=args.lr, batch_size=args.b, epochs=args.e, dataset=args.d, model_name=args.m, weight=args.w, name=args.n, num_workers=args.num_workers)
    pass