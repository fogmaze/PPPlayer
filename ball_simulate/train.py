import random
import sys
import os
sys.path.append(os.getcwd())
import ball_simulate.dataFileOperator as dfo
from argparse import ArgumentParser
import logging
import time
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from core.Constants import *
import core
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import core.Equation3d as equ
from typing import List,Tuple
import tqdm



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
    def forward(self, X1:torch.Tensor, X2:torch.Tensor ,T:torch.Tensor):
            
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

        #shape of T: (seq_len, batch_size, 1)
        T = T.transpose(0, 1).view(-1, x1_batch_size, 1)
        res = None
        for i in range(T.shape[0]) :
            inp = torch.cat((self.llstm_hidden_cell[0][self.lstm_num_layers-1], self.rlstm_hidden_cell[0][self.lstm_num_layers-1], T[i]),1)
            if res == None:
                res = self.mlp2(inp).view(x1_batch_size, 1, self.output_size)
            else:
                res = torch.cat((res, self.mlp2(inp).view(x1_batch_size, 1, self.output_size)), 1)
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



def train(epochs = 100, batch_size =16,scheduler_step_size=7, LR = 0.0001, dataset = "",model_name = "small", weight = None, device = "cuda:0"):
    model_save_dir = time.strftime("./ball_simulate/model_saves/" + model_name + "%Y-%m-%d_%H-%M-%S/",time.localtime())

    torch.multiprocessing.set_start_method('spawn')
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    os.makedirs(model_save_dir)
    file_handler = logging.FileHandler(os.path.join(model_save_dir, 'training.log'))
    format_log = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(format_log)
    file_handler.setFormatter(format_log)
    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

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

    ball_datas_train = dfo.BallDataSet_sync(dataset + ".train.bin",device=device)
    dataloader_train = DataLoader(dataset=ball_datas_train,batch_size=batch_size,shuffle=False, num_workers=0)

    ball_datas_valid = dfo.BallDataSet(dataset + ".valid.bin",device=device)
    dataloader_valid = DataLoader(dataset=ball_datas_valid,batch_size=batch_size)

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
            for r, l, t, ans in tqdm.tqdm(dataloader_train):
                model.reset_hidden_cell(batch_size=r.shape[0])
                optimizer.zero_grad()
                out = model(r,l,t)
                train_loss = criterion(out, ans)
                train_loss.backward()
                trainloss_sum += train_loss.item()
                optimizer.step()
                n += 1

            torch.cuda.empty_cache()
            
            model.eval()
            for r, l, t, ans in tqdm.tqdm(dataloader_valid):
                model.reset_hidden_cell(batch_size=ans.shape[0])
                out = model(r,l,t)
                valid_loss = criterion(out, ans)
                validloss_sum += valid_loss.item()
            
            real_trainingloss = trainloss_sum / len(dataloader_train.dataset) * batch_size
            real_validationloss = validloss_sum / len(dataloader_valid.dataset) * batch_size
            train_loss_history.append(real_trainingloss)
            valid_loss_history.append(real_validationloss)

            print("==========================[epoch:" + str(e) + "]==============================")
            train_logger.info("epoch:{}\tlr:{:e}\ttraining loss:{:0.10f}\tvalidation loss:{:0.10f}".format(e,(optimizer.param_groups[0]['lr']),real_trainingloss,real_validationloss))

            if min_validloss > real_validationloss or e == 0:
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

                min_validloss = real_validationloss

            scheduler.step()
        except KeyboardInterrupt:
            c_exit = input("exit?[Y/n]")
            if c_exit == "Y" or c_exit == "y" or c_exit == chr(13) :
                break
    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    plt.legend(['train_loss', 'valid_loss'], loc='upper left')
    plt.savefig(model_save_dir + "loss.png")


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
    ball_datas = dfo.BallDataSet("ball_simulate/medium.valid.bin",device='cuda:0')
    loader = DataLoader(ball_datas,1)
    loss_sum = 0
    for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(loader):
        model.reset_hidden_cell(batch_size=1)
        out = model(X1,x1_len,X2,x2_len)
        loss = criterion(out, labels.view(out.shape[0],-1))
        loss_sum += loss.item()
    print("loss: " + str(loss_sum / len(ball_datas)))

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

def plotOutput(ax, out, color = 'r'):
    o = out.view(-1,3)
    for p in o:
        ax.scatter(p[0].item(),p[1].item(),p[2].item(),c=color)



def saveVisualizeModelOutput(model:ISEFWINNER_BASE, dataset, imgFileName, seed = 3):
    model.eval()
    criterion = nn.MSELoss()
    model.reset_hidden_cell(batch_size=1)

    r, l, t, ans = dataset[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    t = t.view(1, -1)
    ans = ans.view(1, -1, 3)
    out = model(r,l,t).view(-1, 3)
    ans = ans.view(-1, 3)

    print("loss: " + str(criterion(out, ans).item()))

    normer.unnorm_ans_tensor(ans)
    normer.unnorm_ans_tensor(out)

    ax = createRoom()
    plotOutput(ax, out)
    plotOutput(ax, ans, color='b')

    plt.savefig(imgFileName)
    plt.close()

def visualizeModelOutput(model_name, weight, seed = 3):
    model = MODEL_MAP[model_name](device='cpu')
    batch_size = 1

    model.load_state_dict(torch.load(weight))
    model.eval()

    criterion = nn.MSELoss()
    ball_datas = dfo.BallDataSet("ball_simulate/dataset/medium.valid.bin", device='cpu')

    r, l, t, ans = ball_datas[seed]
    r = r.view(1, -1, 5)
    l = l.view(1, -1, 5)
    t = t.view(1, -1)
    ans = ans.view(1, -1, 3)
    out = model(r,l,t).view(-1, 3)
    ans = ans.view(-1, 3)

    print("loss: " + str(criterion(out, ans).item()))

    ans = normer.unnorm_ans_tensor(ans)
    out = normer.unnorm_ans_tensor(out)

    ax = createRoom()
    plotOutput(ax, out)
    plotOutput(ax, ans, color='b')
    plt.show()
    
MODEL_MAP = {
    "small":ISEFWINNER_SMALL,
    "medium":ISEFWINNER_MEDIUM
}

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-lr', default=0.001, type=float)
    argparser.add_argument('-b', default=16, type=int)
    argparser.add_argument('-e', default=35, type=int)
    argparser.add_argument('-m', default="small", type=str)
    argparser.add_argument('-d', default="./ball_simulate/dataset/tiny", type=str)
    argparser.add_argument('-s', default=10, type=int)
    argparser.add_argument('-w', default=None, type=str)
    argparser.add_argument('--set-data', dest='set_data', action='store_true', default=False)
    argparser.add_argument('--export-model', dest='export', action='store_true', default=False)
    argparser.add_argument('--test', dest='test', action='store_true', default=False)
    args = argparser.parse_args()
    if args.export:
        exit(0)
    if args.test:
        exit(0)
    train(scheduler_step_size=args.s, LR=args.lr, batch_size=args.b, epochs=args.e, dataset=args.d, model_name=args.m, weight=args.w)
    pass