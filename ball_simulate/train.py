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
from core import *
import core
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import core.Equation3d as equ
from typing import List,Tuple
import tqdm

BALL_AREA_HALF_LENGTH = 3
BALL_AREA_HALF_WIDTH = 2
BALL_AREA_HEIGHT = 1

CAMERA_AREA_HALF_LENGTH = 7/2
CAMERA_AREA_HALF_WIDTH = 5/2
CAMERA_AREA_HEIGHT = 1.5


class ISEFWINNER_BASE(nn.Module):
    #input:  [cam_x, cam_y, cam_z, rad_xy, rad_xz] * 2 , [time]
    #output: [speed_xy,      start.x,   start.y    end.x,      end.y,      highest]
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



def train(epochs = 100, batch_size =16,scheduler_step_size=7, LR = 0.0001, dataset = "",model_name = "small", weight = None, device = "cuda:0"):
    torch.multiprocessing.set_start_method('spawn')
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./ball_simulate/training.log')
    format_log = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(format_log)
    file_handler.setFormatter(format_log)
    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

    train_logger.info('start training with args: epochs:{}, batch_size:{}, scheduler_step_size:{}, LR:{}, dataset:{}, model_name:{}, weight:{}, device:{}'.format(epochs,batch_size,scheduler_step_size,LR,dataset,model_name,weight,device))
    
    model_save_dir = time.strftime("./ball_simulate/model_saves/" + model_name + "%Y-%m-%d_%H-%M-%S/",time.localtime())
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

    ball_datas_train = dfo.BallDataSet(dataset + ".train.bin",device=device)
    dataloader_train = DataLoader(dataset=ball_datas_train,batch_size=batch_size,shuffle=True, num_workers=0)

    ball_datas_valid = dfo.BallDataSet(dataset + ".valid.bin",device=device)
    dataloader_valid = DataLoader(dataset=ball_datas_valid,batch_size=batch_size)

    train_loss = 0
    valid_loss = 0
    
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

            print("==========================[epoch:" + str(e) + "]==============================")
            train_logger.info("epoch:{}\tlr:{:e}\ttraining loss:{:0.10f}\tvalidation loss:{:0.10f}".format(e,(optimizer.param_groups[0]['lr']),real_trainingloss,real_validationloss))

            if min_validloss > real_validationloss or e == 0:
                print("save model")
                if not os.path.isdir(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(),model_save_dir + "epoch_" + str(e) + ".pt")
                min_validloss = real_validationloss

            scheduler.step()
        except KeyboardInterrupt:
            c_exit = input("exit?[Y/n]")
            if c_exit == "Y" or c_exit == "y" or c_exit == chr(13) :
                break


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

def plotOutput(ax, out):

    for input_data in inp:
        for work in input_data:
            drawLine3d(ax,work.lineCamBall)
    # plot ball points
    for pos in ans:
        ax.scatter(pos.ball_pos.x,pos.ball_pos.y,pos.ball_pos.z)



def visualizeModelOutput(model_name, weight):
    model = MODEL_MAP[model_name](device='cpu')
    batch_size = 1

    model.load_state_dict(torch.load(weight))
    model.eval()

    criterion = nn.MSELoss()
    ball_datas = dfo.BallDataSet("ball_simulate/medium.valid.bin")

    r, l, t, ans = ball_datas[4]
    out = model(r,l,t)

    print("loss: " + str(criterion(out, ans).item()))
    


MODEL_MAP = {
    "small":ISEFWINNER_SMALL
}

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-lr', default=0.001, type=float)
    argparser.add_argument('-b', default=16, type=int)
    argparser.add_argument('-e', default=35, type=int)
    argparser.add_argument('-m', default="small", type=str)
    argparser.add_argument('-d', default="./ball_simulate/dataset/medium", type=str)
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
    if args.set_data:
        import ball_simulate.simulate as sim
        sim.simulate(GUI=False, dataLength=100000, outputFileName="ball_simulate/dataset/medium.train.bin")
        sim.simulate(GUI=False, dataLength=10000, outputFileName="ball_simulate/dataset/medium.valid.bin")
        exit(0)
    train(scheduler_step_size=args.s, LR=args.lr, batch_size=args.b, epochs=args.e, dataset=args.d, model_name=args.m, weight=args.w)
    pass