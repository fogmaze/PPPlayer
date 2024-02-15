from typing import Dict
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())
import core.Constants as Constants

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        # shape: [batch_size, time, 3]
        return torch.mean(torch.sqrt(torch.sum((output - target) ** 2, -1)))


class ISEFWINNER_BASE(nn.Module):
    #input:  [cam_x, cam_y, cam_z, rad_xy, rad_xz] * 2 , [time]
    #output: [x, y, z]
    device = "cuda:0"
    input_size:int = 5
    output_size:int = 3
    output_len:int  = None
    mlp1_out:int
    mlp2_out:int
    lstm_out:int
    lstm_num_layers:int
    mlp1:nn.Sequential
    lstm:nn.LSTM
    llstm_hidden_cell:tuple
    rlstm_hidden_cell:tuple
    mlp2:nn.Sequential
    llstm_last:torch.Tensor = None
    rlstm_last:torch.Tensor = None

    def fit_one_iteration(self, data, optimizer, loss_fn) :
        r, r_len, l, l_len, target = data
        self.reset_hidden_cell(r.shape[0])
        optimizer.zero_grad()
        output = self(l, l_len, r, r_len)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validate_one_iteration(self, data, loss_fn) :
        r, r_len, l, l_len, target = data
        self.reset_hidden_cell(r.shape[0])
        output = self(l, l_len, r, r_len)
        loss = loss_fn(output, target)
        return loss.item()

    @torch.jit.export
    def reset_hidden_cell(self, batch_size:int):
        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_out, device=self.device), torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_out, device=self.device), torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))
        self.llstm_last = None
        self.rlstm_last = None

    def forward(self, X1:torch.Tensor, X1_len:torch.Tensor, X2:torch.Tensor, X2_len:torch.Tensor, T:torch.Tensor):
        x1_batch_size = len(X1)
        x2_batch_size = len(X2)
        # 輸入全連接層1 
        X1 = self.mlp1(X1.view(-1,self.input_size)).view(x1_batch_size, -1, self.mlp1_out)
        X2 = self.mlp1(X2.view(-1,self.input_size)).view(x2_batch_size, -1, self.mlp1_out)

        # 輸入LSTM
        X1 = X1.transpose(0, 1)
        X2 = X2.transpose(0, 1)
        X1, self.llstm_hidden_cell = self.lstm(X1, self.llstm_hidden_cell)
        X2, self.rlstm_hidden_cell = self.lstm(X2, self.rlstm_hidden_cell)

        # 擷取LSTM最後一次的輸出 
        X1_len_ind = X1_len - 1
        X2_len_ind = X2_len - 1
        X1_ind = X1_len_ind.view(1, x1_batch_size, 1).expand(1, x1_batch_size, self.lstm_out)
        X2_ind = X2_len_ind.view(1, x2_batch_size, 1).expand(1, x2_batch_size, self.lstm_out)
        self.llstm_last = X1.gather(0, X1_ind).view(x1_batch_size, self.lstm_out)
        self.rlstm_last = X2.gather(0, X2_ind).view(x2_batch_size, self.lstm_out)
        
        # 合併前段模型輸出
        X1 = torch.cat((self.llstm_last, self.rlstm_last), 1)

        # 輸入全連接層2
        return self.mlp2(X1).view(-1, self.output_len, self.output_size)

    def forward_left_update(self, X:torch.Tensor) :
        batch_size = len(X)
        X = self.mlp1(X.view(-1,self.input_size)).view(batch_size, 1, self.mlp1_out)
        X = X.transpose(0, 1)
        X, self.llstm_hidden_cell = self.lstm(X, self.llstm_hidden_cell)
        self.llstm_last = X.transpose(0, 1)

    def forward_right_update(self, X:torch.Tensor) :
        batch_size = len(X)
        X = self.mlp1(X.view(-1,self.input_size)).view(batch_size, 1, self.mlp1_out)
        X = X.transpose(0, 1)
        X, self.rlstm_hidden_cell = self.lstm(X, self.rlstm_hidden_cell)
        self.rlstm_last = X.transpose(0, 1)

    def forward_predict(self, T:torch.Tensor) :
        if self.llstm_last is None or self.rlstm_last is None:
            return None
        batch_size = len(T)
        X = torch.cat((self.llstm_last, self.rlstm_last), 2)
        X = X.repeat(1, T.shape[1], 1)
        X = torch.cat((X, T.view(batch_size, T.shape[1], 1)), 2)
        res = self.mlp2(X.view(-1, self.lstm_out * 2 + 1)).view(batch_size, T.shape[1], self.output_size)
        return res

class ISEFWINNER_SMALL(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device
        assert Constants.MODEL3_OUTPUT_LEN
        self.output_len = Constants.MODEL3_OUTPUT_LEN

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
            nn.Linear(mlp2_l3_out, self.output_size * self.output_len)
        )
        
class ISEFWINNER_MEDIUM(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device
        assert Constants.MODEL3_OUTPUT_LEN
        self.output_len = Constants.MODEL3_OUTPUT_LEN

        mlp1_l1_out = 100
        mlp1_l2_out = 80
        mlp1_l3_out = 80
        mlp1_l4_out = 50

        mlp2_l1_out = 90
        mlp2_l2_out = 60
        mlp2_l3_out = 50
        mlp2_l4_out = 80

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
            nn.Linear(mlp2_l4_out, self.output_size * self.output_len)
        )

class ISEFWINNER_BIG(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device
        assert Constants.MODEL3_OUTPUT_LEN
        self.output_len = Constants.MODEL3_OUTPUT_LEN

        mlp1_l1_out = 140
        mlp1_l2_out = 120
        mlp1_l3_out = 100
        mlp1_l4_out = 90

        mlp2_l1_out = 90
        mlp2_l2_out = 90
        mlp2_l3_out = 90
        mlp2_l4_out = 60
        mlp2_l5_out = 80

        self.mlp1_out = mlp1_l4_out
        self.mlp2_out = self.output_size
        self.lstm_out = 70
        self.lstm_num_layers = 7

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
            nn.Linear(mlp2_l5_out, self.output_size * self.output_len)
        )

class ISEFWINNER_LARGE(ISEFWINNER_BASE):
    def __init__(self,device = "cuda:0"):
        self.device = device
        assert Constants.MODEL3_OUTPUT_LEN
        self.output_len = Constants.MODEL3_OUTPUT_LEN

        mlp1_l1_out = 200
        mlp1_l2_out = 150
        mlp1_l3_out = 150
        mlp1_l4_out = 90
        mlp1_l5_out = 70

        mlp2_l1_out = 130
        mlp2_l2_out = 120
        mlp2_l3_out = 100
        mlp2_l4_out = 60
        mlp2_l5_out = 100


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
            nn.Linear(mlp2_l5_out, self.output_size * self.output_len)
        )

MODEL_MAP:Dict[str, ISEFWINNER_BASE] = {
    "small":ISEFWINNER_SMALL,
    "medium":ISEFWINNER_MEDIUM,
    "big":ISEFWINNER_BIG,
    "large":ISEFWINNER_LARGE
}