from argparse import ArgumentParser
import os
import logging
import parse
import shutil
import time
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
from core import *
import core
import pickle
import tqdm

class ISEFWINNER(nn.Module):
    #input:  [time_interval, line_xy.a, line_xy.b, line_xz.a,  line_xz.b] * 2
    #output: [speed_xy,      start.x,   start.y    end.x,      end.y,      highest]
    def __init__(self,device = "cuda:0"):
        self.device = device
        self.input_size = 5
        self.output_size = 6
        self.mlp1_l1_out = 50
        self.mlp1_l2_out = 40
        self.mlp1_l3_out = 40
        self.mlp1_l4_out = 15
        self.lstm_out = 40
        self.lstm_num_layers = 4
        self.mlp2_l1_out = 45
        self.mlp2_l2_out = 30
        self.mlp2_l3_out = 20
        batch_size = 1
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_size,self.mlp1_l1_out),
            nn.ReLU(),
            nn.Linear(self.mlp1_l1_out,self.mlp1_l2_out),
            nn.Tanh(),
            nn.Linear(self.mlp1_l2_out,self.mlp1_l3_out),
            nn.Tanh(),
            nn.Linear(self.mlp1_l3_out,self.mlp1_l4_out),
        )
        self.lstm = nn.LSTM(input_size = self.mlp1_l4_out,hidden_size= self.lstm_out,num_layers = self.lstm_num_layers)

        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=device))
        
        self.mlp2 = nn.Sequential(
            nn.Linear(self.lstm_out * 2,self.mlp2_l1_out),
            nn.ReLU(),
            nn.Linear(self.mlp2_l1_out,self.mlp2_l2_out),
            nn.Tanh(),
            nn.Linear(self.mlp2_l2_out,self.mlp2_l3_out),
            nn.Tanh(),
            nn.Linear(self.mlp2_l3_out, self.output_size)
        )

    @torch.jit.export
    def reset_hidden_cell(self, batch_size:int):
        self.llstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))
        self.rlstm_hidden_cell = (torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device),torch.zeros(self.lstm_num_layers,batch_size,self.lstm_out,device=self.device))

    #input shape:(batch_size, seq_len, input_size)
    def forward(self, X1:torch.Tensor, X1_len:torch.Tensor, X2:torch.Tensor ,X2_len:torch.Tensor, T:torch.Tensor):
        x1_batch_size = len(X1)
        x2_batch_size = len(X2)

        X1 = self.mlp1(X1.view(-1,self.input_size)).view(x1_batch_size, -1, self.mlp1_l4_out)
        X2 = self.mlp2(X2.view(-1,self.input_size)).view(x2_batch_size, -1, self.mlp1_l4_out)
        #shape: (batch_size, seq_len, input_size)

        X1 = X1.transpose(0,1)
        X2 = X2.transpose(0,1)
        #shape: (seq_len, batch_size, input_size)

        #X1 = nn.utils.rnn.pack_padded_sequence(X1,X1_len,enforce_sorted=False)
        #X2 = nn.utils.rnn.pack_padded_sequence(X2,X2_len,enforce_sorted=False)
        #shape: (seq_len, batch_size, input_size)
        
        X1_seq, self.llstm_hidden_cell = self.lstm(X1, self.llstm_hidden_cell)
        X2_seq, self.rlstm_hidden_cell = self.lstm(X2, self.rlstm_hidden_cell)
        #shape: (seq_len, batch_size, input_size)

        #X1_seq,r = nn.utils.rnn.pad_packed_sequence(X1)
        #X2_seq,l = nn.utils.rnn.pad_packed_sequence(X2)
        #shape: (seq_len, batch_size, input_size)

        #shape of T: (seq_len, batch_size, 1)
        res = torch.zeros(len(T),self.output_size,device=self.device)
        for i in range(T.shape[0]) :
            res[i] = self.mlp2(torch.cat((self.llstm_hidden_cell[0][self.lstm_num_layers-1], self.rlstm_hidden_cell[0][self.lstm_num_layers-1], T[i]),1))
        return res


def findLatestSave(name = None):
        all_save_dirs = os.listdir('model_saves/')
        all_save_dirs.sort(reverse=True)
        while len(os.listdir(os.path.join('model_saves/',all_save_dirs[0]))) < 2:
            del all_save_dirs[0]
        def fileneme_order(s):
            return int(parse.parse('epoch_{}.pt',s)[0])
        last_saves = os.listdir(os.path.join('model_saves/',all_save_dirs[0]))
        last_saves.sort(key=fileneme_order,reverse=True)
        return os.path.join('model_saves/',all_save_dirs[0],last_saves[0])


def train(epochs = 100, batch_size =16,scheduler_step_size=7, LR = 0.0001, device = "cuda:0",load_model = True,data_dir='data/'):
    torch.multiprocessing.set_start_method('spawn')
    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('training.log')
    format_log = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(format_log)
    file_handler.setFormatter(format_log)
    train_logger.addHandler(file_handler)
    train_logger.addHandler(console_handler)

    train_logger.info('start training')
    
    model_save_dir = time.strftime("model_saves/" + "/%Y-%m-%d_%H-%M-%S/",time.localtime())
    model = Watcher()

    if load_model:
        try:
            latest_saveName = findLatestSave()
            train_logger.info('loading: ' +latest_saveName) 
            model.load_state_dict(torch.load(latest_saveName))
        except:
            train_logger.error('cannot load model ')

    criterion = nn.MSELoss().cuda()
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,scheduler_step_size,0.1)

    ball_datas = BallSet_disk(os.path.join(data_dir,"train"))
    dataloader_train = DataLoader(dataset=ball_datas,batch_size=batch_size,shuffle=True,num_workers=2)
    ball_datas_valid = BallSet_disk(os.path.join(data_dir,"valid"))
    dataloader_valid = DataLoader(dataset=ball_datas_valid,batch_size=batch_size,num_workers=2)

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
            for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(dataloader_train):
                model.reset_hidden_cell(batch_size=X1.shape[0])
                optimizer.zero_grad()
                out = model(X1,x1_len,X2,x2_len)
                train_loss = criterion(out, labels.view(out.shape[0],-1))
                train_loss.backward()
                trainloss_sum += train_loss.item()
                optimizer.step()
                n += 1

            torch.cuda.empty_cache()
            
            model.eval()
            for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(dataloader_valid):
                model.reset_hidden_cell(batch_size=X1.shape[0])
                out = model(X1,x1_len,X2,x2_len)
                valid_loss = criterion(out, labels.view(out.shape[0],-1))
                validloss_sum += valid_loss.item()
            
            real_trainingloss = trainloss_sum / len(dataloader_train.dataset) * batch_size
            real_validationloss = validloss_sum / len(dataloader_valid.dataset) * batch_size

            print("==========================[epoch:" + str(e) + "]==============================")
            #print("learning rate: " + str(optimizer.param_groups[0]['lr']))
            #print("training loss:" + str(real_trainingloss) + "\tvalidation loss:" + str(real_validationloss))
            train_logger.info("epoch:{}\tlr:{:e}\ttraining loss:{:0.10f}\tvalidation loss:{:0.10f}".format(e,(optimizer.param_groups[0]['lr']),real_trainingloss,real_validationloss))

            if min_validloss > real_validationloss or e == 0:
                print("save model")
                if not os.path.isdir(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(),model_save_dir + "epoch_" + str(e) + ".pt")
                min_validloss = real_validationloss

            scheduler.step()
        except KeyboardInterrupt:
            c_exit = input("exit?[y/n]")
            if c_exit == "y":
                break

def exportLatestModel():
    model = Watcher(device='cpu')
    saveName = findLatestSave()
    print("exporting: " + saveName)
    model.load_state_dict(torch.load(saveName))
    model.eval()
    model_script = torch.jit.script(model)
    model_script.save('model_final.pth')


def setDataDir():
    train = BallSet_disk("data/train")
    test = BallSet_disk("data/test")
    valid = BallSet_disk("data/valid")
    train.createAndSaveTrainData(2000000)
    valid.createAndSaveTrainData(50000)

def validModel():
    model = Watcher(device='cuda:0')
    saveName = findLatestSave()
    print("validating: " + saveName)
    model.cuda()
    model.load_state_dict(torch.load(saveName))
    model.eval()
    criterion = nn.MSELoss().cuda()
    ball_datas = BallSet_disk("data/valid",device='cuda:0')
    loader = DataLoader(ball_datas,1)
    loss_sum = 0
    for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(loader):
        model.reset_hidden_cell(batch_size=1)
        out = model(X1,x1_len,X2,x2_len)
        loss = criterion(out, labels.view(out.shape[0],-1))
        loss_sum += loss.item()
    print("loss: " + str(loss_sum / len(ball_datas)))

def testModel(data_dir='data',batch_size=1,num_data = 50):
    model = Watcher(device='cpu')

    latest_saveName = findLatestSave()
    model.load_state_dict(torch.load(latest_saveName))
    model.eval()

    criterion = nn.MSELoss()
    ball_datas = BallSet_disk(os.path.join(data_dir,"valid"),device='cpu')
    loader = DataLoader(ball_datas,batch_size)
    i = 0
    for X1,x1_len,X2,x2_len,labels in tqdm.tqdm(loader):
        model.reset_hidden_cell(batch_size=batch_size)
        out = model(X1,x1_len,X2,x2_len)
        loss = criterion(out, labels.view(out.shape[0],-1))
        print(loss.item())
        if i >= num_data:
            break
        i += 1
    

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-lr',default=0.001,type=float)
    argparser.add_argument('-b',default=16,type=int)
    argparser.add_argument('-e',default=35,type=int)
    argparser.add_argument('-d',default='data',type=str)
    argparser.add_argument('-s',default=7,type=int)
    argparser.add_argument('--load-model',dest='load',action='store_true',default=False)
    argparser.add_argument('--set-data',dest='set_data',action='store_true',default=False)
    argparser.add_argument('--export-model',dest='export',action='store_true',default=False)
    argparser.add_argument('--test',dest='test',action='store_true',default=False)
    args = argparser.parse_args()
    if args.set_data:
        setDataDir()
        exit(0)
    if args.export:
        exportLatestModel()
        exit(0)
    if args.test:
        testModel()
        exit(0)
    train(load_model=args.load,scheduler_step_size=args.s,LR=args.lr,batch_size=args.b,epochs=args.e,data_dir=args.d)
    pass