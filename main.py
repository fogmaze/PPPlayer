import argparse as ap
import matplotlib.pyplot as plt
import time
import torch
import csv
import multiprocessing as mp
import pickle
from typing import List, Tuple
import os
import sys
sys.path.append(os.getcwd())
import core.display as display
import core.Constants as Constants
import ball_detection.Detection as Detection
from ball_simulate_v2.models import MODEL_MAP
import ball_simulate_v2.models as models
import core.common as common
from ball_detection.ColorRange import *
from camera_reciever.CameraReceiver import CameraReceiver
import robot_controll.controller as rc

def getHitPointInformation(_traj_unnormed:torch.Tensor) :
    END = 2.74/2
    traj = _traj_unnormed.view(-1, 3)
    if not traj[0][0] < END < traj[-1][0] :
        return None, None
    l = 0
    r = traj.shape[0] - 1
    while l < r :
        mid = (l + r) // 2
        if traj[mid][0] < END :
            l = mid + 1
        else :
            r = mid
    l = l - 1
    w = (END - traj[l][0]) / (traj[l+1][0] - traj[l][0])
    hit_point = traj[l] * (1 - w) + traj[l+1] * w
    return hit_point, (l * (1 - w) + (l+1) * w) * Constants.CURVE_SHOWING_GAP

class LineCollector:
    def __init__(self) :
        self.bounceTimestamp = 0
        self.lines:List[Tuple[int, int, int, int, int]] = []
    def put(self, x, y, z, rxy, rxz) :
        if self.checkHit(x, y, z, rxy, rxz) :
            self.lines.clear()
            return False
        else :
            if len(self.lines) == 0 :
                self.bounceTimestamp = time.time()
            if len(self.lines) >= Constants.SIMULATE_INPUT_LEN:
                self.lines.pop(0)
            self.lines.append((x, y, z, rxy, rxz))
            return True
    def checkHit(self, x, y, z, rxy, rxz) :
        pass
    def clear(self) :
        self.lines.clear()

def rad_minus(a, b) :
    return (a - b + np.pi) % (2 * np.pi) - np.pi

class LineCollector_hor(LineCollector) :
    def __init__(self):
        super().__init__()
    movement = None
    last_rxy = None
    def clear(self):
        self.last_rxy = None
        self.movement = None
        super().clear()
    def checkHit(self, x, y, z, rxy, rxz):
        if self.movement == 1:
            if rad_minus(rxy, self.last_rxy) < 0:
                self.movement = None
                self.last_rxy = None
                return True
        elif self.movement == -1 :
            if rad_minus(rxy, self.last_rxy) > 0:
                self.movement = None
                self.last_rxy = None
                return True
        elif self.movement == None and self.last_rxy is not None:
            self.movement = 1 if rxy - self.last_rxy > 0 else -1
        self.last_rxy = rxy
        return False

class Lagger :
    def __init__(self, lag) :
        self.lag = lag
        self.datas = []
    
    def update(self, data) :
        if self.lag == 0 :
            return data
        ret = None
        if len(self.datas) >= self.lag :
            ret = self.datas.pop(0)
        self.datas.append(data)
        return ret

def prepareModelInput(ll:list, rl:list, device="cuda:0") :
    l = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    r = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    if len(ll) > 0 :
        l[:len(ll)] = torch.tensor(ll, device=device)
    if len(rl) > 0 :
        r[:len(rl)] = torch.tensor(rl, device=device) 
    l = l.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    r = r.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    l_len = torch.tensor([len(ll)]).view(1,1).to(device)
    r_len = torch.tensor([len(rl)]).view(1,1).to(device)
    Constants.normer.norm_input_tensor(l)
    Constants.normer.norm_input_tensor(r)
    return l, l_len, r, r_len

def runDec(s, sub_name, detection_load, q, c2s, save_name) :
    dec = Detection.Detection(
        source = s,
        save_name = save_name + sub_name, 
        mode = "dual_analysis",
        queue = q,
        conn=c2s,
        config=detection_load
    )
    print("start")
    
    dec.runDetection()

class PredictionConfig :
    def __init__(self) :
        self.detectionConfigs:Tuple[Detection.DetectionConfig, Detection.DetectionConfig] = (None, None)
        self.loadName = None
        self.weight = None
        self.model_name = None
        self.mode = None

    def load(self, configName) :
        self.loadName = configName
        if os.path.exists(os.path.join("configs", configName, "cam1")) :
            c1 = Detection.DetectionConfig()
            c1.load(os.path.join(configName, "cam1"))
        else :
            raise Exception("config not found")

        if os.path.exists(os.path.join("configs", configName, "cam2")) :
            c2 = Detection.DetectionConfig()
            c2.load(os.path.join(configName, "cam2"))
        else :
            raise Exception("config not found")

        self.detectionConfigs = (c1, c2)

        with open(os.path.join("configs", configName, "weight"), "r") as f :
            self.weight = f.read()

        with open(os.path.join("configs", configName, "model_name"), "r") as f :
            self.model_name = f.read()
            
        with open(os.path.join("configs", configName, "mode"), "r") as f :
            self.mode = f.read()

    def save(self, configName) :
        if not os.path.exists(os.path.join("configs", configName)) :
            os.mkdir(os.path.join("configs", configName))
        if not os.path.exists(os.path.join("configs", configName, "cam1")) :
            os.mkdir(os.path.join("configs", configName, "cam1"))
        if not os.path.exists(os.path.join("configs", configName, "cam2")) :
            os.mkdir(os.path.join("configs", configName, "cam2"))
        self.detectionConfigs[0].save(os.path.join(configName, "cam1"))
        self.detectionConfigs[1].save(os.path.join(configName, "cam2"))
        with open(os.path.join("configs", configName, "weight"), "w") as f :
            f.write(str(self.weight))
        with open(os.path.join("configs", configName, "model_name"), "w") as f :
            f.write(str(self.model_name))
        with open(os.path.join("configs", configName, "mode"), "w") as f :
            f.write(str(self.mode))

def createPredictionConfig(save_name, detectionConfig:Tuple[Detection.DetectionConfig, Detection.DetectionConfig], weight, model_name="medium", mode = "normalB") :
    #common.replaceDir("configs", save_name)
    config = PredictionConfig()
    config.detectionConfigs = detectionConfig
    config.weight = weight
    config.model_name = model_name
    config.mode = mode
    #config.save(save_name)
    return config
            
def main(
        config:PredictionConfig,
        source                   = (0, 1),
        save_name                = "dual_default", 
        robot:rc.Robot           = None,
        visualization            = True,
        ) :

    # main process status : 
    #  predicting
    #  syncing
    status = "predicting"
    SPEED_UP = False
    mode = config.mode
    # set up model mode (for normalization model input and output)
    if mode != "default":
        if mode == "fit" :
            Constants.set2Fitting()
        elif mode == "ne" :
            Constants.set2NoError()
        elif mode == "prediConstantst" :
            Constants.set2Predict()
        elif mode == "normal" :
            Constants.set2Normal()
        elif mode == "normalB" :
            Constants.set2NormalB()
        elif mode == "normalB60" :
            Constants.set2NormalB60()
        elif mode == "normalBR" :
            Constants.set2NormalBR()
        else :
            raise Exception("mode error")

    # load model
    model:models.ISEFWINNER_BASE = MODEL_MAP[config.model_name](device="cuda:0")
    model.cuda()
    model.load_state_dict(torch.load(os.path.join("ball_simulate_v2/model_saves/",config.weight)))
    model.eval()
    Constants.set2Normal()
    NORMED_PREDICT_T = torch.arange(0, Constants.SIMULATE_TEST_LEN * Constants.CURVE_SHOWING_GAP, Constants.CURVE_SHOWING_GAP).to("cuda:0").view(1, -1)
    Constants.normer.norm_t_tensor(NORMED_PREDICT_T)

    
    # create save dir
    common.replaceDir("results/", save_name)
    # create pred.csv which stores the prediction result
    pf = open("results/" + save_name + "/pred.csv", "w")
    pfw = csv.writer(pf)
    pfw.writerow(["which camera", "frame", "hp_x", "hp_y", "hp_z", "hp_t", "x", "y", "z"])
    # create raw.csv which stores the raw data received from detection
    pfr = open("results/" + save_name + "/raw.csv", "w")
    pfrw = csv.writer(pfr)
    pfrw.writerow(["which camera", "frame", "x", "y", "z", "rxy", "rxz", "time"])

    # start realtime visualization
    displayQueue = None
    pd = None
    if visualization :
        displayQueue, pd = display.visualizePrediction_realtime(os.path.join("results", save_name))
    
    queue = mp.Queue() # Queue for receiving detection data from detection
    c12d, c12s = mp.Pipe() # Pipe for communication between main process and detection process
    c22d, c22s = mp.Pipe() # Pipe for communication between main process and detection process

    if type(source) == tuple and len(source) == 2 : # source is a video file or a camera
        if type(source[0]) == int or type(source[0]) == str:
            source1 = source[0]
            source2 = source[1]
        else : # source is android cam (no longer supported)
            source1 = CameraReceiver(source[0])
            source2 = CameraReceiver(source[1])
    elif type(source) == str : # source is a prediction folder
        source1 = os.path.join("results", source + "/cam1/all.mp4")
        source2 = os.path.join("results", source + "/cam2/all.mp4")

    # create detection process
    p1 = mp.Process(target=runDec, args=(source1, "/cam1", config.detectionConfigs[0], queue, c12s, save_name))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", config.detectionConfigs[1], queue, c22s, save_name))

    # create line collector that store line data and check if there is a hit
    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    # create lagger that lag the data

    process_time = 0
    process_time_iter = 0
    tra_time = 0
    tra_iter = 0
    nowCollectorTime = time.time()

    p1.start()
    p2.start()

    # sync with detection process
    while True :
        if c12d.poll() :
            msg = c12d.recv()
            if msg == "ready":
                break
    while True :
        if c22d.poll() :
            msg = c22d.recv()
            if msg == "ready":
                break
    # start detection
    c12d.send("start")
    c22d.send("start")

    # main loop
    while True :
        # receive line data from detection
        if not queue.empty() :
            recv_data = queue.get()
            # write raw data to raw.csv
            pfrw.writerow([1 if recv_data[0] == p1.pid else 2, recv_data[1], recv_data[2], recv_data[3], recv_data[4], recv_data[5], recv_data[6], recv_data[7]])

            tra_time += time.time() - recv_data[7]
            tra_iter += 1

            nowT = time.time()
            isHit = None
            out = None
            if recv_data[0] == p1.pid : # data is from camera 1
                # sync of frame_lag
                if recv_data[2] is not None : # if detected a ball
                    isHit = not lines1.put(recv_data[2], recv_data[3], recv_data[4], recv_data[5], recv_data[6]) # update new data to lineCollector
                    # if one of the camera detected the hit. clear the other lineCollector
                    if isHit :
                        lines2.clear()
                which = 1
            elif recv_data[0] == p2.pid : # data is from camera 2
                if recv_data[2] is not None :
                    isHit = not lines2.put(recv_data[2], recv_data[3], recv_data[4], recv_data[5], recv_data[6])
                    if isHit :
                        lines1.clear()
                which = 2
            if isHit is None or not status == "predicting":
                pass
            elif isHit:
                pass
            elif SPEED_UP :
                pass
            elif len(lines1.lines) == 1 :
                nowCollectorTime = time.time()
            elif len(lines1.lines) > 1 and len(lines2.lines) > 1 : # send to model
                model.reset_hidden_cell(1)
                l, l_len, r, r_len = prepareModelInput(lines1.lines, lines2.lines)
                out:torch.Tensor = model(l, l_len, r, r_len, NORMED_PREDICT_T) 
                Constants.normer.unnorm_ans_tensor(out)
                hp, t = getHitPointInformation(out)
                process_time += time.time() - nowT
                process_time_iter += 1
                if hp is not None :
                    pfw.writerow([which, recv_data[1], float(hp[0]), float(hp[1]), float(hp[2]), float(t)] + out.view(-1).tolist())
                    robot.move(hp[1].item(), hp[2].item())
                    if abs(time.time() - nowCollectorTime - t) < 0.15 :
                        robot.hit()
                        print("hit")
                else :
                    pfw.writerow([which, recv_data[1], -1, -1, -1, -1] + out.view(-1).tolist())
            # send data to display process
            if visualization :
                displayQueue.put(("frameData", which, (recv_data[2], recv_data[3], recv_data[4], recv_data[5], recv_data[6]), out.tolist() if out is not None else None))

        # communication between main process and detection process
        if c12d.poll() :
            recv = c12d.recv()
            if recv == "stop" : # one of the detection process is finished
                c22d.send("stop") # send stop signal to the other detection process
                break
            elif hasattr(recv, "__getitem__") : 
                if recv[0] == "keyPress" :
                    if recv[1] == ord("s") :
                        status = "syncing" if status == "predicting" else "predicting"
        if c22d.poll() :
            recv = c22d.recv()
            if recv == "stop" :
                c12d.send("stop")
                break
            elif hasattr(recv, "__getitem__") :
                if recv[0] == "keyPress" :
                    if recv[1] == ord("s") :
                        status = "syncing" if status == "predicting" else "predicting"
    displayQueue.put(("stop",))   
    c12d.send("stop")
    c22d.send("stop")
    
    p1.join()
    p2.join()
    #pd.join()
    if not process_time_iter == 0 :
        process_time /= process_time_iter 
        print("mean process time:", process_time, "; it/s:", process_time_iter / process_time)
        print("mean tra time:", tra_time / tra_iter, "; it/s:", tra_iter / tra_time)
    pf.close()
    pfr.close()

    # display
    if visualization:
        display.visualizePrediction_video(os.path.join("results", save_name), fps=30)



if __name__ == "__main__" :
    parser = ap.ArgumentParser()
    #parser.add_argument("-cc", "--create_config", help="use this flag to create config. otherwise use config to run prediction", action="store_true")
    #parser.add_argument("-c", "--config", help="config name (nessesary)")
    parser.add_argument("-ip", "--ip", help="ip address of robot", default="")
    parser.add_argument("-s", "--source", help="source (nessesary)", nargs=2)
    parser.add_argument("-dc", "--detection_config", help="detection config name. (nessesary)", nargs=2)
    parser.add_argument("-m", "--model", help="model name.", default="medium")
    parser.add_argument("-w", "--weight", help="weight path", default="normalB/epoch_29/weight.pt")
    parser.add_argument("-n", "--name", help="save name", default=None)
    parser.add_argument("-p", "--port", help="port", default=5678)
    parser.add_argument("--mode", help="mode", default="normalB")
    parser.add_argument("-nv", "--non_visualization", help="Skip visualize the result when finished", action="store_true", default=False)

    args = parser.parse_args()
    source = (
        args.source[0] if not args.source[0].isnumeric() else int(args.source[0]),
        args.source[1] if not args.source[1].isnumeric() else int(args.source[1])
    )
    # no longer used
    #if args.create_config and False:
        #dc = (Detection.DetectionConfig(), Detection.DetectionConfig())
        #dc[0].load(args.detection_config[0])
        #dc[1].load(args.detection_config[1])
        #createPredictionConfig(args.config, dc, args.weight, args.model, args.mode)
    # start predicting
    dc = (Detection.DetectionConfig(), Detection.DetectionConfig())
    dc[0].load(args.detection_config[0])
    dc[1].load(args.detection_config[1])
    robot = rc.Robot(args.ip, int(args.port))
    main(createPredictionConfig("tmp", dc, args.weight, args.model, args.mode), source, args.config if args.name==None else args.name, robot, not args.non_visualization)

    exit()
    main("medium", "normalB", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt")
