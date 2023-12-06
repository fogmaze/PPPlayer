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
import ball_predection.sync as sync

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
        self.lag = 0
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
        if os.path.exists(os.path.join("configs", configName, "lag")) :
            with open(os.path.join("configs", configName, "lag"), "r") as f :
                self.lag = int(f.read())

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
        with open(os.path.join("configs", configName, "lag"), "w") as f :
            f.write(str(self.lag))
        with open(os.path.join("configs", configName, "weight"), "w") as f :
            f.write(str(self.weight))
        with open(os.path.join("configs", configName, "model_name"), "w") as f :
            f.write(str(self.model_name))
        with open(os.path.join("configs", configName, "mode"), "w") as f :
            f.write(str(self.mode))

def createPredictionConfig(source:Tuple, save_name, detectionConfig:Tuple[Detection.DetectionConfig, Detection.DetectionConfig], weight, model_name="medium", mode = "normalB") :
    common.replaceDir("configs", save_name)
    config = PredictionConfig()
    config.detectionConfigs = detectionConfig
    config.lag = sync.getPredictionLagframes(source, detectionConfig)
    config.weight = weight
    config.model_name = model_name
    config.mode = mode
    config.save(save_name)
            
def predict(
        config:PredictionConfig,
        source                   = (0, 1),
        save_name                = "dual_default", 
        visualization            = True
        ) :


    SPEED_UP = False
    mode = config.mode
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

    
    print(save_name)
    common.replaceDir("results/", save_name)
    pf = open("results/" + save_name + "/pred.csv", "w")
    pfw = csv.writer(pf)
    pfw.writerow(["which camera", "frame", "hp_x", "hp_y", "hp_z", "hp_t", "x", "y", "z"])

    queue = mp.Queue()
    c12d, c12s = mp.Pipe()
    c22d, c22s = mp.Pipe()


    if type(source) == tuple and len(source) == 2 :
        if type(source[0]) == int or type(source[0]) == str:
            
            source1 = source[0]
            source2 = source[1]
        else :
            source1 = CameraReceiver(source[0])
            source2 = CameraReceiver(source[1])
    elif type(source) == str :
        source1 = os.path.join("results", source + "/cam1/all.mp4")
        source2 = os.path.join("results", source + "/cam2/all.mp4")

    p1 = mp.Process(target=runDec, args=(source1, "/cam1", config.detectionConfigs[0], queue, c12s, save_name))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", config.detectionConfigs[1], queue, c22s, save_name))

    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    lagger1 = Lagger(10)
    lagger2 = Lagger(0)

    process_time = 0
    process_time_iter = 0
    tra_time = 0
    tra_iter = 0

    p1.start()
    p2.start()

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
    f, ax = display.createFigRoom()
    c12d.send("start")
    c22d.send("start")

    while True :
        if not queue.empty() :
            recv_data = queue.get()
            tra_time += time.time() - recv_data[7]
            tra_iter += 1

            nowT = time.time()
            isHit = None
            if recv_data[0] == p1.pid:
                new_data = lagger1.update(recv_data)
                if new_data is None :
                    continue
                if new_data[2] is not None :
                    isHit = not lines1.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
                    if isHit :
                        lines2.clear()
                which = 1
            elif recv_data[0] == p2.pid:
                new_data = lagger2.update(recv_data)
                if new_data is None :
                    continue
                if new_data[2] is not None :
                    isHit = not lines2.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
                    if isHit :
                        lines1.clear()
                which = 2
            # send to model
            if isHit is None :
                pass
            elif isHit:
                pass
            elif SPEED_UP :
                pass
            elif len(lines1.lines) > 1 and len(lines2.lines) > 1 :
                model.reset_hidden_cell(1)
                l, l_len, r, r_len = prepareModelInput(lines1.lines, lines2.lines)
                out:torch.Tensor = model(l, l_len, r, r_len, NORMED_PREDICT_T) 
                Constants.normer.unnorm_ans_tensor(out)
                hp, t = getHitPointInformation(out)
                #print("hit point:", hp, "time:", t)
                process_time += time.time() - nowT
                process_time_iter += 1
                if hp is not None :
                    pfw.writerow([which, new_data[1], float(hp[0]), float(hp[1]), float(hp[2]), float(t)] + out.view(-1).tolist())
                else :
                    pfw.writerow([which, new_data[1], -1, -1, -1, -1] + out.view(-1).tolist())

        if c12d.poll() :
            recv = c12d.recv()
            if recv == "stop" :
                c22d.send("stop")
                break
        if c22d.poll() :
            recv = c22d.recv()
            if recv == "stop" :
                c12d.send("stop")
                break
    c12d.send("stop")
    c22d.send("stop")
    
    p1.join()
    p2.join()
    if not process_time_iter == 0 :
        process_time /= process_time_iter 
        print("mean process time:", process_time, "; it/s:", process_time_iter / process_time)
        print("mean tra time:", tra_time / tra_iter, "; it/s:", tra_iter / tra_time)
    pf.close()

    # display
    if visualization :
        display.visualizePrediction_video(os.path.join("results", save_name), fps=30)

if __name__ == "__main__" :
    parser = ap.ArgumentParser()
    parser.add_argument("-cc", "--create_config", help="use this flag to create config. otherwise use config to run prediction", action="store_true", default=False)
    parser.add_argument("-c", "--config", help="config name (nessesary)")
    parser.add_argument("-s", "--source", help="source (nessesary)", nargs=2)
    parser.add_argument("-dc", "--detection_config", help="detection config name. (only needed when creating config)", nargs=2)
    parser.add_argument("-m", "--model", help="model name. (only needed when creating configs)", default="medium")
    parser.add_argument("-w", "--weight", help="weight path (only needed when creating configs)", default="normalB/epoch_29/weight.pt")
    parser.add_argument("--mode", help="mode", default="normalB")
    parser.add_argument("-nv", "--non_visualization", help="Skip visualize the result when finished", action="store_true", default=False)

    args = parser.parse_args()
    source = (
        args.source[0] if not args.source[0].isnumeric() else int(args.source[0]),
        args.source[1] if not args.source[1].isnumeric() else int(args.source[1])
    )
    if args.create_config :
        dc = (Detection.DetectionConfig(), Detection.DetectionConfig())
        dc[0].load(args.detection_config[0])
        dc[1].load(args.detection_config[1])
        createPredictionConfig(source, args.config, dc, args.weight, args.model, args.mode)
    else :
        config = PredictionConfig()
        config.load(args.config)
        predict(config, source, args.config, not args.non_visualization)

    exit()

    #predict("medium", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt", frame_size=(640, 480),calibrationFiles_initial=("calibration", "calibration"), calibrationFiles=("calibration_hd", "calibration"), color_ranges="cr3", source=("exp/3.mp4", "exp/4.mp4"), visualization=True, initial_frames_gray=ini)
    predict("medium", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt", ("exp/a50.mp4", "exp/a15.mp4"), ("s480p30_a50_", "s480p30_a15_"), "dual_test_annoy", visualization=True, )
    #predict("medium", "ball_simulate_v2/model_saves/predict/epoch_29/weight.pt", source=(0, 1))
    exit()
    ini = cv2.imread("exp/t1696227891.9957368.jpg", cv2.IMREAD_GRAYSCALE)
    pos, ho = Detection.setup_camera_img(ini[1], "calibration")
    print(pos.to_str(), ho.shape)
    save("pos", pos)
    save("ho", ho)
    exit()