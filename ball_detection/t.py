import time
import tqdm
import pickle
import os
import sys
import multiprocessing as mp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import csv
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import ball_detection.Detection as Det
import ball_detection.ColorRange as CR
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ball_detection.ColorRange import ColorRange, load, save


def calculateBallSize(xml_dir, max_len) :
    ws = []
    hs = []
    v = 0
    for ind in range(max_len) :
        try:
            tree = ET.parse(xml_dir + str(ind).zfill(4) + '.xml')
            root = tree.getroot()
            objs = root.findall('object')
            for obj in objs:
                obj_w = int(obj.find('bndbox').find('xmax').text) - int(obj.find('bndbox').find('xmin').text)
                obj_h = int(obj.find('bndbox').find('ymax').text) - int(obj.find('bndbox').find('ymin').text)
                ws.append(obj_w)
                hs.append(obj_h)
                v += 1
            ind += 1
            print(ind)
        except:
            print(ind, "is not exist")
            pass
    ws = np.array(ws)
    hs = np.array(hs)
    print(ws.mean() * hs.mean())
    
def cmpResult(xml_dir, res_data, img_dir, img_start, frame_len, root_dir="./") :
    # list all xml files in xml_dir
    marked_xmls = os.listdir(xml_dir)
    marked_i = [int(t.split('.')[0]) for t in marked_xmls]
    detecteds = []
    detection_result = {}
    for data in res_data :
        detection_result[int(data[0])] = data[1:]
    tp = 0
    for i in range(img_start, frame_len+img_start) :
        if i in marked_i and i - img_start in detection_result :
            tp += 1
    fp = len(detection_result) - tp
    tn = len(marked_i) - tp
    fn = frame_len - tp - fp - tn
    print("tp: " + str(tp) + " tn: " + str(tn) + " fp: " + str(fp) + " fn: " + str(fn) + " frame_len: " + str(frame_len))
    print("tpr: " + str(tp/ frame_len) + " tnr: " + str(fn / frame_len) + " fpr: " + str(fp / frame_len) + " fnr: " + str(fn / frame_len))
    for marked in marked_xmls:
        ind_marked = int(marked.split('.')[0]) - img_start
        if ind_marked in detection_result :
            detecteds.append(detection_result[ind_marked])
            # delete detection_result[ind_marked]
            del detection_result[ind_marked]
            #detecteds.pop(ind_marked)
        else :
            detecteds.append(None)
    #corr_rate = corr_len / len(marked_xmls) * 100
    distances = []
    #error_rate = len(detection_result) / len(marked_xmls) * 100
    
    for i, mark in enumerate(marked_xmls) :
        detected = detecteds[i]
        if detected is None :
            continue
        mid_detected = getMiddleOfRect(int(detected[1]), int(detected[2]), int(detected[3]), int(detected[4]))
        tree = ET.parse(os.path.join(xml_dir, mark))
        root = tree.getroot()        
        objs = root.findall('object')[0]
        xmin = int(objs.find('bndbox').find('xmin').text)
        ymin = int(objs.find('bndbox').find('ymin').text)
        xmax = int(objs.find('bndbox').find('xmax').text)
        ymax = int(objs.find('bndbox').find('ymax').text)
        mid_marked = (xmin + xmax) / 2, (ymin + ymax) / 2
        distances.append(np.sqrt((mid_detected[0] - mid_marked[0]) ** 2 + (mid_detected[1] - mid_marked[1]) ** 2))
        #show img
        if False and img_dir is not None:
            img = cv2.imread(os.path.join(img_dir, mark.split('.')[0] + '.jpg'))
            # draw marked
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            # draw detected
            cv2.rectangle(img, (int(detected[1]), int(detected[2])), (int(detected[3]) + int(detected[1]), int(detected[4]) + int(detected[2])), (0, 0, 255), 1)
            cv2.line(img, (int(mid_detected[0]), int(mid_detected[1])), (int(mid_marked[0]), int(mid_marked[1])), (255, 0, 0), 1)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("img", 1920,1080)
            cv2.imshow("img", img)
            cv2.waitKey(0)
    distances = np.array(distances)

    with open(os.path.join(root_dir, "result.csv"), "w") as f :
        writer = csv.writer(f)
        for d in distances :
            writer.writerow([d])
    #sns.boxplot(x=distances, color="gray")

    #plt.tick_params(axis="both", labelsize=32)

    #plt.xlabel("Distance between marked and detected (px)", fontsize=32)

    #plt.show()
    return tp, tn , fp , fn , np.percentile(distances, 25), np.percentile(distances, 50), np.percentile(distances, 75)


def getMiddleOfRect(x, y, w, h) :
    return x + w / 2, y + h / 2

def form(xml_dir ="/home/changer/Downloads/320_60_tagged/result/"):
    fns = os.listdir(xml_dir)
    for fn in fns :
        try:
            int(fn.split('.')[0])
        except:
            # delete the file
            os.remove(os.path.join(xml_dir, fn))
            print(fn, "is deleted")

def findRange_hsv_img(color_range:np.ndarray, source, xml_dir, frame_size, frame_rate = 30, beg = 0, consider_poly = None, root_dir="./"):
    detection = Det.Detection_img(source, color_range=color_range, frame_size=frame_size, frame_rate=frame_rate, mode="compute", beg_ind = beg, consider_poly=consider_poly)
    i = detection.runDetection(realTime=False)
    return cmpResult(xml_dir, detection.data, img_dir=source, img_start=beg, frame_len=i, root_dir=root_dir)

def findRange_hsv(color_range:np.ndarray, source, xml_dir, frame_size, frame_rate = 30):
    detection = Det.Detection(source, color_range=color_range, frame_size=frame_size, frame_rate=frame_rate, mode="compute")
    i = detection.runDetection(debugging=False)
    return cmpResult(xml_dir, detection.data, None, 0, i)

def caculateBestColorRange() :
    las_t = time.perf_counter()
    for hh in tqdm.tqdm(range(20, 40)) :
        for hs in range(235, 256) :
            for hv in range(235, 256) :
                for lh in range(2,22) :
                    for ls in range(77, 97) :
                        for lv in range(229, 249) :
                            if hh < lh and hs < ls and hv < lv :
                                continue
                            c = ColorRange([hh,hs,hv], [lh,ls,lv])
                            loss = findRange_hsv(c, "/home/changer/Downloads/320_60_tagged/all.mp4", "/home/changer/Downloads/320_60_tagged/result/", (640,480), 30)
                            print(time.perf_counter() - las_t)
                            las_t = time.perf_counter()

def findColorRange() :
    def recur_print_result() :
        while True:
            with open("color_range_2", "rb") as f :
                c = pickle.load(f)
            print("30fps:", findRange_hsv(c, "/home/changer/Downloads/320_60_tagged/all.mp4", "/home/changer/Downloads/320_60_tagged/result/", (640,480), 30))
            print("60fps:", findRange_hsv_img(c, "/home/changer/Downloads/hd_60_tagged/frames", "/home/changer/Downloads/hd_60_tagged/result/", (1920, 1080), 60, beg=1000))
    p = mp.Process(target=recur_print_result)
    with open("color_range_2", "rb") as f :
        cr = pickle.load(f)
    p.start()
    cr.runColorRange_video("ball_detection/result/hd_60_detection_r2/all.mp4", recursive=True, save_file_name="color_range_2")
    p.terminate()
    p.join()

def extractFrames(source, dest, max_len = 1000) :
    if not os.path.isdir(dest) :
        os.mkdir(dest)
    cap = cv2.VideoCapture(source)
    i = 0
    while True :
        ret, frame = cap.read()
        if not ret or i >= max_len:
            break
        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        if k == ord(" ") :
            cv2.imwrite(os.path.join(dest, str(i).zfill(4) + ".jpg"), frame)
        
        i += 1
    cap.release()

if __name__ == "__main__" :
    #os.mkdir("results/main6/cam2/position_")
    #for f in os.listdir("results/main6/cam2/position_in") :
        ## file name + 1
        #num = int(f.split('.')[0]) - 1
        ## copy file
        #os.system("cp " + os.path.join("results/main6/cam2/position_in", f) + " " + os.path.join("results/main6/cam2/position_", str(num).zfill(4) + ".xml"))
    #exit()

    
    dirname = "results/main6/cam2/position"
    con = Det.DetectionConfig()
    con.load("15_cam2")
    po = con.consider_poly
    with open("configs/cr3", "rb") as f :
        c = pickle.load(f)
    print(findRange_hsv_img(c, "results/main6/cam2/frames", dirname+"_in/", (640,480), 30, consider_poly=po, root_dir="results/main6/cam2/"))
    print("--------------------------------------------------")
    exit()
    
    os.mkdir(dirname+"_in") if not os.path.isdir(dirname+"_in") else None
    if not os.path.isdir(dirname) :
        print("no such directory")
        exit()
    for file in os.listdir(dirname) :
        tree = ET.parse(os.path.join(dirname, file))
        root = tree.getroot()        
        objs = root.findall('object')
        if len(objs) == 0 :
            continue
        objs = objs[0]
        xmin = int(objs.find('bndbox').find('xmin').text)
        ymin = int(objs.find('bndbox').find('ymin').text)
        xmax = int(objs.find('bndbox').find('xmax').text)
        ymax = int(objs.find('bndbox').find('ymax').text)
        mid_marked = (xmin + xmax) / 2, (ymin + ymax) / 2
        if mpltPath.Path(po).contains_point(mid_marked) :
            # copy file
            print("in")
            os.system("cp " + os.path.join(dirname, file) + " " + os.path.join(dirname+"_in", file))
    exit()  


    with open("/home/changer/Downloads/320_60_tagged/poly", "rb") as f:
        po = pickle.load(f)
 
     

    exit()
    #print(findRange_hsv(c, "/home/changer/Downloads/320_60_tagged/all.mp4", "/home/changer/Downloads/320_60_tagged/result/", (640,480), 30))
    print(findRange_hsv_img(c, "/home/changer/Downloads/hd_60_tagged/frames", "/home/changer/Downloads/hd_60_tagged/result/", (1920, 1080), 60, beg=1000))
    #print("--------------------------------------------------")
    #print(cmpResult("/home/changer/Downloads/hd_60_tagged/result/","ball_detection/result/hd_60_detection/","/home/changer/Downloads/320_60_tagged/frames/", 1000))