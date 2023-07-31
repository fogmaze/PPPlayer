import pickle
import os
import csv
import cv2
import numpy as np
import xml.etree.ElementTree as ET


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
    
def cmpResult(xml_dir, res_dir, img_dir, img_start) :
    # list all xml files in xml_dir
    marked_xmls = os.listdir(xml_dir)
    detecteds = []
    detection_result = {}
    with open(os.path.join(res_dir, "detection.csv"), 'r') as f :
        reader = csv.reader(f)
        i = 0
        for row in reader :
            if i == 0 :
                i += 1
                continue
            detection_result[int(row[0])] = row[1:]
    corr_len = 0
    for marked in marked_xmls:
        ind_marked = int(marked.split('.')[0]) - img_start
        if ind_marked in detection_result :
            detecteds.append(detection_result[ind_marked])
            # delete detection_result[ind_marked]
            del detection_result[ind_marked]
            #detecteds.pop(ind_marked)
            corr_len += 1
        else :
            detecteds.append(None)
    print("corr_len", corr_len)
    print("corr_rate", corr_len / len(marked_xmls))
    distances = []
    print('error_len', len(detection_result))
    print('error_rate', len(detection_result) / len(marked_xmls))
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
        if True:
            img = cv2.imread(os.path.join(img_dir, mark.split('.')[0] + '.jpg'))
            # draw marked
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            # draw detected
            cv2.rectangle(img, (int(detected[1]), int(detected[2])), (int(detected[3]) + int(detected[1]), int(detected[4]) + int(detected[2])), (0, 0, 255), 1)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("img", 1920,1080)
            cv2.imshow("img", img)
            cv2.waitKey(0)
    distances = np.array(distances)
    print("mean distance", distances.mean())
    print("std distance", distances.std())


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

def findRange():
    cr = pickle.load("color_range")
    print(cr)

if __name__ == "__main__" :
    findRange()
    #cmpResult("/home/changer/Downloads/320_60_tagged/result/","ball_detection/result/320_60_detection/","/home/changer/Downloads/320_60_tagged/frames/", 0)
    #print("--------------------------------------------------")
    #cmpResult("/home/changer/Downloads/hd_60_tagged/result/","ball_detection/result/hd_60_detection/","/home/changer/Downloads/320_60_tagged/frames/", 1000)