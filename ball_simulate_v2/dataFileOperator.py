import csv
import torch
from ctypes import *
import torch
import platform
import sys
import os
sys.path.append(os.getcwd())
import threading

def loadLib(mode = "normalBR"):
    libname = "dataFileOperatorV2-{}-{}".format(SIMULATE_INPUT_LEN[mode], SIMULATE_TEST_LEN[mode])
    lib = CDLL("bin/lib{}.so".format(libname))

    class Data_Point_(Structure):
        _fields_ = [
            ("x",c_double),
            ("y",c_double),
            ("z",c_double)
        ]
    Data_Point = Data_Point_

    class Data_Input_(Structure):
        _fields_ = [
            ("camera_x", c_double),
            ("camera_y", c_double),
            ("camera_z", c_double),
            ("line_rad_xy", c_double * SIMULATE_INPUT_LEN[mode]),
            ("line_rad_xz", c_double * SIMULATE_INPUT_LEN[mode]),
            ("timestamps", c_double * SIMULATE_INPUT_LEN[mode]),
            ("seq_len", c_int)
        ]
    Data_Input = Data_Input_

    class DataStruct_(Structure):
        _fields_ = [
            ("inputs", Data_Input * 2),
            ("curvePoints", Data_Point * SIMULATE_TEST_LEN[mode]),
            ("curveTimestamps", c_double * SIMULATE_TEST_LEN[mode])
        ]
    DataStruct = DataStruct_

    lib.main.argtypes = []
    lib.main.restype = c_int
    lib.loadFromFile.argtypes = [c_char_p]
    lib.loadFromFile.restype = c_void_p
    lib.releaseData.argtypes = [c_void_p]
    lib.releaseData.restype = None
    lib.loadIsSuccess.argtypes = [c_void_p]
    lib.loadIsSuccess.restype = c_bool
    lib.getFileDataLength.argtypes = [c_void_p]
    lib.getFileDataLength.restype = c_int
    lib.getFileData.argtypes = [c_void_p, c_int]
    lib.getFileData.restype = c_void_p
    lib.createHeader.argtypes = [c_int]
    lib.createHeader.restype = c_void_p
    lib.putData.argtypes = [c_void_p, c_int, DataStruct]
    lib.putData.restype = c_bool
    lib.saveToFile.argtypes = [c_void_p, c_char_p]
    lib.saveToFile.restype = c_bool
    lib.getFileData_sync.argtypes = [c_char_p, c_int]
    lib.getFileData_sync.restype = DataStruct
    lib.getFileDataLength_sync.argtypes = [c_char_p]
    lib.getFileDataLength_sync.restype = c_int
    lib.createEmptyFile_sync.argtypes = [c_char_p, c_int]
    lib.createEmptyFile_sync.restype = None
    lib.putData_sync.argtypes = [c_char_p, c_int, DataStruct]
    lib.putData_sync.restype = None
    lib.merge.argtypes = [c_char_p, c_char_p, c_char_p]
    lib.merge.restype = None
    return lib

SIMULATE_INPUT_LEN = {
    "fit": 100,
    "ne": 40,
    "predict": 40,
    "normal": 40,
    "normalB": 40,
    "normalB60": 80,
    "normalBR": 40,
    # Add values for other modes if needed
}

SIMULATE_TEST_LEN = {
    "fit": 100,
    "ne": 250,
    "predict": 250,
    "normal": 50,
    "normalB": 50,
    "normalB60": 50,
    "normalBR": 50,
    # Add values for other modes if needed
}

class BallDataSet_put(torch.utils.data.Dataset) :
    def __init__(self, fileName, dataLength, device = "cuda:0", mode = "normalBR"):
        self.mode = mode
        self.loadLib()

        # Add conditions for other modes if needed

        self.fileName = fileName
        self.device = torch.device(device)

        self.lib.createEmptyFile_sync(self.fileName.encode('utf-8'), dataLength)
        
        if not os.path.exists(fileName) :
            raise Exception("file not found")
        self.length = self.lib.getFileDataLength_sync(self.fileName.encode('utf-8'))
        pass
    
    def loadLib(self) :
        mode = self.mode
        libname = "dataFileOperatorV2-{}-{}".format(SIMULATE_INPUT_LEN[mode], SIMULATE_TEST_LEN[mode])
        lib = CDLL("bin/lib{}.so".format(libname))

        class Data_Point_(Structure):
            _fields_ = [
                ("x",c_double),
                ("y",c_double),
                ("z",c_double)
            ]
        Data_Point = Data_Point_

        class Data_Input_(Structure):
            _fields_ = [
                ("camera_x", c_double),
                ("camera_y", c_double),
                ("camera_z", c_double),
                ("line_rad_xy", c_double * SIMULATE_INPUT_LEN[mode]),
                ("line_rad_xz", c_double * SIMULATE_INPUT_LEN[mode]),
                ("timestamps", c_double * SIMULATE_INPUT_LEN[mode]),
                ("seq_len", c_int)
            ]
        Data_Input = Data_Input_

        class DataStruct_(Structure):
            _fields_ = [
                ("inputs", Data_Input * 2),
                ("curvePoints", Data_Point * SIMULATE_TEST_LEN[mode]),
                ("curveTimestamps", c_double * SIMULATE_TEST_LEN[mode])
            ]
        DataStruct = DataStruct_

        lib.main.argtypes = []
        lib.main.restype = c_int
        lib.loadFromFile.argtypes = [c_char_p]
        lib.loadFromFile.restype = c_void_p
        lib.releaseData.argtypes = [c_void_p]
        lib.releaseData.restype = None
        lib.loadIsSuccess.argtypes = [c_void_p]
        lib.loadIsSuccess.restype = c_bool
        lib.getFileDataLength.argtypes = [c_void_p]
        lib.getFileDataLength.restype = c_int
        lib.getFileData.argtypes = [c_void_p, c_int]
        lib.getFileData.restype = c_void_p
        lib.createHeader.argtypes = [c_int]
        lib.createHeader.restype = c_void_p
        lib.putData.argtypes = [c_void_p, c_int, DataStruct]
        lib.putData.restype = c_bool
        lib.saveToFile.argtypes = [c_void_p, c_char_p]
        lib.saveToFile.restype = c_bool
        lib.getFileData_sync.argtypes = [c_char_p, c_int]
        lib.getFileData_sync.restype = DataStruct
        lib.getFileDataLength_sync.argtypes = [c_char_p]
        lib.getFileDataLength_sync.restype = c_int
        lib.createEmptyFile_sync.argtypes = [c_char_p, c_int]
        lib.createEmptyFile_sync.restype = None
        lib.putData_sync.argtypes = [c_char_p, c_int, DataStruct]
        lib.putData_sync.restype = None
        lib.merge.argtypes = [c_char_p, c_char_p, c_char_p]
        lib.merge.restype = None
        self.lib = lib
        self.dataStructClass = DataStruct
    
    def __len__(self):
        return self.length
    
    def putData(self, index:int, data):
        return self.lib.putData_sync(self.fileName.encode('utf-8'), index, data)

    def __getitem__(self, index):
        return None
       
    def saveToFile(self):
        pass


class BallDataSet_sync(torch.utils.data.Dataset) :
    def __init__(self, fileName, dataLength = None, device = "cuda:0", mode = "normalBR"):
        self.mode = mode

        lib = loadLib(self.mode)
        if self.mode == "fit":
            self.SIMULATE_INPUT_LEN = 100
            self.SIMULATE_TEST_LEN = 100
        elif self.mode == "ne":
            self.SIMULATE_INPUT_LEN = 40
            self.SIMULATE_TEST_LEN = 250
        elif self.mode == "predict":
            self.SIMULATE_INPUT_LEN = 40
            self.SIMULATE_TEST_LEN = 250
        elif self.mode == "normal":
            self.SIMULATE_INPUT_LEN = 40
            self.SIMULATE_TEST_LEN = 50
        elif self.mode == "normalB":
            self.SIMULATE_INPUT_LEN = 40
            self.SIMULATE_TEST_LEN = 50
        elif self.mode == "normalB60":
            self.SIMULATE_INPUT_LEN = 80
            self.SIMULATE_TEST_LEN = 50
        elif self.mode == "normalBR":
            self.SIMULATE_INPUT_LEN = 40
            self.SIMULATE_TEST_LEN = 50
        # Add conditions for other modes if needed

        self.fileName = fileName
        self.device = torch.device(device)

        if dataLength != None:
            lib.createEmptyFile_sync(self.fileName.encode('utf-8'), dataLength)
        
        if not os.path.exists(fileName) :
            raise Exception("file not found")
        self.length = lib.getFileDataLength_sync(self.fileName.encode('utf-8'))
        pass
    
    def __len__(self):
        return self.length
    
    def putData(self, index:int, data):
        lib = loadLib(self.mode)
        return lib.putData_sync(self.fileName.encode('utf-8'), index, data)

    def __getitem__(self, index):
        lib = loadLib(self.mode)
        d_ori = lib.getFileData_sync(self.fileName.encode('utf-8'), index)
        d_list_r = [None] * self.SIMULATE_INPUT_LEN
        d_list_l = [None] * self.SIMULATE_INPUT_LEN
        d_list_t = [None] * self.SIMULATE_TEST_LEN
        d_list_ans = [None] * self.SIMULATE_TEST_LEN
        for i in range(self.SIMULATE_INPUT_LEN):
            d_list_r[i] = [d_ori.inputs[0].camera_x, d_ori.inputs[0].camera_y, d_ori.inputs[0].camera_z ,d_ori.inputs[0].line_rad_xy[i], d_ori.inputs[0].line_rad_xz[i]]
            d_list_l[i] = [d_ori.inputs[1].camera_x, d_ori.inputs[1].camera_y, d_ori.inputs[1].camera_z ,d_ori.inputs[1].line_rad_xy[i], d_ori.inputs[1].line_rad_xz[i]]
        
        for i in range(self.SIMULATE_TEST_LEN):
            d_list_ans[i] = [d_ori.curvePoints[i].x, d_ori.curvePoints[i].y, d_ori.curvePoints[i].z]
            d_list_t[i] = d_ori.curveTimestamps[i]
        
        return torch.tensor(d_list_r, device=self.device), torch.tensor(d_ori.inputs[0].seq_len, device=self.device), torch.tensor(d_list_l, device=self.device), torch.tensor(d_ori.inputs[1].seq_len, device=self.device), torch.tensor(d_list_t, device=self.device), torch.tensor(d_list_ans, device=self.device)
    
    def saveToFile(self):
        pass

def merge(a, b, out) :
    #lib.merge(a.encode('utf-8'), b.encode('utf-8'), out.encode('utf-8'))
    pass

def testPutData():
    d = BallDataSet_sync("t.bin", dataLength=2)
    for i in range(2) :
        a = d.DataStruct()
        a.inputs[0].camera_x = i
        a.inputs[0].camera_y = 2
        a.inputs[0].camera_z = 3
        a.inputs[0].line_rad_xy[0] = 4
        a.inputs[0].line_rad_xy[1] = 5
        a.inputs[0].line_rad_xy[2] = 6
        a.inputs[0].line_rad_xz[0] = 7
        a.inputs[0].line_rad_xz[1] = 8
        a.inputs[0].line_rad_xz[2] = 9
        a.inputs[0].timestamps[0] = 10
        a.inputs[1].camera_x = 11
        a.inputs[1].camera_y = 12
        a.inputs[1].camera_z = 13
        a.inputs[1].line_rad_xy[0] = 14
        a.inputs[1].line_rad_xy[1] = 15
        a.inputs[1].line_rad_xy[2] = 16
        a.inputs[1].line_rad_xz[0] = 17
        a.inputs[1].line_rad_xz[1] = 18
        a.inputs[1].line_rad_xz[2] = 19
        a.inputs[1].timestamps[0] = 20
        a.curvePoints[0].x = 21
        a.curvePoints[0].y = 22
        a.curvePoints[1].x = 23
        a.curvePoints[1].y = 24
        a.curveTimestamps[0] = 25
        d.putData(i, a)

def testLoadData():
    a = BallDataSet_sync("t.bin")
    print("load success")
    print(len(a))
    print(a[0].inputs[0].camera_x)
    print(a[0].inputs[0].camera_z)
    print(a[1].inputs[0].camera_x)
    print(a[1].curveTimestamps[0])
    print(a[1].curveTimestamps[1])
if __name__ == "__main__":
    ds = BallDataSet_sync("ball_simulate_v2/dataset/medium_fit.train.bin")
    a = ds[0]
    b = ds[1]


    pass