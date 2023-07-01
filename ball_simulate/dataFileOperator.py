import torch
from ctypes import *
import torch
import platform
import sys
import os
sys.path.append(os.getcwd())
import core.Constants as c
import threading

# check os
if platform.system() == "Windows":
    lib = cdll.LoadLibrary("build/dataFileOperator.dll")
elif platform.system() == "Linux":
    lib = CDLL("build/libdataFileOperator.so")
else:
    raise Exception("Unsupport os!")

class Data_Point(Structure):
    _fields_ = [
        ("x",c_double),
        ("y",c_double),
        ("z",c_double)
    ]

class Data_Input(Structure):
    _fields_ = [
        ("camera_x", c_double),
        ("camera_y", c_double),
        ("camera_z", c_double),
        ("line_rad_xy", c_double * c.SIMULATE_INPUT_LEN),
        ("line_rad_xz", c_double * c.SIMULATE_INPUT_LEN),
        ("timestamps", c_double * c.SIMULATE_INPUT_LEN),
    ]

class DataStruct(Structure):
    _fields_ = [
        ("inputs", Data_Input * 2),
        ("curvePoints", Data_Point * c.SIMULATE_TEST_LEN),
        ("curveTimestamps", c_double * c.SIMULATE_TEST_LEN)
    ]

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
lib.releaseData_sync.argtypes = [c_void_p]
lib.releaseData_sync.restype = None
lib.getFileDataLength_sync.argtypes = [c_char_p]
lib.getFileDataLength_sync.restype = c_int


class BallDataSet(torch.utils.data.Dataset) :
    def __init__(self, fileName, dataLength = None, device = "cuda:0"):
        if not os.path.exists(fileName):
            #cerate file
            print("create file")
            if dataLength == None:
                raise Exception("dataLength can't be None!")
            self.data = lib.createHeader(dataLength)
        elif dataLength != None:
            print("create file")
            self.data = lib.createHeader(dataLength)
        else:
            self.data = lib.loadFromFile(fileName.encode('utf-8'))
        self.filename = fileName
        if lib.loadIsSuccess(self.data):
            self.length = lib.getFileDataLength(self.data)
        else:
            raise Exception("Load data failed!")
        self.device = torch.device(device)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        d_ori = DataStruct.from_address(lib.getFileData(self.data, index))
        d_list_r = [None] * c.SIMULATE_INPUT_LEN
        d_list_l = [None] * c.SIMULATE_INPUT_LEN
        d_list_t = [None] * c.SIMULATE_TEST_LEN
        d_list_ans = [None] * c.SIMULATE_TEST_LEN
        for i in range(c.SIMULATE_INPUT_LEN):
            d_list_r[i] = [d_ori.inputs[0].camera_x, d_ori.inputs[0].camera_y, d_ori.inputs[0].camera_z ,d_ori.inputs[0].line_rad_xy[i], d_ori.inputs[0].line_rad_xz[i]]
            d_list_l[i] = [d_ori.inputs[1].camera_x, d_ori.inputs[1].camera_y, d_ori.inputs[1].camera_z ,d_ori.inputs[1].line_rad_xy[i], d_ori.inputs[1].line_rad_xz[i]]
        
        for i in range(c.SIMULATE_TEST_LEN):
            d_list_ans[i] = [d_ori.curvePoints[i].x, d_ori.curvePoints[i].y, d_ori.curvePoints[i].z]
            d_list_t[i] = d_ori.curveTimestamps[i]
        
        return torch.tensor(d_list_r, device=self.device), torch.tensor(d_list_l, device=self.device), torch.tensor(d_list_t, device=self.device), torch.tensor(d_list_ans, device=self.device)

    def __del__(self):
        lib.releaseData(self.data)
        print("release data")

    def putData(self, index:int, data:DataStruct):
        return lib.putData(self.data, index, data)
    
    def saveToFile(self):
        return lib.saveToFile(self.data, self.filename.encode('utf-8'))


class BallDataSet_sync(torch.utils.data.Dataset) :
    def __init__(self, fileName, device = "cuda:0"):
        self.fileName = fileName
        self.device = torch.device(device)
    
    def __len__(self):
        return lib.getFileDataLength_sync(self.fileName.encode('utf-8'))
    
    def __getitem__(self, index):
        d_ori = lib.getFileData_sync(self.fileName.encode('utf-8'), index)
        d_list_r = [None] * c.SIMULATE_INPUT_LEN
        d_list_l = [None] * c.SIMULATE_INPUT_LEN
        d_list_t = [None] * c.SIMULATE_TEST_LEN
        d_list_ans = [None] * c.SIMULATE_TEST_LEN
        for i in range(c.SIMULATE_INPUT_LEN):
            d_list_r[i] = [d_ori.inputs[0].camera_x, d_ori.inputs[0].camera_y, d_ori.inputs[0].camera_z ,d_ori.inputs[0].line_rad_xy[i], d_ori.inputs[0].line_rad_xz[i]]
            d_list_l[i] = [d_ori.inputs[1].camera_x, d_ori.inputs[1].camera_y, d_ori.inputs[1].camera_z ,d_ori.inputs[1].line_rad_xy[i], d_ori.inputs[1].line_rad_xz[i]]
        
        for i in range(c.SIMULATE_TEST_LEN):
            d_list_ans[i] = [d_ori.curvePoints[i].x, d_ori.curvePoints[i].y, d_ori.curvePoints[i].z]
            d_list_t[i] = d_ori.curveTimestamps[i]
        
        return torch.tensor(d_list_r, device=self.device), torch.tensor(d_list_l, device=self.device), torch.tensor(d_list_t, device=self.device), torch.tensor(d_list_ans, device=self.device)

def testPutData():
    d = BallDataSet("t.bin", dataLength=2)
    for i in range(2) :
        a = DataStruct()
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
        print(i)
    d.saveToFile()

def testLoadData():
    a = BallDataSet("t.bin")
    print("load success")
    print(len(a))
    print(a[0].inputs[0].camera_x)
    print(a[0].inputs[0].camera_z)
    print(a[1].inputs[0].camera_x)
    print(a[1].curveTimestamps[0])
    print(a[1].curveTimestamps[1])
if __name__ == "__main__":
    ds = BallDataSet("./ball_simulate/dataset/medium.valid.bin")
    dss = BallDataSet_sync("./ball_simulate/dataset/medium.valid.bin")

    a = ds[7]
    b = dss[7]
    pass