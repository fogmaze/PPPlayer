from ctypes import *
import torch
import platform
import sys
import os
sys.path.append(os.getcwd())
import core.Constants as c

# check os
if platform.system() == "Windows":
    lib = cdll.LoadLibrary("build/dataFileOperator.dll")
elif platform.system() == "Linux":
    lib = cdll.LoadLibrary("build/libdataFileOperator.so")
else:
    raise Exception("Unsupport os!")

class Data_Point(Structure):
    _fields_ = [
        ("x",c_double),
        ("y",c_double)
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


class BallDataSet(torch.utils.data.Dataset):
    def __init__(self, fileName):
        if not os.path.exists(fileName):
            #cerate file
            print("create file")
            self.data = lib.createHeader(c.SIMULATE_TEST_LEN)
        else:
            self.data = lib.loadFromFile(fileName.encode('utf-8'))
        self.filename = fileName
        if lib.loadIsSuccess(self.data):
            self.length = lib.getFileDataLength(self.data)
        else:
            raise Exception("Load data failed!")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return DataStruct.from_address(lib.getFileData(self.data, index))

    def __del__(self):
        lib.releaseData(self.data)
        print("release data")

    def putData(self, index:int, data:DataStruct):
        return lib.putData(self.data, index, data)
    
    def saveToFile(self):
        return lib.saveToFile(self.data, self.filename.encode('utf-8'))

if __name__ == "__main__":
    d = BallDataSet("t.bin")
    
    pass