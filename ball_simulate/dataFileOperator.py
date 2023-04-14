from ctypes import *
import torch
import platform

# check os
if platform.system() == "Windows":
    lib = cdll.LoadLibrary("build/dataFileOperator.dll")
elif platform.system() == "Linux":
    lib = cdll.LoadLibrary("build/libdataFileOperator.so")
else:
    raise Exception("Unsupport os!")

class DataStruct(Structure):
    _fields = [
        ("data", c_int)
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
lib.getFileData.restype = DataStruct

class BallDataSet(torch.utils.data.Dataset):
    def __init__(self, fileName):
        self.data = lib.loadFromFile(fileName.encode('utf-8'))
        if lib.loadIsSuccess(self.data):
            self.length = lib.getFileDataLength(self.data)
        else:
            raise Exception("Load data failed!")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return lib.getFileData(self.data, index)

    def __del__(self):
        lib.releaseData(self.data)
        print("release data")

if __name__ == "__main__":
    ballDataSet = BallDataSet("test.bin")
    pass