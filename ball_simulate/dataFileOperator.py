from ctypes import *
import torch

ll = cdll.LoadLibrary
lib = ll("build/libdataFileOperator.so")

lib.getFileDataLength.argtypes = [c_char_p]
lib.getFileDataLength.restype = c_int

def get_data_size(path):
    return lib.getFileDataLength(path.encode('utf-8'))

class Data(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        return None


if __name__ == "__main__":
    print(get_data_size("data/1.bin"))