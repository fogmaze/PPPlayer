import os
import random

def replaceDir(base_path, dir_name) :
    if dir_name[-1] == '/' or dir_name[-1] == '\\':
        dirname = dir_name[:-1]
    else :
        dirname = dir_name

    old_name = os.path.join(base_path, dirname)
    while os.path.isdir(os.path.join(base_path, dirname)) :
        old_name = os.path.join(base_path, dirname+"_"+str(random.randint(0, 1000)))
    if old_name != os.path.join(base_path, dirname) :
        os.rename(os.path.join(base_path, dirname), os.path.join(base_path, old_name))
        print("Renamed {} to {}".format(os.path.join(base_path, dirname), os.path.join(base_path, old_name)))
    os.mkdir(os.path.join(base_path, dirname))