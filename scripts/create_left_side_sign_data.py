import json
import os
import glob
import open3d as o3d
import numpy as np
import h5py

import sys
import subprocess

from create_bogacz_data import *

NUM_POINT = 32768

json_file = "../dataset/left_UrIII_data_2021.json"
data_key_temp = json.load(open(json_file, mode="r"), strict=False)

data_list_train = []
class_list_train = []

data_list_test = []
class_list_test = []

all_zips = glob.glob("../dataset/*.zip")
all_zips.sort()

for zip_file in all_zips:

    os.system("mkdir ../temp")

    cmd = ["unzip", "-l", zip_file]
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]

    all_files = str(output).split("\\n")[3:-3]

    for afile in all_files:
        afile_rep = " ".join(afile.split("/")[-1].split("_")[:2])

        print(afile_rep)

        for data in data_key_temp:
            if data["file"] == afile_rep:

                os.system(
                    "unzip " + zip_file + " " + afile.split("   ")[-1] + " -d ../temp/."
                )

                pcd_o3d = o3d.io.read_point_cloud("../temp/" + afile.split("   ")[-1])
                pcd = downScalePcdNormal(pcd_o3d, NUM_POINT)

                pcd = pc_normalize(pcd)

                if data["type"] == "train":
                    data_list_train.append(pcd)
                    class_list_train.append(data["class"])
                else:
                    data_list_test.append(pcd)
                    class_list_test.append(data["class"])
                print(data["file"], data["class"], data["type"])
    os.system("rm -r ../temp")


f = h5py.File("../output/cuneiform_file_o3d_left_train.h5", "w")
f.create_dataset("data", (len(class_list_train), NUM_POINT, 6), dtype="f4")
f.create_dataset("label", (len(class_list_train), 1), dtype="i")

f["data"][:] = np.array(data_list_train)
f["label"][:] = np.reshape(
    np.array(class_list_train, np.int), (len(class_list_train), 1)
)
f.close()

f = h5py.File("../output/cuneiform_file_o3d_left_test.h5", "w")
f.create_dataset("data", (len(class_list_test), NUM_POINT, 6), dtype="f4")
f.create_dataset("label", (len(class_list_test), 1), dtype="i")

f["data"][:] = np.array(data_list_test)
f["label"][:] = np.reshape(np.array(class_list_test, np.int), (len(class_list_test), 1))
f.close()
