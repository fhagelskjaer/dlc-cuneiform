import json
import os, sys
import glob
import open3d as o3d
import numpy as np
import h5py

def pc_normalize(pc):
    centroid = np.mean(pc[:, :3], axis=0)
    pc[:, :3] = pc[:, :3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    pc[:, :3] = pc[:, :3] / m
    return pc


def downScalePcdNormal(pcd, num_point):
    pcd.estimate_normals()
    size = 0.10
    while np.asarray(pcd.points).shape[0] > 65536:
        pcd = pcd.voxel_down_sample(size)
        # print( size, np.asarray(pcd.points).shape[0] )
        size += 0.01
    pointcloud_pointnet = np.concatenate(
        [np.asarray(pcd.points), np.asarray(pcd.normals)], axis=1
    )  #  [
    # print( pointcloud_pointnet.shape )
    index_list = list(range(len(pointcloud_pointnet)))
    np.random.shuffle(index_list)
    pointcloud_pointnet = pointcloud_pointnet[index_list[:num_point], :]
    return pointcloud_pointnet


def get_excluded_data(json_obj, classes, test_files):
    data_key = []
    for key in json_obj.keys():
        if "from_cdli_search_view" in json_obj[key]:
            if json_obj[key]["from_cdli_search_view"]["period"] in classes:
                print(key, key in test_files)
                if not key in test_files:
                    data_key.append(
                        {
                            "file": key,
                            "class": json_obj[key]["from_cdli_search_view"]["period"],
                        }
                    )
    print(len(data_key))
    return data_key


def get_included_data(json_obj, classes, test_files):
    data_key = []
    for key in json_obj.keys():
        if "from_cdli_search_view" in json_obj[key]:
            if json_obj[key]["from_cdli_search_view"]["period"] in classes:
                print(key, key in test_files)
                if key in test_files:
                    data_key.append(
                        {
                            "file": key,
                            "class": json_obj[key]["from_cdli_search_view"]["period"],
                        }
                    )
    print(len(data_key))
    return data_key


if __name__ == "__main__":

    task = sys.argv[1]

    classes = [
        "ED IIIb (ca. 2500-2340 BC)",
        "Old Assyrian (ca. 1950-1850 BC)",
        "Old Babylonian (ca. 1900-1600 BC)",
        "Ur III (ca. 2100-2000 BC)",
    ]

    NUM_POINT = 32768

    json_file = "../dataset/HeiCuBeDa_B_Hilprecht_Database_190318.json"  # the full description of all tablets
    json_obj = json.load(open(json_file, mode="r"), strict=False)

    json_file_test = "../dataset/bogacz_icfhr_2020_figures_heicubeda_test.json"  # the test files from the original paper
    json_obj_test = json.load(open(json_file_test, mode="r"), strict=False)
    test_files = []
    for number in json_obj_test["numbers"]:
        test_files.append(json_obj_test["categories_number"][number])

    json_file_train = "../dataset/bogacz_icfhr_2020_figures_heicubeda_train.json"  # the training files used in the original paper
    json_obj_train = json.load(open(json_file_train, mode="r"), strict=False)
    train_files = []
    for number in json_obj_train["numbers"]:
        train_files.append(json_obj_train["categories_number"][number])

    if task == "test":
        print("Test Dataset")
        data_key = get_included_data(
            json_obj, classes, test_files
        )
        output_name = "../output/cuneiform_file_o3d_test.h5"
        not_xl = True
    elif task == "train":
        print("Train Small Dataset")
        data_key = get_included_data(
            json_obj, classes, train_files
        )
        output_name = "../output/cuneiform_file_o3d_train.h5"
        not_xl = True
    elif task == "train_l":
        print("Train Large Dataset")
        data_key = get_excluded_data(
            json_obj, classes, test_files
        )
        output_name = "../output/cuneiform_file_o3d_train_l.h5"
        not_xl = True
    else:
        print("No valid task")
        sys.exit()

    data_list = []
    class_list = []

    all_zips = glob.glob("../dataset/*.zip")
    all_zips.sort()

    print(all_zips)

    for zip_file in all_zips:
        os.system("mkdir ../temp")
        os.system(
            "unzip " + zip_file + " -d ../temp/." + " >/dev/null 2>&1"
        )

        all_files = glob.glob("../temp/*/*/*.ply")

        for afile in all_files:
            afile_rep = " ".join(afile.split("/")[-1].split("_")[:2])
            for data in data_key:
                if data["file"] == afile_rep:

                    info = os.stat(afile)
                    if info.st_size > (300 * 1024 * 1024):
                        continue

                    pcd_o3d = o3d.io.read_point_cloud(afile)
                    pcd = downScalePcdNormal(pcd_o3d, NUM_POINT)

                    pcd = pc_normalize(pcd)
                    data_list.append(pcd)
                    class_list.append(classes.index(data["class"]))

        os.system("rm -r ../temp")

    f = h5py.File(output_name, "w")
    f.create_dataset("data", (len(class_list), NUM_POINT, 6), dtype="f4")
    f.create_dataset("label", (len(class_list), 1), dtype="i")

    f["data"][:] = np.array(data_list)
    class_list = np.array(class_list, np.int)
    class_list = np.reshape(class_list, (len(class_list), 1))
    f["label"][:] = class_list
    f.close()
