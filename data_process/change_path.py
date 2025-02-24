import os
from utils.io import dump_json, load_json, dump_jsonl, load_jsonl

def change_img_path2(data_origin_list):
    for entry in data_origin_list:
        entry['images'] = entry['images'].replace("/GPFS/data/wenhaowang-1/FedMobile/androidcontrol_1108/unpack-1109", "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol")

    return data_origin_list

directory = '/ailab/user/wangwenhao/ms-swift/ablation'
for path in os.listdir(directory):
    resu_list = load_json(os.path.join(directory, path))
    data = change_img_path2(resu_list)
    dump_json(data, os.path.join(directory, path))