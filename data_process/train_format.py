import os
from typing import Literal
import json
from collections import defaultdict
from tqdm import tqdm
import argparse
from template import *
from utils.io import dump_json, load_json, dum_jsonl, load_jsonl


def get_train_format(resu_list, level='high'):
    data_list = []
    data_origin_list = []

    for index, entry in enumerate(resu_list):
        # if index == choose:
        #     break
        # print('ins：', instruction)
        acts = entry["acts_convert"]
        imgs = entry["imgs"]
        for i in range(len(imgs)):
            if level == 'high':
                query_gt = template_train_hl.format(ins=entry['instruction'])
            else:
                query_gt = template_train_ll.format(ins=entry['instruction'], sub_ins=entry["sub_instructions"][i])

            data_origin_list.append({'query': query_gt,
                                     'response': acts[i],
                                     'images': imgs[i],
                                     'client_id': entry['client_number']})
    return data_list, data_origin_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2-7b")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--choice", type=int, default=0)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # path = "/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/data_process/episode_with_client.jsonl"
    resu_list = load_jsonl(args.data_path)
    _, data = get_train_format(resu_list, 'high')
    dump_json(data, f'./output/high/fed_train_hete1_v1.json')
    _, data = get_train_format(resu_list, 'low')
    dump_json(data, f'./output/low/fed_train_hete1_v1.json')