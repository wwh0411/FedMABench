import os
from typing import Literal
import json
from collections import defaultdict
from tqdm import tqdm
import argparse
from template import *
from utils.io import dump_json, load_json, dump_jsonl, load_jsonl


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
    base_path = '/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/sization/episode_with_client'
    client_num_choices = [10]
    sample_num_choices = [1000, 3000, 5000, 7000]
    hete_type_choices = ['random', 'category', 'dirichlet']
    train_val_choices = ['train', 'val']
    ratio = 0.1

    # get train & val
    for client_num in client_num_choices:
        for sample_num in sample_num_choices:
            for hete_type in hete_type_choices:
                if hete_type == 'random':
                    arg = 0
                elif hete_type == 'category':
                    arg = 2
                elif hete_type == 'dirichlet':
                    arg = '0.50'
                else:
                    arg = 0
                for train_val in train_val_choices:
                    if train_val == 'val':
                        file_name = f"c{client_num}_n{int(sample_num * ratio)}_{train_val}_{hete_type}_x{arg}.jsonl"
                    else:
                        file_name = f"c{client_num}_n{sample_num}_{train_val}_{hete_type}_x{arg}.jsonl"
                        resu_list = load_jsonl(os.path.join(base_path, file_name))
                        _, data = get_train_format(resu_list, 'high')
                        dump_json(data, os.path.join('./output/high', file_name))
                        _, data = get_train_format(resu_list, 'low')
                        dump_json(data, os.path.join('./output/low', file_name))
