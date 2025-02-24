import json
from utils.io import load_json, dump_json
from template import *
import argparse


def reformat(resu_list):

    data_list = []
    for index, entry in enumerate(resu_list):
        acts = entry['acts']
        imgs = entry['imgs']
        for i, act in enumerate(acts):
            if act.startswith('Wait'):
                sub = ''
            elif act.startswith('Check'):
                sub = 'Check if the task is finished'
            elif act.startswith('Click'):
                sub = act
            elif act.startswith('Scroll right'):
                sub = 'Swipe from right to left to view'
            elif act.startswith('Go back'):
                sub = 'Go to the previous page'
            elif act.startswith('Scroll down'):
                sub = 'Swipe up for more'
            elif act.startswith('Open'):
                sub = 'open' + act[9:] + ' app'
            else:
                sub = act


            query = template_train_ll.format(ins=entry['ins_pre'], sub_ins=sub)
            query = template_train_ll.format(ins=entry['ins_pre'], sub_ins=entry['sub_ins_gt'][i])
            data_list.append({'query': query,
                                     'response': acts[i],
                                     'images': imgs[i]})
    return data_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2-7b")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--choice", type=int, default=0)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--merge", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()


    data_path = f'/ailab/user/wangwenhao/ms-swift/output/data_format/alg{args.choice}_train_{args.sample}_v1.json'
    output_path = f'/ailab/user/wangwenhao/ms-swift/output/low_new/alg{args.choice}_train_{args.sample}_v2.json'


    def read_json(path):
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data

    data_list = reformat(read_json(data_path))
    dump_json(data_list, output_path)

