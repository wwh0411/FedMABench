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


def get_val_format_from_steps():
    test_data = '/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/unpack-androidcontrol-vlm-train_wsub-8652.jsonl'
    json_data = load_jsonl(test_data)
    # 遍历所有数据并按 instruction 进行分类
    resu_list = []
    instructions_dict = defaultdict(lambda: {"acts": [], "images": [], "sub_instructions": []})

    for entry in json_data:
        instruction = entry["instruction"]
        response = entry["response"]
        image_abs = entry["image_abs"]
        sub_instruction = entry["sub_instruction"]

        instructions_dict[instruction]["acts"].append(response)
        instructions_dict[instruction]["images"].append(image_abs)
        instructions_dict[instruction]["sub_instructions"].append(sub_instruction)

    data_list = []
    for index, instruction in tqdm(enumerate(list(instructions_dict.keys())[8000:8200])):
        acts = instructions_dict[instruction]["acts"]
        imgs = instructions_dict[instruction]["images"]
        sub_instructions = instructions_dict[instruction]["sub_instructions"]
        for i in range(len(acts)):
            # data_list.append({'query': template_train_ll.format(ins=instruction, sub_ins=sub_instructions[i]),
            #                   'response': acts[i],
            #                   'images': imgs[i]})
            data_list.append({'query': template_train_ll.format(ins=instruction, sub_ins=''),
                              'response': acts[i],
                              'images': imgs[i]})

    dump_json(data_list, '/ailab/user/wangwenhao/ms-swift/output/gt_val_200_v1_low_none.json')


def get_val_format_for_aitw(cate):
    data_path = f'/ailab/user/wangwenhao/ms-swift/episode-wise-{cate}.jsonl'
    json_data = load_jsonl(data_path)

    json_data = json_data[5000:5100]
    # 遍历所有数据并按 instruction 进行分类
    data_list = []
    instructions_dict = defaultdict(lambda: {"acts": [], "images": [], "sub_instructions": []})

    for entry in json_data:
        instruction = entry["instruction"]
        acts = entry["acts_convert"]
        imgs = entry["imgs"]
        for i in range(len(acts)):
            # data_list.append({'query': template_train_ll.format(ins=instruction, sub_ins=sub_instructions[i]),
            #                   'response': acts[i],
            #                   'images': imgs[i]})
            data_list.append({'query': template_train_hl.format(ins=instruction),
                              'response': acts[i],
                              'images': imgs[i]})



    dump_json(data_list, f'/ailab/user/wangwenhao/ms-swift/output_aitw/{cate}_gt_val_100_v1.json')


def change_img_path(data_origin_list):
    for entry in data_origin_list:
        entry['images'] = entry['images'].replace("/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol", "/GPFS/data/wenhaowang-1/FedMobile/androidcontrol_1108/unpack-1109")

    return data_origin_list

def change_img_path2(data_origin_list):
    for entry in data_origin_list:
        entry['images'] = entry['images'].replace("/GPFS/data/wenhaowang-1/FedMobile/androidcontrol_1108/unpack-1109", "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol")

    return data_origin_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2-7b")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--choice", type=int, default=0)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    # get_val_format_from_steps()
    # cate_list = ['general', 'google_apps', 'install', 'web_shopping', 'singl']
    # cate_list = ['single']
    # for cate in cate_list:
    #     get_val_format_for_aitw(cate)
    path = '/ailab/user/wangwenhao/ms-swift/basic_low_level_val.jsonl'
    resu_list = load_json(path)
    data = change_img_path2(resu_list)
    dump_json(data, path)
    exit()
    # path = "/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/data_process/episode_with_client.jsonl"
    base_path = '/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/sization/episode_with_client_n1'
    client_num_choices = [10]
    sample_num_choices = [1000, 2000, 3000, 5000, 7000]
    hete_type_choices = ['random', 'category', 'dirichlet'][1:2]
    train_val_choices = ['train', 'val']
    ratio = 0.1


    # get train & val
    for client_num in client_num_choices:
        for sample_num in sample_num_choices:
            for hete_type in hete_type_choices:
                if hete_type == 'random':
                    arg = 0
                elif hete_type == 'category':
                    arg = 1
                elif hete_type == 'dirichlet':
                    arg = '0.5'
                else:
                    arg = 0
                for train_val in train_val_choices:
                    if train_val == 'val':
                        file_name = f"c{client_num}_n{int(sample_num * ratio)}_{train_val}_{hete_type}_x{arg}.jsonl"
                    else:
                        file_name = f"c{client_num}_n{sample_num}_{train_val}_{hete_type}_x{arg}.jsonl"
                    try:
                        resu_list = load_jsonl(os.path.join(base_path, file_name))
                    except:
                        continue
                    _, data = get_train_format(resu_list, 'high')
                    dump_json(data, os.path.join('./output/high', file_name))
                    if args.debug:
                        data = change_img_path(data)
                        dump_json(data, os.path.join('./output_cmic/high', file_name))
                    _, data = get_train_format(resu_list, 'low')
                    dump_json(data, os.path.join('./output/low', file_name))
                    if args.debug:
                        data = change_img_path(data)
                        dump_json(data, os.path.join('./output_cmic/low', file_name))


