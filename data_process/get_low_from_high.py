import json
print(0)
from utils.io import load_json, dump_json
print(1)
from template import *
import argparse

def complete_sub_ins_gt(data, reference_data):
    """
    遍历原数据并根据参考数据补全 sub_ins_gt 字段，
    同时替换 imgs 中的路径 ms-swift_old 为 ms-swift。
    """
    completed_data = []

    for i, item in enumerate(data):
        # 从参考数据获取 sub_ins_gt
        if i < len(reference_data) and 'sub_ins_gt' in reference_data[i]:
            item['sub_ins_gt'] = reference_data[i]['sub_ins_gt']

        # 替换 imgs 中的路径
        if 'imgs' in item:
            item['imgs'] = [img.replace("ms-swift_old", "ms-swift") for img in item['imgs']]

        completed_data.append(item)

    return completed_data


def get_low_prompt_from_high(resu_list, is_gt=False):

    data_list = []
    for index, entry in enumerate(resu_list):
        acts = entry["acts"]
        imgs = entry["imgs"]
        for i in range(len(imgs)):
            if is_gt:
                query = template_train_ll.format(ins=entry['ins_gt'], sub_ins='')
            else:
                query = template_train_ll.format(ins=entry['ins_pre'], sub_ins='')
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
    if 'get_low_prompt':

        data_path = f'/ailab/user/wangwenhao/ms-swift/output/{args.model}/data_format/alg{args.choice}_train_{args.sample}_v1.json'
        output_path = f'/ailab/user/wangwenhao/ms-swift/output/{args.model}/low_none/alg{args.choice}_train_{args.sample}_v1.json'

        if args.choice == 0:
            data_path = f'/ailab/user/wangwenhao/ms-swift/output/{args.model}/data_format/alg2_train_{args.sample}_v1.json'
            output_path = f'/ailab/user/wangwenhao/ms-swift/output/{args.model}/low_none/gt_train_{args.sample}_v1.json'

        def read_json(path):
            with open(path, 'r', encoding="utf-8") as f:
                data = json.load(f)
            return data

        data_list = get_low_prompt_from_high(read_json(data_path))
        dump_json(data_list, output_path)
        exit()
    # 文件路径
    original_file_path = '/ailab/user/wangwenhao/ms-swift/output/data_format/alg5_train_5000_v1.json'  # 需要补全的原始数据
    reference_file_path = '/ailab/user/wangwenhao/ms-swift/output/data_format/alg7_train_5000_v1.json'  # 参考数据，用于补全

    # 加载原始数据和参考数据
    original_data = load_json(original_file_path)
    reference_data = load_json(reference_file_path)

    # 补全 sub_ins_gt 并替换 imgs 中的路径
    completed_data = complete_sub_ins_gt(original_data, reference_data)

    dump_json(completed_data, '/ailab/user/wangwenhao/ms-swift/output/data_format/alg5_train_5000_v1.json')
