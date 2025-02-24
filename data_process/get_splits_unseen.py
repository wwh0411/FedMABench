import random
from utils.io import dump_json, load_json, dump_jsonl, load_jsonl
from collections import defaultdict
import re
import json
from template import *
from tqdm import tqdm

# load data
test_data = '/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/unpack-androidcontrol-vlm-train_wsub-8652.jsonl'
json_data = load_jsonl(test_data)
instructions_dict = defaultdict(lambda: {"acts": [], "images": [], "sub_instructions": []})

# 遍历所有数据并按 instruction 进行分类
for entry in json_data:
    instruction = entry["instruction"]
    response = entry["response"]
    image_abs = entry["image_abs"]
    sub_instruction = entry["sub_instruction"]

    instructions_dict[instruction]["acts"].append(response)
    instructions_dict[instruction]["images"].append(image_abs)
    instructions_dict[instruction]["sub_instructions"].append(sub_instruction)


    # 使用正则表达式提取六位数字的 episode ID
    if "episode_id" not in instructions_dict[instruction].keys():
        match = re.search(r'androidcontrol_\d+/unpack-androidcontrol/(\d{6})/', image_abs)

        if match:
            episode_id = match.group(1)
            instructions_dict[instruction]["episode_id"] = episode_id

instructions = list(instructions_dict.keys())
six_ep_list = [instructions_dict[x]["episode_id"] for x in instructions]
int_ep_list = [int(x) for x in six_ep_list]
print(len(six_ep_list))
    # 假设新文件中包含的 episode ID 的 list
new_file_path = "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/android_control_splits.json"  # 替换为你的新文件路径
train_list = load_json(new_file_path)['train']
test = load_json('/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/android_control_test_subsplits.json')
app_list = test["app_unseen"]
cate_list = test["category_unseen"]
task_list = test["task_unseen"]
iid_list = test["IDD"]


def get_train_format(resu_list, level='high'):
    data_list = []
    data_origin_list = []

    for index, entry in enumerate(resu_list):
        # print(entry)
        # if index == choose:
        #     break
        # print('ins：', instruction)
        acts = entry["acts"]
        imgs = entry["imgs"]
        for i in range(entry['num_step']):
            if level == 'high':
                query = template_train_hl.format(ins=entry['ins_pre'])
                query_gt = template_train_hl.format(ins=entry['ins_gt'])
            else:
                if entry["des_acts"]:
                    query = template_train_ll.format(ins=entry['ins_pre'], sub_ins=entry["des_acts"][i])
                else:
                    query = template_train_ll.format(ins=entry['ins_pre'], sub_ins='')
                query_gt = template_train_ll.format(ins=entry['ins_gt'], sub_ins=entry["sub_ins_gt"][i])
            data_list.append({'query': query,
                              'response': acts[i],
                              'images': imgs[i]})
            data_origin_list.append({'query': query_gt,
                                     'response': acts[i],
                                     'images': imgs[i]})
    return data_list, data_origin_list


def main(train_list, select_num, train=False):
    # 从六位 ID 中筛选出同时存在于新文件 list 中的 ID
    if train:
        common_ids = [eid for eid in int_ep_list[:5000] if eid in train_list]
    else:
        common_ids = [eid for eid in int_ep_list if eid in train_list]



    # 随机选择 1000 个
    selected_ids = random.sample(common_ids, select_num) if len(common_ids) >=select_num else common_ids

    # 输出结果
    print(f"共找到 {len(common_ids)} 个匹配的 ID，已随机选择 {len(selected_ids)} 个。")
    print("选择的 ID:", selected_ids)

    # 将选出的 1000 个 ID 保存到文件
    # output_file_path = "selected_ids.json"  # 替换为你的输出文件路径
    # with open(output_file_path, "w") as output_file:
    #     json.dump(selected_ids, output_file)
    if train:
        origin_file = '/ailab/user/wangwenhao/ms-swift/output/qwen2-7b/data_format/alg4_train_7000_v1.json'
        original_resu = load_json(origin_file)
        sampled_resu = []
        for index, instruction in tqdm(enumerate(instructions)):
            if int(instructions_dict[instruction]["episode_id"]) in selected_ids:
                sampled_resu.append(original_resu[index])
                assert original_resu[index]["ins_gt"] == instruction

        # dump_json(sampled_resu, "/ailab/user/wangwenhao/ms-swift/output/generalize/train_1000.json")



        data_list, data_origin_list = get_train_format(sampled_resu, 'high')
        dump_json(data_list, f'/ailab/user/wangwenhao/ms-swift/output/generalize/alg4_high_train_{select_num}.json')
        dump_json(data_origin_list, f'/ailab/user/wangwenhao/ms-swift/output/generalize/gt_high_train_{select_num}.json')
        data_list, data_origin_list = get_train_format(sampled_resu, 'low')
        dump_json(data_list, f'/ailab/user/wangwenhao/ms-swift/output/generalize/alg4_low_train_{select_num}.json')
        dump_json(data_origin_list, f'/ailab/user/wangwenhao/ms-swift/output/generalize/gt_low_train_{select_num}.json')
    else:
        high_list = []
        low_list = []
        for index, instruction in tqdm(enumerate(instructions)):
            if int(instructions_dict[instruction]["episode_id"]) in selected_ids:
                acts = instructions_dict[instruction]["acts"]
                imgs = instructions_dict[instruction]["images"]
                sub_ins = instructions_dict[instruction]["sub_instructions"]
                for i in range(len(acts)):
                    high_list.append({'query': template_train_hl.format(ins=instruction),
                                      'response': acts[i],
                                      'images': imgs[i]})
                    low_list.append({'query': template_train_ll.format(ins=instruction, sub_ins=sub_ins[i]),
                                      'response': acts[i],
                                      'images': imgs[i]})
        dump_json(high_list, '/ailab/user/wangwenhao/ms-swift/output/generalize/iid_gt_high_val_60.json')
        dump_json(low_list, '/ailab/user/wangwenhao/ms-swift/output/generalize/iid_gt_low_val_60.json')

main(train_list, 5000, train=True)
# main(iid_list, 60, train=False)