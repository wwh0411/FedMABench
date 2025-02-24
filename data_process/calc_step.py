import json
from utils.io import *
from template import *
# 定义一个函数来计算平均 action 数量
def calculate_average_actions(file_path):
    total_actions = 0
    episode_count = 0

    # 打开文件并读取每一行
    with open(file_path, 'r') as file:
        for line in file:
            # 解析每行 JSON 数据
            episode = json.loads(line.strip())

            # 选择 'acts_origin' 或 'acts_convert' 来计算 action 数量
            # 如果你想要计算原始 actions，可以使用 'acts_origin'
            # 如果你想要计算转换后的 actions，可以使用 'acts_convert'
            actions = episode.get("acts_origin", [])  # 或者 'acts_convert'

            # 统计每个 episode 的 action 数量
            total_actions += len(actions)
            episode_count += 1

    # 计算并返回平均 action 数量
    if episode_count == 0:
        return 0
    return total_actions / episode_count, episode_count


# 使用示例
file_path = '/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/episode-wise-all.jsonl'  # 替换为你的 JSON 文件路径
file_path = '/ailab/user/wangwenhao/ms-swift/episode-wise-web_shopping.jsonl'
average_actions, epi = calculate_average_actions(file_path)
print(file_path)
print(f"Average number of actions per episode: {average_actions:.2f}")
print(epi)


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

            try:
                data_origin_list.append({'query': query_gt,
                                     'response': acts[i],
                                     'images': imgs[i],
                                     'client_id': entry['client_id']})
            except:
                data_origin_list.append({'query': query_gt,
                                         'response': acts[i],
                                         'images': imgs[i],
                                         })
    return data_list, data_origin_list
# 初始化五个子集
subset_1 = []
subset_2 = []
subset_3 = []
subset_4 = []
subset_5 = []

# 读取 JSON 文件并处理每行
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每行 JSON 数据
        data = json.loads(line.strip())

        # 统计 step 数量，这里以 "acts_origin" 为依据
        step_count = len(data.get("acts_origin", []))

        # 根据 step 数量划分到对应的子集
        if 1 <= step_count <= 2:
            subset_1.append(data)
        elif 3 <= step_count <= 7:
            subset_2.append(data)
        elif 8 <= step_count <= 12:
            subset_3.append(data)
        elif 13 <= step_count <= 17:
            subset_4.append(data)
        elif step_count >= 18:
            subset_5.append(data)

# 输出划分结果
print(f"子集 1 (1-2 steps): {len(subset_1)} 条数据")
print(f"子集 2 (3-7 steps): {len(subset_2)} 条数据")
print(f"子集 3 (8-12 steps): {len(subset_3)} 条数据")
print(f"子集 4 (13-17 steps): {len(subset_4)} 条数据")
print(f"子集 5 (>=18 steps): {len(subset_5)} 条数据")

# 保存每个子集到独立的文件中
output_files = {
    "subset_1.json": subset_1,
    "subset_2.json": subset_2,
    "subset_3.json": subset_3,
    "subset_4.json": subset_4,
    "subset_5.json": subset_5,
}

# for file_name, subset in output_files.items():
#     dump_jsonl(subset, file_name)

print("子集划分完成，并已保存到对应文件中。")


import copy

# huafen 1
subset_all = []
subset_all_1 = []
subset_all_2 = []
subset_all_3 = []

eval_set = []

# 定义划分的数量比例
split_sizes = [4, 10, 20, 30, 36]  # 每个 client 的数据数量
split_sizes = split_sizes * 2  # 因为分为两轮，每轮 5 份

for index, (file_name, subset) in enumerate(output_files.items()):
    # 获取前 200 个数据
    selected_data = subset[:200]
    subset_all.extend(selected_data)
    eval_set.extend(subset[300:320])
    # 第一部分：前 100 个标记 client_id 为 index
    for x in selected_data[:100]:
        x['client_id'] = index

    # 第二部分：后 100 个标记 client_id 为 index + 5
    for x in selected_data[100:200]:
        x['client_id'] = index + 5

    # 添加到 subset_all_1
    subset_all_1.extend(copy.deepcopy(selected_data))
    # subset_all_1 = copy.deepcopy(subset_all)

    # 再标记前 200 个数据，每 10 个循环分配 client_id
    for i, x in enumerate(selected_data):
        x['client_id'] = i % 10
    subset_all_2.extend(copy.deepcopy(selected_data))
    # subset_all_2 = copy.deepcopy(subset_all)
    # 新部分：将数据划分成 10 份，标记 client_id
    start = 0
    for client_id in range(10):  # client_id 0-9
        size = split_sizes[client_id % 5]  # 循环使用 split_sizes 的数量
        chunk = selected_data[start:start + size]  # 按数量取出数据
        for x in chunk:
            x['client_id'] = client_id  # 标记 client_id
        subset_all_3.extend(copy.deepcopy(chunk))  # 添加到 subset_all_3
        start += size  # 更新起始位置
    # subset_all_3 = copy.deepcopy(subset_all)
    #

import json
import copy

# 初始化 subset4
subset_all_4 = []


def split_subset_balanced_v2(subset, target_parts=10, step_tolerance=0.2, min_data_per_part=50):
    """
    改进的分配方法：确保每个子集的 step 总和误差不超过给定比例，且每个子集的数据数量不低于 min_data_per_part。
    """
    # 按照 step 数量排序，保证大的 episode 优先分配
    sorted_subset = sorted(subset, key=lambda x: len(x.get('acts_origin', [])), reverse=True)

    # 计算总 step 数量和目标 step 数量
    total_steps = sum(len(x.get('acts_origin', [])) for x in sorted_subset)
    target_step_sum = total_steps / target_parts
    tolerance_range = (1 - step_tolerance) * target_step_sum, (1 + step_tolerance) * target_step_sum

    # 初始化子集
    parts = [[] for _ in range(target_parts)]
    parts_steps = [0] * target_parts  # 每个子集的 step 总和
    parts_counts = [0] * target_parts  # 每个子集的 episode 数量

    # 贪心分配数据
    for episode in sorted_subset:
        step_count = len(episode.get('acts_origin', []))

        # 找到最优子集：优先考虑 step 平衡，其次考虑数据数量不足
        best_index = min(
            range(target_parts),
            key=lambda i: (
                abs(parts_steps[i] + step_count - target_step_sum),  # step 总和平衡
                parts_counts[i]  # 优先分配到数据量较少的子集
            )
        )

        # 将 episode 分配到该子集中
        parts[best_index].append(episode)
        parts_steps[best_index] += step_count
        parts_counts[best_index] += 1

    # 后处理：确保每个子集数据量不低于 min_data_per_part
    for i in range(target_parts):
        while len(parts[i]) < min_data_per_part:
            # 从数据量最多的子集移动数据
            max_index = parts_counts.index(max(parts_counts))
            if max_index == i or len(parts[max_index]) <= min_data_per_part:
                break  # 如果没有足够的数据可以移动，退出平衡
            # 移动一个 episode
            episode_to_move = parts[max_index].pop()
            step_count = len(episode_to_move.get('acts_origin', []))
            parts[i].append(episode_to_move)
            parts_steps[i] += step_count
            parts_steps[max_index] -= step_count
            parts_counts[i] += 1
            parts_counts[max_index] -= 1

    # 检查分配结果
    for i, part in enumerate(parts):
        step_sum = parts_steps[i]
        num_data = len(part)
        avg_step_length = step_sum / num_data if num_data > 0 else 0
        print(f"Client ID {i}:")
        print(f"  Number of Data: {num_data}")
        print(f"  Step Sum: {step_sum}")
        print(f"  Average Step Length: {avg_step_length:.2f}")

    return parts


# 将 subset 数据划分成 10 份
subset_parts = split_subset_balanced_v2(subset_all, target_parts=10)

# 标记 client_id 并保存到 subset4
for client_id, part in enumerate(subset_parts):
    for episode in part:
        episode["client_id"] = client_id
        subset_all_4.append(episode)




print(subset_all_1[0])
print(subset_all_1[100])

# 统计每个 client_id 对应的数据数量和平均 step 长度
client_stats = {}

for data in subset_all_4:
    client_id = data['client_id']
    step_length = len(data.get('acts_origin', []))  # 获取 step 长度

    if client_id not in client_stats:
        client_stats[client_id] = {'count': 0, 'total_steps': 0}

    client_stats[client_id]['count'] += 1
    client_stats[client_id]['total_steps'] += step_length

# 打印每个 client_id 的统计结果
print("Client ID Statistics:")
for client_id in sorted(client_stats.keys()):
    count = client_stats[client_id]['count']
    total_steps = client_stats[client_id]['total_steps']
    avg_steps = total_steps / count if count > 0 else 0
    print(f"Client ID {client_id}:")
    print(f"  Number of Data: {count}")
    print(f"  Average Step Length: {avg_steps:.2f}")

# _, data = get_train_format(subset_all_1, 'high')
# dump_json(data, 'step_skew.json')
#
# _, data = get_train_format(subset_all_2, 'high')
# dump_json(data, 'step_iid.json')
#
# _, data = get_train_format(subset_all_3, 'high')
# dump_json(data, 'epi_skew.json')

_, data = get_train_format(subset_all_4, 'high')
dump_json(data, 'step_sum.json')

_, data = get_train_format(eval_set, 'high')
dump_json(data, 'eval.json')
