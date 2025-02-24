import os
import re
import json
# 设置文件夹路径
folder_path = r'/ailab/user/wangwenhao/FedMobile/bash/output/qwen2-vl-7b-instruct'
# folder_path = r'/ailab/user/wangwenhao/FedMobile/bash/output/internvl2-2b'


# 获取文件夹中的所有子文件夹（目录）
directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

# 定义版本号筛选规则，提取文件夹名中的版本号
filtered_dirs = []
for dir_name in directories:
    # 使用正则表达式匹配以 'v' 开头并提取版本号
    match = re.match(r"^v(\d+)-", dir_name)
    if match:
        version = int(match.group(1))  # 提取版本号并转换为整数
        if version > 249:  # 筛选版本号大于 40 的文件夹
            filtered_dirs.append(dir_name)

# 对筛选后的文件夹名按版本号排序（升序）
filtered_dirs.sort(key=lambda x: int(re.match(r"^v(\d+)-", x).group(1)))

# 输出筛选后的文件夹名
for dir_name in filtered_dirs:
    print(dir_name)

# 遍历筛选后的文件夹，读取其中的 sft_args.json 文件
for dir_name in filtered_dirs:
    # 构造 sft_args.json 文件的路径
    sft_args_path = os.path.join(folder_path, dir_name, 'sft_args.json')

    # 检查文件是否存在
    if os.path.exists(sft_args_path):
        # 打开并读取 JSON 文件
        with open(sft_args_path, 'r', encoding='utf-8') as f:
            sft_args = json.load(f)

        # 获取需要的字段
        fed_alg = sft_args.get('fed_alg', 'N/A')
        client_num = sft_args.get('client_num', 'N/A')
        client_sample = sft_args.get('client_sample', 'N/A')
        dataset = sft_args.get('dataset', 'N/A')

        # 打印文件夹名和对应的字段
        print(f"Folder: {dir_name}")
        print(f"  fed_alg: {fed_alg}")
        print(f"  client_num: {client_num}")
        print(f"  client_sample: {client_sample}")
        print(f"  dataset: {dataset}")
        print("-" * 40)  # 分隔符，方便阅读
