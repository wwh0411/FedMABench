import os
import json
import argparse
import re
from collections import defaultdict
from tqdm import tqdm
from eval_gpt import calculate_tfidf


def read_jsonl(path):
    """ 读取 JSONL 文件 """
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def read_category_mapping(category_file):
    """ 读取 episode -> category 的映射文件 """
    with open(category_file, 'r', encoding='utf-8') as file:
        category_data = json.load(file)
    return {str(k): v["category"] for k, v in category_data.items()}  # 确保 episode 号是字符串


def judge_step(a, b):
    """ 计算 TF-IDF 相似度，判断是否匹配 """
    return 1 if calculate_tfidf(a, b) > 0.6 else 0


def calculate_step_accuracy(data, category_mapping):
    """
    计算 step 级别的准确率，并按 category 分类统计。
    """
    step_accuracies = []
    category_accuracies = defaultdict(list)

    for item in data:
        # 提取 episode 号
        image_path = item.get("images", "")
        episode_id = image_path.split("/")[-2]  # 假设 episode 号是倒数第二级文件夹

        # 计算 step 级别的准确性
        if item['label'] != 'Click at a button':
            step_accuracy = judge_step(item['label'], item['response'])
            step_accuracies.append(step_accuracy)

            # 获取 category 并存入对应的列表
            category = category_mapping.get(episode_id, "Unknown")
            category_accuracies[category].append(step_accuracy)

    # 计算每个 category 的 step 级别准确率
    category_accuracy_results = {
        category: sum(accuracies) / len(accuracies)
        for category, accuracies in category_accuracies.items()
    }

    return step_accuracies, category_accuracy_results


def test_main(data_path, category_file):
    """ 读取数据并计算 accuracy """
    data = read_jsonl(data_path)
    category_mapping = read_category_mapping(category_file)

    # 计算 step 级别的总准确率 & category 级别的 step accuracy
    step_accuracies, category_accuracy_results = calculate_step_accuracy(data, category_mapping)

    step_accuracy = sum(step_accuracies) / len(step_accuracies)
    print(f"Step-level accuracy: {step_accuracy * 100:.2f}%")

    print("\nCategory-level step accuracy:")
    print(category_accuracy_results)
    for category, acc in sorted(category_accuracy_results.items()):
        print(f"Category {category}: {acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL data file")
    parser.add_argument("--category_file", type=str, default='/ailab/user/wangwenhao/ms-swift/val_cate_hete.json', help="Path to the category mapping JSON file")

    args = parser.parse_args()

    test_main(args.data_path, args.category_file)
