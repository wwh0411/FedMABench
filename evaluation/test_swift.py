import os
from tqdm import tqdm
import json

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def judge_step(a, b):
    return 1


def calculate_step_accuracy(data):
    """
    计算每一行的准确率，判断每一行的 label 和 response 是否一致。
    返回每行的准确性列表 [1, 0, 1, ...]。
    """
    line_accuracies = []
    for item in data:
        # 如果 label 和 response 相同，认为是相关的
        line_accuracies.append(judge_step(item['label'], item['response']))

    return line_accuracies


def calculate_episode_accuracy(data):
    """
    按照 query 对数据进行分组，然后计算每个 episode 的准确率。
    如果一个 episode 中所有行的 label 和 response 都相同，准确率为 1，否则为 0。
    """
    episode_accuracies = []
    episodes = {}

    # 按照 query 对数据进行分组
    for item in data:
        query = item['query']
        if query not in episodes:
            episodes[query] = []
        episodes[query].append(item)

    # 对每个 episode 检查所有行是否相关
    for query, episode_items in episodes.items():
        # 如果 episode 中所有行的准确性都为 1，则认为该 episode 的准确率为 1
        if all(item['label'] == item['response'] for item in episode_items):
            episode_accuracies.append(1)
        else:
            episode_accuracies.append(0)

    return episode_accuracies


def main():
    # 读取数据
    data_path = r'/ailab/user/wangwenhao/FedMobile/output/qwen2-vl-7b-instruct/v7-20241219-094924/global_lora_49/infer_result/20241220-001128.jsonl'
    data = read_jsonl(data_path)
    
    # 获取数据所在目录和文件名
    output_dir = os.path.dirname(data_path)
    base_filename = os.path.splitext(os.path.basename(data_path))[0]  # 获取不带后缀的文件名
    output_file = os.path.join(output_dir, base_filename + '.log')    # 构建 .log 文件路径

    # 打开文件保存输出

    # 计算每一行的准确率
    line_accuracies = calculate_step_accuracy(data)
    line_accuracy = sum(line_accuracies) / len(line_accuracies)
    print(f"Line-level accuracy: {line_accuracy:.4f}")

    # 计算按 query 分组的 episode 准确率
    episode_accuracies = calculate_episode_accuracy(data)
    episode_accuracy = sum(episode_accuracies) / len(episode_accuracies)
    print(f"Episode-level accuracy: {episode_accuracy:.4f}")


if __name__ == '__main__':
    main()
