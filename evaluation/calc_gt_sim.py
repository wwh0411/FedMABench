import json
import re
import os
from eval_gpt import calculate_tfidf



# 从文件中读取 JSON 数据并计算相似度
def process_json_file(file_path):
    """
    处理给定的 JSON 文件，计算所有样本的相似度平均值
    """
    # 确保文件存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在!")
        return

    # 打开并读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 存储所有相似度得分
    similarity_scores = []

    # 遍历每个样本，提取 Subordinate Instruction 和 Response，并计算相似度
    for sample in data:
        query_text = sample.get("query", "")
        response = sample.get("response", "")

        # 提取 Subordinate Instruction
        subordinate_instruction_match = re.search(
            r"### Subordinate Instruction ###\n(.*?)\n### Response Requirements ###", query_text, re.DOTALL)
        if subordinate_instruction_match:
            subordinate_instruction = subordinate_instruction_match.group(1).strip()
        else:
            subordinate_instruction = None

        # 如果 Subordinate Instruction 和 Response 都存在，计算相似度
        if subordinate_instruction and response:
            similarity_score = calculate_tfidf(subordinate_instruction, response)
            similarity_scores.append(similarity_score)

    # 如果没有计算到相似度，返回提示信息
    if not similarity_scores:
        print("没有找到有效的 Subordinate Instruction 和 Response.")
        return

    # 计算并输出平均相似度
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"所有样本的平均相似度: {avg_similarity:.2f}")


# 示例调用
file_path = "/ailab/user/wangwenhao/ms-swift/output/gt_val_200_v1_low.json"  # 替换为你实际的 JSON 文件路径
process_json_file(file_path)
