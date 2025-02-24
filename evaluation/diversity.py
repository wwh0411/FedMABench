import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
# os.environ["TRANSFORMERS_OFFLINE"]='1'
# 使用 Sentence-BERT 模型
models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2", "stsbroberta-base-v2",
                    "distilbert-base-nli-stsb-meantokens"]
model = SentenceTransformer(models[1])
# model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/ailab/user/wangwenhao/FedMobile/evaluation/all-MiniLM-L6-v2')  # 选择一个预训练模型，你也可以选择其他模型
print(model)
def read_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data
# 假设你的指令列表是这样的
path = '/ailab/user/wangwenhao/ms-swift/output/gpt-4o-mini/data_format/alg3_train_1000_v1.json'
json_data = read_json(path)

instructions = [x['ins_pre'] for x in json_data]
# instructions = [x['ins_gt'] for x in json_data]
# 1. 获取每个指令的嵌入（embedding）
embeddings = model.encode(instructions)
print(embeddings.shape)

print('finish 1')
# 2. 计算每对嵌入之间的余弦相似度
cosine_similarities = cosine_similarity(embeddings)
print(cosine_similarities.shape)
# 3. 计算所有余弦相似度的平均值
# 排除对角线上的相似度，因为它们是指令和自身的相似度
num_instructions = len(instructions)
cosine_similarities_no_diag = cosine_similarities[np.triu_indices(num_instructions, k=1)]
print('a:', cosine_similarities_no_diag)
average_cosine_similarity = np.mean(cosine_similarities_no_diag)
print(np.mean(cosine_similarities))
# 输出结果
print(f"Average cosine similarity (excluding diagonal): {average_cosine_similarity:.4f}")
