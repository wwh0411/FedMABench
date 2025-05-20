print('0')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print('1`')
import os
import json
print('1')
# from transformers import AutoTokenizer, AutoModel
# import torch
print('2')
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import argparse
from tqdm import tqdm
# os.environ['TRANSFORMERS_OFFLINE']="1"


def calculate_tfidf(sentence1, sentence2):
    # 创建一个TF-IDF向量化器
    vectorizer = TfidfVectorizer()

    # 向量化句子
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

    # 计算第一个和第二个文档的余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return similarity[0][0]


# 下载NLTK的punkt数据包
# nltk.download('punkt')

def calculate_bleu(reference, candidate):
    # 1. 计算BLEU分数
    reference_tokens = nltk.word_tokenize(reference.lower())  # 参考句子
    candidate_tokens = nltk.word_tokenize(candidate.lower())  # 生成句子
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method2)  # 计算BLEU分数

    return bleu_score


def calculate_rouge(reference, candidate):
    # 2. 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)  # 计算ROUGE分数
    return rouge_scores['rouge1'].recall


def bert_encode(texts, model, tokenizer):
    # 将文本编码为BERT的输入格式
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # 获取BERT的输出，包括每个token的embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
    # 我们取[CLS]标记的输出作为整个句子的表示
    return model_output.last_hidden_state[:, 0, :].numpy()


def calculate_embed_similarity(sentence1, sentence2):
    # 加载预训练的BERT模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True, device_map='cuda')

    # 编码句子
    # embeddings = bert_encode([sentence1, sentence2], model, tokenizer)
    embeddings_1 = model.encode(sentence1, max_length=128, task="text-matching")
    embeddings_2 = model.encode(sentence2, max_length=128, task="text-matching")

    # 计算余弦相似度
    similarity = cosine_similarity([embeddings_1], [embeddings_2])

    return similarity[0][0]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_ins(query):
    import re
    match = re.search(r'### User Instruction ###\n(.*?)\n\n', query, re.DOTALL)

    # 如果匹配到结果，打印出来
    if match:
        user_instruction = match.group(1).strip()  # 去除首尾空格
        # print(user_instruction)
        return user_instruction
    else:
        print(ins)


def eval_main(path, method, level='high'):
    resu_list = load_json(path)
    sim_list = []
    for entry in tqdm(resu_list):
        if level == 'high':
            ins_pre = entry['ins_pre']
            ins_gt = entry['ins_gt']

            if method == 'tfidf':
                sim = calculate_tfidf(ins_pre, ins_gt)
            elif method == 'embed':
                sim = calculate_embed_similarity(ins_pre, ins_gt)
            elif method == 'bleu':
                sim = calculate_bleu(ins_gt, ins_pre)
            elif method == 'rouge':
                sim = calculate_rouge(ins_gt, ins_pre)
            else:
                raise NotImplementedError
            sim_list.append(sim)
        else:
            sub_ins_gt = entry['sub_ins_gt']
            sub_ins_pre = entry['des_acts']
            if not sub_ins_pre:
                return 0
            sim_sub_list = []
            if not sub_ins_pre:
                sim_list.append(0)
            for ins_pre, ins_gt in zip(sub_ins_pre, sub_ins_gt):
                if method == 'tfidf':
                    sim = calculate_tfidf(ins_pre, ins_gt)
                elif method == 'embed':
                    sim = calculate_embed_similarity(ins_pre, ins_gt)
                elif method == 'bleu':
                    sim = calculate_bleu(ins_gt, ins_pre)
                elif method == 'rouge':
                    sim = calculate_rouge(ins_gt, ins_pre)
                else:
                    raise NotImplementedError
                sim_sub_list.append(sim)
            sim_list.append(sum(sim_sub_list) / len(sim_sub_list))

    print(method, 'avg:', sum(sim_list) / len(sim_list))
    return sim_list


def get_relation_hl_ll(sim_list_hl, sim_list_ll):
    import matplotlib.pyplot as plt

    # 示例数据


    # 创建散点图
    plt.scatter(sim_list_hl, sim_list_ll)

    # 添加标题和标签
    plt.title('散点图示例')
    plt.xlabel('列表 A')
    plt.ylabel('列表 B')

    plt.savefig('scatter_plot.png', dpi=300)


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    # 设置默认目录路径
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='The directory of the data to process. Default: "%(default)s"'
    )
    parser.add_argument('--choice', type=str, default='all')
    args = parser.parse_args()

    if args.choice == 'all':
        for method in ['tfidf', 'rouge', 'bleu']:
            print(method)
            sim_list_hl = eval_main(args.data_path, method, 'high')
            sim_list_ll = eval_main(args.data_path, method, 'low')
            # get_relation_hl_ll(sim_list_hl, sim_list_ll)
    else:
        sim_list_hl = eval_main(args.data_path, args.choice, 'high')
        sim_list_ll = eval_main(args.data_path, args.choice, 'low')
        # get_relation_hl_ll(sim_list_hl, sim_list_ll)