import os
from tqdm import tqdm
import json

import argparse



def test_main(args):
    # 读取数据
    data_path = args.data_path
    data = read_jsonl(data_path)

    # 获取数据所在目录和文件名
    output_dir = os.path.dirname(data_path)
    base_filename = os.path.splitext(os.path.basename(data_path))[0]  # 获取不带后缀的文件名
    output_file = os.path.join(output_dir, base_filename + '.log')    # 构建 .log 文件路径

    # 打开文件保存输出
    with open(output_file, 'w') as f:
        num_correct = 0
        num_total = 0
        for item in tqdm(data):
            # 将输出保存到文件中
            f.write(f'[{num_correct} / {num_total}] / {num_correct / (num_total + 1e-6):.2f}\n')

            if calculate_similarity(item['label'], item['response']) > 0.6:
                num_correct += 1
            # else:
            f.write('P:\t' + item['response'] + '\n')
            f.write('G:\t' + item['label'] + '\n')
            f.write(str(calculate_similarity(item['label'], item['response'])))
            # if item['label'].lower() != item['response'].lower():
            #     f.write('P:\t' + item['response'] + '\n')
            #     f.write('G:\t' + item['label'] + '\n')
            # else:
            #     num_correct += 1
            num_total += 1
    return num_correct / (num_total + 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    # parser.add_argument("--save_failed_generation", action='store_true', default=False)

    args = parser.parse_args()

    score = test_main(args)
    print('score:', score)
