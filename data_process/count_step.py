from utils.io import dump_json, load_json, dump_jsonl, load_jsonl
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2-7b")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--choice", type=int, default=0)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    path = '/ailab/user/wangwenhao/ms-swift/episode-wise-single.jsonl'
    data = load_jsonl(path)
    print(len(data))
    data = data[:1000]
    name = 'Single'

    dump_jsonl(data, f'Basic-AitW {name}.jsonl')