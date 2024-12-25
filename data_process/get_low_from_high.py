import json
print(0)
from utils.io import load_json, dump_json
print(1)


def complete_sub_ins_gt(data, reference_data):
    """
    遍历原数据并根据参考数据补全 sub_ins_gt 字段，
    同时替换 imgs 中的路径 ms-swift_old 为 ms-swift。
    """
    completed_data = []

    for i, item in enumerate(data):
        # 从参考数据获取 sub_ins_gt
        if i < len(reference_data) and 'sub_ins_gt' in reference_data[i]:
            item['sub_ins_gt'] = reference_data[i]['sub_ins_gt']

        # 替换 imgs 中的路径
        if 'imgs' in item:
            item['imgs'] = [img.replace("ms-swift_old", "ms-swift") for img in item['imgs']]

        completed_data.append(item)

    return completed_data


def main():
    # 文件路径
    original_file_path = '/ailab/user/wangwenhao/ms-swift/output/data_format/alg5_train_5000_v1.json'  # 需要补全的原始数据
    reference_file_path = '/ailab/user/wangwenhao/ms-swift/output/data_format/alg7_train_5000_v1.json'  # 参考数据，用于补全

    # 加载原始数据和参考数据
    original_data = load_json(original_file_path)
    reference_data = load_json(reference_file_path)

    # 补全 sub_ins_gt 并替换 imgs 中的路径
    completed_data = complete_sub_ins_gt(original_data, reference_data)

    dump_json(completed_data, '/ailab/user/wangwenhao/ms-swift/output/data_format/alg5_train_5000_v1.json')


if __name__ == "__main__":
    main()
