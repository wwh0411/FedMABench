import numpy as np
import cv2
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


def imread_unicode(path):
    # 使用numpy直接读取数据
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return image


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def dump_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def dump_jsonl(data_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.close()


def merge_json(path1, path2, output_path):
    dump_json(load_json(path1) + load_json(path2), output_path)


def load_xml(path, pretty=False):
    with open(path, encoding="utf-8") as f:
        raw_str = f.read()
        xml_root = ET.fromstring(raw_str)
        tree = ET.ElementTree(xml_root)
        if pretty:
            return_str = minidom.parseString(raw_str).toprettyxml(indent="\t")
        else:
            return_str = raw_str
        return tree, return_str
