from transformers import AutoTokenizer
from PIL import Image
import math
from template import *
import json
from utils.io import load_jsonl
from collections import defaultdict
from tqdm import tqdm


# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/ailab/user/wangwenhao/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct")
path = '/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/unpack-androidcontrol-vlm-train_wsub-8652.jsonl'

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def calculate_image_tokens(image_path):
    # 打开图片并获取其尺寸
    with Image.open(image_path) as img:
        width, height = img.size

    # 计算最接近的 28 的倍数的尺寸
    adjusted_width = math.ceil(width / 28) * 28
    adjusted_height = math.ceil(height / 28) * 28
    # 计算 token 数量
    token = (adjusted_width // 28) * (adjusted_height // 28)
    token = min(768, token)

    return token


def calculate_text_tokens(text):
    # 使用 tokenizer 计算 token 数
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)



def get_text_query(acts, imgs, method=0):
    # stage 1
    des_list = []
    des_pic_list = []
    history_list = []

    tokens_all = 0
    image_tokens = sum(calculate_image_tokens(img) for img in imgs)
    # sceenshots concatenation if needed
    if method in [1, 3, 5, 7]:
        prompt_type = 'with_image'
        tokens_all += min(image_tokens, 768)
    else:
        prompt_type = 'without_image'

    # action description if needed
    if method in [4, 5, 6, 7]:
        tokens_all += image_tokens
        for act in acts:
            # print(act)
            query = template_describe_action.format(act=act)
            tokens_all += calculate_text_tokens(query)



    # picture descriptio if needed
    # if method in [6, 7, 8, 9]:
    #     for index, (pic, act) in enumerate(zip(pics, acts)):
    #         query = template_describe_picture
    #         description = infer_stream(engine, InferRequest(messages=[get_message(query, pic, mm_type)]))
    #         des_pic_list.append(description)
    #
    # else:
    #     des_pic_list = None
    # print(des_pic_list)
    # add des into final
    # if method in [6, 7, 8]:
    #     for index, (pic_des, act_des) in enumerate(zip(des_pic_list, des_list)):
    #         history_list.append(template_6_sub.format(i=index + 1, pic_des=pic_des, act_des=act_des))

    # get final query
    if method == 1:
        query = template_1_wo_action
    elif method in [2, 3, 4, 5]:
        query = template_final[prompt_type].format(history='\n'.join(acts))
    # elif method in [4, 5]:
    #     query = template_final[prompt_type].format(history='\n'.join(des_list))
    # elif method in [6, 7]:
    #     query = template_final_w_des[prompt_type].format(history='\n'.join(history_list))

    tokens_all += calculate_text_tokens(query)

    return tokens_all


def main():
    json_data = load_jsonl(path)
    instructions_dict = defaultdict(lambda: {"acts": [], "images": [], "sub_instructions": []})

    # 遍历所有数据并按 instruction 进行分类
    for entry in json_data:
        instruction = entry["instruction"]
        response = entry["response"]
        image_abs = entry["image_abs"]
        sub_instruction = entry["sub_instruction"]

        instructions_dict[instruction]["acts"].append(response)
        instructions_dict[instruction]["images"].append(image_abs)
        instructions_dict[instruction]["sub_instructions"].append(sub_instruction)

    instructions = list(instructions_dict.keys())
    tokens_list = []
    for instruction in tqdm(instructions[:5000]):
        # image_tokens = sum(calculate_image_tokens(img) for img in instructions_dict[instruction]["images"])
        imgs = instructions_dict[instruction]["images"]
        acts = instructions_dict[instruction]["acts"]
        tokens = get_text_query(acts, imgs, method=3)
        tokens_list.append(tokens)


    print(sum(tokens_list))
    print(sum(tokens_list) / len(tokens_list))
main()
