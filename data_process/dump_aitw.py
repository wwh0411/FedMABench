import os.path

import tensorflow as tf
import cv2
import pickle as pkl
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
# from utils.decode import decode_image, decode_tree, convert_win_to_xml
# from utils.node_process import extract_nodes, dump_nodes_to_xml, is_point_in_node
# from utils.bbox import calculate_iof
from tqdm import tqdm

def decode_image(byte_string):
    image = tf.io.decode_image(byte_string, channels=None, dtype=tf.dtypes.uint8, name=None, expand_animations=True)
    image_np = image.numpy()
    return image_np

def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
  """Decodes image from example and reshapes.

  Args:
    example: Example which contains encoded image.
    image_height: The height of the raw image.
    image_width: The width of the raw image.
    image_channels: The number of channels in the raw image.

  Returns:
    Decoded and reshaped image tensor.
  """
  image = tf.io.decode_raw(
      example.features.feature['image/encoded'].bytes_list.value[0],
      out_type=tf.uint8,
  )

  height = tf.cast(image_height, tf.int32)
  width = tf.cast(image_width, tf.int32)
  n_channels = tf.cast(image_channels, tf.int32)

  return tf.reshape(image, (height, width, n_channels))


def dump_one_episode_obersavations(screenshots, screenshot_widths, screenshot_heights, forests,
                                   actions, out_ep_dir: Path):
    for step_id, (img, w, h, forest, action) in enumerate(zip(
            screenshots, screenshot_widths, screenshot_heights, forests, actions)):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        action = eval(action)
        all_node_list = extract_nodes(forest.windows, h, w)
        if action['action_type'] in ['click', 'long_press']:  # 检查点击动作是否能够匹配上
            check_list = [is_point_in_node((action['x'], action['y']), node) for node in all_node_list]
            assert sum(check_list) > 0, 'no matching node for action.'
        out_file = out_ep_dir / f'{step_id:02d}.xml'
        dump_nodes_to_xml(all_node_list, out_file, out_file.with_suffix('.lightxml'))
        cv2.imwrite(str(out_file.with_suffix('.png')), img)


def dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir):
    annotations = {
        'goal': goal,
        'actions': actions,
        'sub_goal': step_instructions,
    }
    with open(out_ep_dir / 'task_info.json', 'w') as json_file:
        json.dump(annotations, json_file)


def getTFRecordFormat(files):
    # 加载TFRecord数据
    ds = tf.data.TFRecordDataset(files)
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    for batch_data in ds.take(1):
        for serialized_example in batch_data:
            example_proto = tf.train.Example.FromString(serialized_example.numpy())
            for key, feature in example_proto.features.feature.items():
                ftype = None
                fvalue = None

                if feature.HasField('bytes_list'):
                    ftype = 'bytes_list'
                    fvalue = (feature.bytes_list.value)
                elif feature.HasField('float_list'):
                    ftype = 'float_list'
                    fvalue = (feature.float_list.value)
                elif feature.HasField('int64_list'):
                    ftype = 'int64_list'
                    fvalue = (feature.int64_list.value)

                if ftype:
                    result = '{0} : {1} {2}'.format(key, ftype, fvalue)
                    print(result)


import json
import os
from collections import defaultdict
from pathlib import Path
import tensorflow as tf
import cv2
from tqdm import tqdm


# def _decode_image(example, height, width, channels):
#     # 这里假设有一个方法可以解码图像，可以按需修改
#     img_data = example.features.feature['image/encoded'].bytes_list.value[0]
#     # img = tf.image.decode_jpeg(img_data, channels=channels)
#     # img = tf.image.resize(img, [height, width])
#     img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGBA2BGR)
#     return img
# 动作类型的常量
TYPE = 3
DUAL_POINT = 4
PRESS_BACK = 5
PRESS_HOME = 6
PRESS_ENTER = 7
STATUS_TASK_COMPLETE = 10
STATUS_TASK_IMPOSSIBLE = 11


def format_action_description(action_type, ui_positions, ui_types, ui_texts, text_input=None):
    """
    根据 action_type 和 UI 相关信息生成描述文本。

    :param action_type: 动作类型（例如 DUAL_POINT, TYPE, PRESS_BACK 等）
    :param ui_positions: UI 元素的坐标列表
    :param ui_types: UI 元素的类型列表（如 'button', 'text' 等）
    :param ui_texts: UI 元素的 OCR 文本列表
    :param text_input: 对于 TYPE 类型的动作，输入的文本内容
    :return: 格式化后的动作描述字符串
    """

    if action_type == DUAL_POINT:
        # 对于 DUAL_POINT 类型，获取触摸点的 UI 元素信息
        action_description = "dual point"
        for ui_pos, ui_type, ui_text in zip(ui_positions, ui_types, ui_texts):
            y, x, height, width = ui_pos
            # 如果有 UI 文本，则以文本为准，否则使用 UI 类型
            if ui_text.strip():
                action_description += f" at the {ui_type} with text '{ui_text}'"
            else:
                action_description += f" at the {ui_type}"
        return action_description

    elif action_type == TYPE:
        # 对于 TYPE 类型，返回输入的文本
        if text_input:
            return f"type '{text_input}'"
        return "type ''"  # 如果没有文本输入，则返回空文本

    elif action_type == PRESS_BACK:
        return "press back"

    elif action_type == PRESS_HOME:
        return "press home"

    elif action_type == PRESS_ENTER:
        return "press enter"

    elif action_type == STATUS_TASK_COMPLETE:
        return "check status: task_complete"

    elif action_type == STATUS_TASK_IMPOSSIBLE:
        return "check status: task_impossible"

    else:
        return "unknown action"


def map_touch_to_ui_info(touch_point, ui_positions, ui_types, ui_texts):
    """
    将触摸点映射到 UI 元素，并返回 UI 元素的文本或类型。

    touch_point: dict，包含 y, x 坐标
    ui_positions: list，UI 元素的 bounding box [(y, x, height, width), ...]
    ui_types: list，UI 元素的类型 ["text", "button", "icon", ...]
    ui_texts: list，UI 元素的 OCR 文本 ["text1", "text2", ...]

    返回值：映射的 UI 信息（文本或类型）
    """
    for idx, (ui_pos, ui_type, ui_text) in enumerate(zip(ui_positions, ui_types, ui_texts)):
        y, x, height, width = ui_pos

        # 检查触摸点是否在 UI 元素的范围内
        if y <= touch_point['y'] <= (y + height) and x <= touch_point['x'] <= (x + width):
            # 如果 UI 文本不为空，则返回 UI 文本
            if ui_text and ui_text.strip():  # 判断文本不为空
                # print(type(ui_text))
                return f"Click at the button with text \"{ui_text}\""
            else:
                return f"Click at the button \"{ui_type}\""  # 否则返回 UI 元素的类型

    return 'Click at a button'  # 如果没有匹配到任何 UI 元素，返回 None


def get_action_details(action_type, example):
    """根据 action_type 填充详细信息并映射触摸点和文本到 UI 类型"""
    action_dict = {
        "action_type": action_type
    }

    # 提取 UI 注释相关信息
    if 'image/ui_annotations_positions' in example.features.feature:
        ui_positions = example.features.feature['image/ui_annotations_positions'].float_list.value
        ui_positions = [(ui_positions[i], ui_positions[i + 1], ui_positions[i + 2], ui_positions[i + 3]) for i in
                        range(0, len(ui_positions), 4)]
        # action_dict["ui_positions"] = ui_positions

    if 'image/ui_annotations_text' in example.features.feature:
        ui_texts = example.features.feature['image/ui_annotations_text'].bytes_list.value
        ui_texts = [t.decode('utf-8') for t in ui_texts]
        # ui_texts = convert_bytes_to_str(ui_texts)

    if 'image/ui_annotations_ui_types' in example.features.feature:
        ui_types = example.features.feature['image/ui_annotations_ui_types'].bytes_list.value
        ui_types = [t.decode('utf-8') for t in ui_types]
        # ui_types = convert_bytes_to_str(ui_types)

    # 获取触摸点（对于 DUAL_POINT 类型）
    if action_type == DUAL_POINT:
        touch_yx = example.features.feature['results/yx_touch'].float_list.value
        lift_yx = example.features.feature['results/yx_lift'].float_list.value
        action_dict["touch_point"] = {"y": touch_yx[0], "x": touch_yx[1]} if touch_yx else None
        action_dict["lift_point"] = {"y": lift_yx[0], "x": lift_yx[1]} if lift_yx else None

        # 映射触摸点到 UI 元素文本或类型
        if "touch_point" in action_dict:
            touch_point = action_dict["touch_point"]
            mapped_ui_info = map_touch_to_ui_info(touch_point, ui_positions, ui_types, ui_texts)
            action_dict["mapped_ui_info"] = mapped_ui_info
            # print(mapped_ui_info)

            string_formatted = mapped_ui_info

    elif action_type == TYPE:
        # 如果是文本输入类型，获取输入的文本
        text = example.features.feature['results/type_action'].bytes_list.value[0].decode(
            'utf-8') if 'results/type_action' in example.features.feature else ""
        action_dict["text"] = text
        string_formatted = f"Type text \"{text}\""
    # 如果有按钮按下的动作，补充按钮信息
    elif action_type == PRESS_BACK:
        action_dict["button"] = "back"
        string_formatted = "Navigate back"
    elif action_type == PRESS_HOME:
        action_dict["button"] = "home"
        string_formatted = "Navigate home"

    elif action_type == PRESS_ENTER:
        action_dict["button"] = "enter"
        string_formatted = "Press enter"

    # 状态信息
    elif action_type == STATUS_TASK_COMPLETE:
        action_dict["status"] = "task_complete"
        string_formatted = "Check status: complete"

    elif action_type == STATUS_TASK_IMPOSSIBLE:
        action_dict["status"] = "task_impossible"
        string_formatted = "Check status: impossible"

    # action_dict['string'] = string_formatted
    else:
        string_formatted = ""

    return action_dict, string_formatted


def dump_all_episodes(dataset, out_root_dir: Path, cate=None):
    # 创建一个默认字典来存储每个ep_id的actions列表
    episodes_step_instructions = defaultdict(list)
    total_info = []
    # 遍历数据集
    for i, d in tqdm(enumerate(dataset)):
        # if i :
        #     break
        example = tf.train.Example()
        example.ParseFromString(d)

        # 解析每个数据项
        ep_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        episode_length = example.features.feature['episode_length'].int64_list.value[0]

        step_id = example.features.feature['step_id'].int64_list.value

        goal_info = example.features.feature['goal_info'].bytes_list.value
        action_type = example.features.feature['results/action_type'].int64_list.value[0]

        # 获取图像的尺寸和通道数
        image_height = example.features.feature['image/height'].int64_list.value[0]
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_channels = example.features.feature['image/channels'].int64_list.value[0]

        # 解码图像
        image = _decode_image(example, image_height, image_width, image_channels)
        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGBA2BGR)

        # 设置目标文件夹路径
        try:
            out_ep_dir = out_root_dir / f'{int(ep_id):06d}'
        except:
            out_ep_dir = out_root_dir / ep_id[:50]
        out_ep_dir.mkdir(exist_ok=True, parents=True)



        # 提取action并按顺序添加到列表
        actions = []
        actions_converted = []
        action_details, string_formatted = get_action_details(action_type, example)
        actions.append(action_details)
        actions_converted.append(string_formatted)

        episodes_step_instructions[ep_id].extend(actions)

        # 如果已经是该 ep_id 的最后一帧或者最后一个步骤（根据数据判断）
        # if len(episodes_step_instructions[ep_id]) == episode_length:
        # 获取目标信息（goal_info）
        goal_info_str = goal_info[0].decode('utf-8') if goal_info else ""
        sub_goal = []  # 假设这是你希望提取的子目标，可以根据实际数据填充

        # 格式化 JSON 数据
        episode_data = {
            "goal": goal_info_str,
            "actions": episodes_step_instructions[ep_id],
            # "actions_converted": actions_converted,
            "sub_goal": sub_goal
        }

        # 将每个 ep_id 的数据存储为 JSON 文件
        json_filename = out_ep_dir / "episode_data.json"
        with open(json_filename, 'w') as json_file:
            json.dump(episode_data, json_file)

            # # 清空该 ep_id 的 actions 列表，以便下一个 ep_id 使用
            # episodes_step_instructions[ep_id] = []
        # 存储图像
        img_filename = out_ep_dir / f'{step_id[0]:02d}.png'
        cv2.imwrite(str(img_filename), img)

        total_info_d = {"episode_id": ep_id,
                    "instruction": goal_info_str,
                    "sub_instruction": sub_goal,
                    "act_origin": actions,
                    "act_convert": actions_converted,
                    "img": str(img_filename)}
        total_info.append(total_info_d)

    from utils.io import dump_jsonl
    out_path = out_root_dir.parent / f"step-wise-{cate}.jsonl"
    print(f'Save to {out_path}.')
    dump_jsonl(total_info, out_path)

    episode_dict = defaultdict(
        lambda: {"instruction": "", "sub_instructions": [], "acts_origin": [], "acts_convert": [], "imgs": []})

    for item in total_info:
        episode = episode_dict[item["episode_id"]]
        episode["instruction"] = item["instruction"]
        episode["sub_instructions"].extend(item["sub_instruction"])
        episode["acts_origin"].extend(item["act_origin"])
        episode["acts_convert"].extend(item["act_convert"])
        episode["imgs"].append(item["img"])

    # 输出合并后的数据
    result = [
        {"episode_id": episode_id, "instruction": values["instruction"], "sub_instructions": values["sub_instructions"],
         "acts_origin": values["acts_origin"], "acts_convert": values["acts_convert"], "imgs": values["imgs"]}
        for episode_id, values in episode_dict.items()
    ]

    out_path = out_root_dir.parent / f"episode-wise-{cate}.jsonl"
    print(f'Save to {out_path}.')
    print("episode_length", len(result))
    dump_jsonl(result, out_path)

# def dump_all_episodes(dataset, out_root_dir: Path):
#     episodes_step_instructions = defaultdict(list)
#     for d in tqdm(dataset):
#         example = tf.train.Example()
#         example.ParseFromString(d)
#
#         ep_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
#         episode_length = example.features.feature['episode_length'].int64_list.value[0]
#         print(episode_length)
#         step_id = example.features.feature['step_id'].int64_list.value
#         print(step_id)
#         goal_info = example.features.feature['goal_info'].bytes_list.value#[0].decode('utf-8')
#         print(goal_info)
#         action_type = example.features.feature['results/action_type'].int64_list.value
#
#         print(goal_info)
#         print(ep_id)
#
#
#
#         image_height = example.features.feature['image/height'].int64_list.value[0]
#         image_width = example.features.feature['image/width'].int64_list.value[0]
#         image_channels = example.features.feature['image/channels'].int64_list.value[0]
#         image = _decode_image(example, image_height, image_width, image_channels)
#         print(type(image))
#         img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGBA2BGR)
#         cv2.imwrite('test.png', img)
#
#
#         out_ep_dir = out_root_dir / f'{ep_id:06d}'
#         out_ep_dir.mkdir(exist_ok=True, parents=True)
#         if (out_ep_dir / f'{len(step_instructions):02d}.png').exists():
#             continue
#         # goal = ep.features.feature['goal'].bytes_list.value[0].decode('utf-8')
#         # print(goal)
#         # screenshots = [decode_image(x) for x in ep.features.feature['screenshots'].bytes_list.value]  # N
#         # screenshot_widths = [x for x in ep.features.feature['screenshot_widths'].int64_list.value]  # N
#         # screenshot_heights = [x for x in ep.features.feature['screenshot_heights'].int64_list.value]  # N
#         # actions = [x.decode('utf-8') for x in ep.features.feature['actions'].bytes_list.value]  # N - 1
#
#
#         assert ep_id not in episodes_step_instructions, f'{ep_id} has been processed'
#         episodes_step_instructions[ep_id].append(step_instructions)
#         actions.append("{\"action_type\":\"status\",\"goal_status\":\"successful\"}")
#         try:
#             dump_one_episode_obersavations(screenshots, screenshot_widths, screenshot_heights, forests,
#                                            actions, out_ep_dir)
#             dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir)
#
#         except Exception as e:
#             error_message = str(e)
#             import traceback
#             err_str = traceback.format_exc()  # 获取错误信息的字符串表示
#             print(err_str)
#             error_log_file = out_root_dir / 'error_log.txt'  # 设置错误日志文件的名称
#             with open(error_log_file, 'a') as file:  # 打开文件以追加模式写入
#                 file.write(f'[{ep_id}]: {error_message}' + "\n")
#
#         if len(episodes_step_instructions) % 20 == 0:
#             print(f'Read {len(episodes_step_instructions)} episodes.')
#
#         if ep_id == 11:
#             exit()
#         exit()
#     return episodes_step_instructions

cate = 'single'
filenames = tf.io.gfile.glob(f'/ailab/user/wangwenhao/ms-swift/aitw/{cate}/*')
# getTFRecordFormat(filenames)
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()




ep_dict = dump_all_episodes(raw_dataset, Path(f'/ailab/user/wangwenhao/ms-swift/unpack-aitw-{cate}'), cate)



