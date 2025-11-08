import os
import re
import json
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
from utils.io import load_json, load_xml, dump_jsonl


def read_xml_files(episode_id):
    """
    Read all XML files for a given episode and return their content.
    """
    episode_path = f'data/light_xml/{episode_id}/'
    xml_contents = {}

    # Ensure files are sorted numerically by their filename (assuming filenames are like 0.xml, 1.xml, etc.)
    xml_files = sorted(os.listdir(episode_path), key=lambda x: int(os.path.splitext(x)[0]))

    for filename in xml_files:
        if filename.endswith('.xml'):
            file_path = os.path.join(episode_path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            xml_str = ET.tostring(root, encoding='unicode', method='xml')
            dom = xml.dom.minidom.parseString(xml_str)
            pretty_xml_str = dom.toprettyxml()
            step_id = os.path.splitext(filename)[0]
            xml_contents[step_id] = pretty_xml_str

    return xml_contents


def convert_action_coordinates_to_index(action_str, bounds_info):
    """
    Convert actions with x, y coordinates to actions with index based on bounds information.
    """

    action = json.loads(action_str)  # Parse the string to a dictionary
    if action["action_type"] in ["click", "long_press"]:
        x, y = action["x"], action["y"]
        index = find_index_for_coordinates(x, y, bounds_info)
        converted_action = {"action_type": action["action_type"], "index": index}
    else:
        converted_action = action
    return converted_action


def find_index_for_coordinates(x, y, bounds_info):
    """
    Find the index of the UI element whose bounds contain the point (x, y).
    """
    min_area = np.inf
    min_area_index = -1
    for index, info in bounds_info.items():
        bounds = info['bounds']
        if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
            if info['area'] < min_area:
                min_area = info['area']
                min_area_index = index
    # assert min_area_index != -1, f'min_area_index not found.'
    return min_area_index


def decode_bounds_from_tree(tree):
    id2bounds = {}
    for node in tree.iter():
        if node.tag == 'Root':
            continue
        b = node.attrib['boundsInScreen']
        match = re.match(r"Rect\((-?\d+),\s*(-?\d+)\s*-\s*(-?\d+),\s*(-?\d+)\)", b)
        if match:
            box = match.groups()  # [x1, y1, x2, y2]
            box = list(map(int, box))
        else:
            box = []
        id2bounds[node.attrib['id']] = box
    return id2bounds


def extract_bounds_in_tree(tree, id2bounds):
    ret = {}
    for node in tree.iter():
        if node.tag == 'Root':
            continue
        node_id = node.attrib['id']
        x1, y1, x2, y2 = id2bounds[node_id]
        ret[node_id] = {'bounds': [x1, y1, x2, y2],
                        'area': (x2 - x1) * (y2 - y1),
                        'desc': str(node.attrib)
                        }  # TODO: 优化节点描述, 用于 history
    return ret


def euclidean_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)


def get_scroll_direction(touch, lift):
    x_diff = lift['x'] - touch['x']
    y_diff = lift['y'] - touch['y']
    if abs(y_diff) > abs(x_diff):
        return "up" if y_diff < 0 else "down"
    else:
        return "left" if x_diff < 0 else "right"


def simplify_actions(action, image_path, threshold=0.04):
    """
    分类并简化 action，只保留 click 的中心坐标，scroll 的方向
    :return: list of simplified dict
    """
    with Image.open(image_path) as img:
        width, height = img.size

    touch = action["touch_point"]
    lift = action["lift_point"]
    dist = euclidean_distance(touch, lift)

    if dist <= threshold:
        avg_x = (touch['x'] + lift['x']) / 2
        avg_y = (touch['y'] + lift['y']) / 2
        abs_x = int(avg_x * width)
        abs_y = int(avg_y * height)
        return {
            "action_type": "click",
            "x": abs_x,
            "y": abs_y
        }
    else:
        direction = get_scroll_direction(touch, lift)
        return {
            "action_type": "scroll",
            "direction": direction
        }


def reformat_action_aitw(action, image_path):
    new_action = dict()
    if action["action_type"] == 3:
        new_action["action_type"] = 'type'
        new_action["input_text"] = action["text"]
    elif action["action_type"] == 4: # click
        new_action = simplify_actions(action, image_path)
    elif action["action_type"] == 5:
        new_action["action_type"] = 'navigate_back'
    elif action["action_type"] == 6:
        new_action["action_type"] = 'navigate_home'
    elif action["action_type"] == 7:
        new_action["action_type"] = 'press_enter'
    # elif action["action_type"] in [10, 11]:
    #     new_action["action_type"] = 'status'
    #     new_action["goal_status"] = 'successful' if action["action_type"] == 10 else 'impossible'
    elif action["action_type"] == 10:
        new_action["action_type"] = 'complete'
    elif action["action_type"] == 11:
        new_action["action_type"] = 'impossible'
    else:
        raise NotImplementedError('Not included action type')
    return new_action


def describe_action(action, bounds_info):
    if action["action_type"] in ["click", "long_press"]:
        act = action['action_type'].capitalize()
        index = action['index']
        if index != -1:
            node_desc = eval(bounds_info[index]['desc'])

            if 'text' in node_desc.keys():
                desc = f"{act} on button with the text \"{node_desc['text']}\""
            elif 'tooltip_text' in node_desc.keys():
                desc = f"{act} on button with the text \"{node_desc['tooltip_text']}\""
            elif 'content-desc' in node_desc.keys():
                desc = f"{act} on button with the function of \"{node_desc['content-desc']}\""
            else:
                desc = f"{act} on button"
        else:
            desc = f"{act} on button"

    elif action['action_type'] in ['open_app', 'wait', 'done', 'navigate_back', 'input_text', 'navigate_home', 'press_enter']:
        if action['action_type'] == 'open_app':
            desc = f"Open App: {action['app_name']}"
        elif action['action_type'] == 'input_text':
            desc = f"Type text: {action['text']}"
        elif action['action_type'] == 'wait':
            desc = "Wait for response"
        elif action['action_type'] == 'navigate_back':
            desc = "Go back to the previous page"
        elif action['action_type'] == 'navigate_home':
            desc = "Return to the home page"
        else:
            desc = action['action_type'].capitalize()
    elif action['action_type'] == 'scroll':
        desc = f"Scroll {action['direction']}"
    elif action['action_type'] == 'status':
        desc = f"Check status: {action['goal_status']}"
    else:
        raise ValueError(f"action type {action['action_type']}")
    return desc


def process_episodes(data_dir: Path, episode_id_list=None):
    """
    Process specified episodes or all episodes if no specific episode is provided, and save their compiled messages.
    """
    all_messages_step = []
    all_messages_episode = []
    episode_paths = [x.parent for x in data_dir.rglob('episode_data.json')]

    if episode_id_list is not None:
        episode_paths = [x for x in episode_paths if int(x.stem) in episode_id_list]
    episode_paths = sorted(episode_paths)
    print('len', len(episode_paths))

    pattern = re.compile(r"^\d+$")

    for episode_path in tqdm(episode_paths):
        episode_id = str(episode_path.relative_to(data_dir))
        task_info = load_json(episode_path / 'episode_data.json')
        actions = task_info['actions']
        goal = task_info['goal']
        # sub_goals = task_info['sub_goal']
        # sub_goals.append('Check if the task is finished')
        screenshots = [i for i in episode_path.glob("*.png") if pattern.match(i.stem)]
        screenshots = sorted(screenshots, key=lambda x: int(x.stem))

        new_actions = []
        actions_convert = []

        # Ensure both actions and xml_contents have the same length
        assert len(screenshots) == len(actions), (f"The number of screenshots ({len(screenshots)}) does not match "
                                           f"the number of actions ({len(actions)}) for episode {episode_path.stem}.")
        for step_index, (screenshot, action) in enumerate(zip(screenshots, actions)):
            # Load bounds information for the current step
            # Convert action coordinates to index using bounds info
            action = reformat_action_aitw(action, str(screenshot))
            new_actions.append(action)
            # action_json = json.loads(action)  # Parse the string to a dictionary

            # Generate the action prompt with converted actions


            # message = make_action_prompt(goal, light_str, action_history, true_value=converted_action)
            # actions_convert.append(action_convert)


            # image_rela_path = 'androidcontrol_1108/' + str(png.relative_to(data_dir.parent))

            # Create the new format

            new_format_message = {
                "episode_id": episode_id,
                "instruction": goal,
                "act_origin": action,
                # "images": [str(relative_som_image_path)],
                "img": str(screenshot),
            }

            all_messages_step.append(new_format_message)


        all_messages_episode.append({"episode_id": episode_id,
                                     "instruction": goal,
                                     "acts_origin": new_actions,
                                     "imgs": [str(x) for x in screenshots],
                                     })

    out_path = data_dir.parent / "step-wise-general.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    print(f'Save to {out_path}.')
    dump_jsonl(all_messages_step, out_path)
    out_path = data_dir.parent / "episode-wise-general.jsonl"
    print(f'Save to {out_path}.', len(all_messages_episode))
    dump_jsonl(all_messages_episode, out_path)


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process some episodes.")
    # 设置默认目录路径
    default_data_dir = "/GPFS/data/wenhaowang-1/ms-swift/aitw_unpack/unpack-aitw-general"
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=default_data_dir, 
        help='The directory of the data to process. Default: "%(default)s"'
    )
    args = parser.parse_args()

    # 使用命令行参数
    data_dir = Path(args.data_dir)
    # split = load_json('/remote-home/iot_liuguangyi/data/AndroidControl/splits.json')

    process_episodes(data_dir, None)

if __name__ == '__main__':
    main()
