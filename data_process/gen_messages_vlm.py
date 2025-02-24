import os
import re
import json
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np
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


def load_bounds_info(episode_id, step_id):
    """
    Load the index and bounds information for a given episode and step.
    """
    bounds_info_path = f'data/light_xml/{episode_id}/{step_id}.json'
    with open(bounds_info_path, 'r', encoding='utf-8') as file:
        bounds_info = json.load(file)
    return bounds_info


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
    assert min_area_index != -1, f'min_area_index not found.'
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


def describe_action(action, bounds_info):
    if action["action_type"] in ["click", "long_press"]:
        index = action['index']
        node_desc = eval(bounds_info[index]['desc'])
        act = action['action_type'].capitalize()
        if 'text' in node_desc.keys():
            desc = f"{act} on button with the text \"{node_desc['text']}\""
        elif 'tooltip_text' in node_desc.keys():
            desc = f"{act} on button with the text \"{node_desc['tooltip_text']}\""
        elif 'content-desc' in node_desc.keys():
            desc = f"{act} on button with the function of \"{node_desc['content-desc']}\""
        else:
            desc = f"{act} on button"

    elif action['action_type'] in ['open_app', 'wait', 'done', 'navigate_back', 'input_text', 'navigate_home']:
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


def save_message_to_file(episode_id, step_id, message):
    """
    Save the compiled message to a local JSON file.
    """
    output_path = f'data/prompt/{episode_id}/{step_id}_message.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(message, file, ensure_ascii=False, indent=4)


def process_episodes(data_dir: Path, episode_id_list=None, choice=None):
    """
    Process specified episodes or all episodes if no specific episode is provided, and save their compiled messages.
    """
    # If episode_id is specified, process only that episode
    all_messages_step = []
    all_messages_episode = []
    episode_paths = [x.parent for x in data_dir.rglob('task_info.json')]

    if episode_id_list is not None:
        episode_paths = [x for x in episode_paths if int(x.stem) in episode_id_list]
    episode_paths = sorted(episode_paths)
    print('len', len(episode_paths))

    pattern = re.compile(r"^\d+$")

    for episode_path in tqdm(episode_paths):
        episode_id = str(episode_path.relative_to(data_dir))
        task_info = load_json(episode_path / 'task_info.json')
        actions = task_info['actions']
        goal = task_info['goal']
        sub_goals = task_info['sub_goal']
        sub_goals.append('Check if the task is finished')
        xmls = [i for i in episode_path.glob("*.xml") if pattern.match(i.stem)]
        xmls = sorted(xmls, key=lambda x: int(x.stem))
        pngs = [x.with_suffix('.png') for x in xmls]
        lightxmls = [x.with_suffix('.lightxml') for x in xmls]

        action_history = []

        # Ensure both actions and xml_contents have the same length
        assert len(xmls) == len(actions), (f"The number of XML steps ({len(xmls)}) does not match "
                                           f"the number of actions ({len(actions)}) for episode {episode_path.stem}.")
        for step_index, (xml, lightxml, png, action, sub_goal) in enumerate(zip(xmls, lightxmls, pngs, actions, sub_goals)):
            # Load bounds information for the current step
            raw_tree, raw_str = load_xml(xml, pretty=True)
            light_tree, light_str = load_xml(lightxml, pretty=True)

            index2bounds = decode_bounds_from_tree(raw_tree)
            bounds_info = extract_bounds_in_tree(light_tree, index2bounds)

            # Convert action coordinates to index using bounds info
            converted_action = convert_action_coordinates_to_index(action, bounds_info)
            # Generate the action prompt with converted actions
            action_desc = describe_action(converted_action, bounds_info)

            # message = make_action_prompt(goal, light_str, action_history, true_value=converted_action)
            # message = make_action_prompt(goal, None, None, true_value=action_desc)

            action_history.append(action_desc)

            image_abs_path = str(png)
            image_rela_path = 'androidcontrol_1108/' + str(png.relative_to(data_dir.parent))

            # Create the new format
            if choice == 'gen':
                new_format_message = {
                    "episode_id": episode_id,
                    "instruction": goal,
                    "sub_instruction": sub_goal,
                    "act_origin": action,
                    "act_convert": action_desc,
                    # "images": [str(relative_som_image_path)],
                    "img": image_abs_path,
                }
            else:
                new_format_message = {
                    "query": f"<image>{template}",  # Assuming the first part of message is the user content
                    "response": action_desc,
                    "images": [image_rela_path]
                }

            all_messages_step.append(new_format_message)


        all_messages_episode.append({"episode_id": episode_id, "instruction": goal,
                                     "sub_instructions": sub_goals,
                                     "acts_origin": actions,
                                     "acts_convert": action_history,
                                     "imgs": [str(png) for png in pngs],})

    out_path = data_dir.parent / "step-wise-all.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    print(f'Save to {out_path}.')
    dump_jsonl(all_messages_step, out_path)
    out_path = data_dir.parent / "episode-wise-all.jsonl"
    print(f'Save to {out_path}.')
    dump_jsonl(all_messages_episode, out_path)


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process some episodes.")
    # 设置默认目录路径
    default_data_dir = "/ailab/user/wangwenhao/ms-swift/androidcontrol_1108/unpack-androidcontrol"
    default_data_dir = "/ailab/user/wangwenhao/ms-swift_old/androidcontrol_1108/unpack-androidcontrol"
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=default_data_dir, 
        help='The directory of the data to process. Default: "%(default)s"'
    )
    parser.add_argument('--choice', type=str, default='gen')
    args = parser.parse_args()

    # 使用命令行参数
    data_dir = Path(args.data_dir)
    # split = load_json('/remote-home/iot_liuguangyi/data/AndroidControl/splits.json')

    process_episodes(data_dir, [i for i in range(30000)][5000:8000], args.choice)

if __name__ == '__main__':
    main()
