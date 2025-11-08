import os.path

import tensorflow as tf
import cv2
import pickle as pkl
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from utils.decode import decode_image, decode_tree, convert_win_to_xml
from utils.node_process import extract_nodes, dump_nodes_to_xml, is_point_in_node
from utils.bbox import calculate_iof
from tqdm import tqdm


def dump_one_episode_obersavation(screenshots, screenshot_widths, screenshot_heights, forests,
                                  actions, out_ep_dir: Path):
    for step_id, (screenshot, w, h, forest, action) in enumerate(zip(
            screenshots, screenshot_widths, screenshot_heights, forests, actions)):
        out_file = out_ep_dir / f'{step_id:02d}.xml'
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(out_file.with_suffix('.png')), screenshot)
        action = eval(action)
        all_node_list = extract_nodes(forest.windows, h, w)
        # if action['action_type'] in ['click', 'long_press']:  # 检查点击动作是否能够匹配上
        #     check_list = [is_point_in_node((action['x'], action['y']), node) for node in all_node_list]
        #     assert sum(check_list) > 0, 'no matching node for action.'

        try:
            dump_nodes_to_xml(all_node_list, out_file, out_file.with_suffix('.lightxml'))
        except:
            print('incorrect xml parsing', str(out_ep_dir))


def dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir):
    annotations = {
        'goal': goal,
        'actions': actions,
        'sub_goal': step_instructions,
    }
    with open(out_ep_dir / 'task_info.json', 'w') as json_file:
        json.dump(annotations, json_file)


def dump_all_episodes(dataset, out_root_dir: Path):
    episodes_step_instructions = defaultdict(list)
    for d in tqdm(dataset):
        ep = tf.train.Example()
        ep.ParseFromString(d)

        ep_id = ep.features.feature['episode_id'].int64_list.value[0]
        step_instructions = [x.decode('utf-8') for x in
                             ep.features.feature['step_instructions'].bytes_list.value]  # N - 1
        out_ep_dir = out_root_dir / f'{ep_id:06d}'
        out_ep_dir.mkdir(exist_ok=True, parents=True)
        if (out_ep_dir / f'{len(step_instructions):02d}.png').exists():
            continue
        goal = ep.features.feature['goal'].bytes_list.value[0].decode('utf-8')
        screenshots = [decode_image(x) for x in ep.features.feature['screenshots'].bytes_list.value]  # N
        screenshot_widths = [x for x in ep.features.feature['screenshot_widths'].int64_list.value]  # N
        screenshot_heights = [x for x in ep.features.feature['screenshot_heights'].int64_list.value]  # N
        actions = [x.decode('utf-8') for x in ep.features.feature['actions'].bytes_list.value]  # N - 1
        forests = [decode_tree(x) for x in ep.features.feature['accessibility_trees'].bytes_list.value]  # N

        assert ep_id not in episodes_step_instructions, f'{ep_id} has been processed'
        episodes_step_instructions[ep_id].append(step_instructions)
        actions.append("{\"action_type\":\"status\",\"goal_status\":\"successful\"}")
        try:
            dump_one_episode_obersavation(screenshots, screenshot_widths, screenshot_heights, forests,
                                           actions, out_ep_dir)
            dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir)

        except Exception as e:
            error_message = str(e)
            import traceback
            err_str = traceback.format_exc()  # 获取错误信息的字符串表示
            print(err_str)
            error_log_file = out_root_dir / 'error_log.txt'  # 设置错误日志文件的名称
            with open(error_log_file, 'a') as file:  # 打开文件以追加模式写入
                file.write(f'[{ep_id}]: {error_message}' + "\n")

        if len(episodes_step_instructions) % 200 == 0:
            print(f'Read {len(episodes_step_instructions)} episodes.')
    return episodes_step_instructions


tfrecord_files = tf.io.gfile.glob(r'/GPFS/data/zijieyu/ac_data/*')
tfrecord_files = sorted(tfrecord_files)[:20]
# getTFRecordFormat(filenames)
raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP').as_numpy_iterator()

ep_dict = dump_all_episodes(raw_dataset, Path(r'/GPFS/data/wenhaowang-1/FedMobile/androidcontrol_1108/android_control_unpack_all'))