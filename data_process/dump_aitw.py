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



def dump_all_episodes(dataset, out_root_dir: Path):
    episodes_step_instructions = defaultdict(list)
    for d in tqdm(dataset):
        example = tf.train.Example()
        example.ParseFromString(d)

        ep_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        episode_length = example.features.feature['episode_length'].int64_list.value[0]
        print(episode_length)
        step_id = example.features.feature['step_id'].int64_list.value
        print(step_id)
        goal_info = example.features.feature['goal_info'].bytes_list.value#[0].decode('utf-8')
        print(goal_info)
        goal_info = example.features.feature['results/action_type'].int64_list.value

        print(goal_info)
        print(ep_id)



        image_height = example.features.feature['image/height'].int64_list.value[0]
        image_width = example.features.feature['image/width'].int64_list.value[0]
        image_channels = example.features.feature['image/channels'].int64_list.value[0]
        image = _decode_image(example, image_height, image_width, image_channels)
        print(type(image))
        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGBA2BGR)
        cv2.imwrite('test.png', img)


        out_ep_dir = out_root_dir / f'{ep_id:06d}'
        out_ep_dir.mkdir(exist_ok=True, parents=True)
        if (out_ep_dir / f'{len(step_instructions):02d}.png').exists():
            continue
        # goal = ep.features.feature['goal'].bytes_list.value[0].decode('utf-8')
        # print(goal)
        screenshots = [decode_image(x) for x in ep.features.feature['screenshots'].bytes_list.value]  # N
        screenshot_widths = [x for x in ep.features.feature['screenshot_widths'].int64_list.value]  # N
        screenshot_heights = [x for x in ep.features.feature['screenshot_heights'].int64_list.value]  # N
        actions = [x.decode('utf-8') for x in ep.features.feature['actions'].bytes_list.value]  # N - 1
        forests = [decode_tree(x) for x in ep.features.feature['accessibility_trees'].bytes_list.value]  # N

        assert ep_id not in episodes_step_instructions, f'{ep_id} has been processed'
        episodes_step_instructions[ep_id].append(step_instructions)
        actions.append("{\"action_type\":\"status\",\"goal_status\":\"successful\"}")
        try:
            dump_one_episode_obersavations(screenshots, screenshot_widths, screenshot_heights, forests,
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

        if len(episodes_step_instructions) % 20 == 0:
            print(f'Read {len(episodes_step_instructions)} episodes.')

        if ep_id == 11:
            exit()
        exit()
    return episodes_step_instructions


filenames = tf.io.gfile.glob(r'/ailab/user/wangwenhao/ms-swift_old/aitw/general/*')
# getTFRecordFormat(filenames)
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()




ep_dict = dump_all_episodes(raw_dataset, Path(r'./unpack-aitw-general'))



# for key, feature in ep.features.feature.items():
#         #     ftype = None
#         #     fvalue = None
#         #
#         #     if feature.HasField('bytes_list'):
#         #         ftype = 'bytes_list'
#         #         fvalue = (feature.bytes_list.value)
#         #     elif feature.HasField('float_list'):
#         #         ftype = 'float_list'
#         #         fvalue = (feature.float_list.value)
#         #     elif feature.HasField('int64_list'):
#         #         ftype = 'int64_list'
#         #         fvalue = (feature.int64_list.value)
#         #
#         #     if ftype:
#         #         result = '{0} : {1}'.format(key, ftype)
#         #         print(result)