import re
import tensorflow as tf
from android_env.proto.a11y import android_accessibility_forest_pb2
import xml.etree.ElementTree as ET


def decode_image(byte_string):
    image = tf.io.decode_image(byte_string, channels=None, dtype=tf.dtypes.uint8, name=None, expand_animations=True)
    image_np = image.numpy()
    return image_np


def decode_tree(byte_string):
    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
    forest.ParseFromString(byte_string)
    # print(forest)
    # print(forest.windows[0])
    # ['bounds_in_screen', 'display_id', 'id', 'layer', 'title', 'window_type', 
    #  'is_accessibility_focused', 'is_active', 'is_focused', 'is_in_picture_in_picture_mode', 'tree']
    # print(forest.windows[0].tree)
    # print(forest.windows[0].tree.nodes[0])  # 一般一个页面有几十个节点
    # ['unique_id', 'bounds_in_screen', 'class_name', 'content_description', 
    # 'hint_text', 'package_name', 'text', 'text_selection_start', 'text_selection_end', 
    # 'view_id_resource_name', 'window_id', 'is_checkable', 'is_checked', 'is_clickable', 
    # 'is_editable', 'is_enabled', 'is_focusable', 'is_focused', 'is_long_clickable', 
    # 'is_password', 'is_scrollable', 'is_selected', 'is_visible_to_user', 'actions', 
    # 'child_ids', 'clickable_spans', 'depth', 'labeled_by_id', 'label_for_id', 'drawing_order', 
    # 'tooltip_text']

    return forest


def convert_win_to_xml(window, save_path=None):
    nodes = {x.unique_id: x for x in window.tree.nodes}

    all_child_ids = []
    for x in window.tree.nodes:
        all_child_ids.extend(x.child_ids)

    # find root node
    root_id = set(nodes.keys()) - set(all_child_ids)
    assert len(root_id) == 1, root_id
    root_node = nodes[root_id.pop()]

    # 创建根元素，并在构建树时处理所有节点
    xml_root = build_xml_tree(root_node, nodes)

    # 将构建好的树转换为 ElementTree，并保存到文件
    tree = ET.ElementTree(xml_root)
    if save_path:
        tree.write(save_path, encoding='utf-8', xml_declaration=True)

    # 检查树的字符串表示
    xml_str = ET.tostring(xml_root, encoding='unicode')
    try:
        tree = ET.fromstring(xml_str)
    except:
        raise Exception('Viewtree decode failed')
    return xml_str

def build_xml_tree(node, nodes):
    # 使用 process_node 处理当前节点
    tag, params = process_node(node)
    et_node = ET.Element(tag, **params)

    # 递归处理子节点
    for cid in node.child_ids:
        child_node = nodes[cid]
        child_et_node = build_xml_tree(child_node, nodes)
        et_node.append(child_et_node)

    return et_node


def process_node(node):
    tag = node.class_name.replace('$', '.').split('.')[-1]
    tag = re.sub(r'[^\w]', '', tag)
    if len(tag) == 0:
        tag = 'Node'        # Node 还是 View
    id = str(node.unique_id)
    _b = node.bounds_in_screen
    boundsInScreen = f"Rect({_b.left}, {_b.top} - {_b.right}, {_b.bottom})"

    params = dict(boundsInScreen=boundsInScreen, id=id)
    if node.content_description:
        params.update(contentDescription=node.content_description)

    for attr_name in ['package_name', 'drawing_order', 'tooltip_text', 'hint_text']:
        if getattr(node, attr_name):
            params.update({attr_name: str(getattr(node, attr_name))})

    # unused: 'text_selection_start', 'text_selection_end', 'window_id', 'actions', 'child_ids', 'depth',
    # 'clickable_spans', 'label_for_id', 'labeled_by_id'

    # for attr_name in ['labeled_by_id']:
    #     if getattr(node, attr_name):
    #         raise ValueError(f'attr_name: {attr_name} {getattr(node, attr_name)}')
    #         params.update({attr_name: getattr(node, attr_name)})

    if node.is_enabled:
        params.update(enabled="true")
    if node.is_visible_to_user:
        params.update(visible="true")
    if node.is_focusable:
        params.update(focusable="true")
    if node.is_focused:
        params.update(focused="true")
    if node.is_clickable:
        params.update(clickable="true")
    if node.is_long_clickable:
        params.update(long_clickable="true")
    if node.is_editable:
        params.update(editable="true")
    if node.is_checked:
        params.update(checked="true")
    if node.is_checkable:
        params.update(checkable="true")
    if node.is_scrollable:
        params.update(scrollable="true")
    if node.is_selected:
        params.update(selected="true")
    if node.is_password:
        params.update(password="true")
    if node.text:
        params.update(text=node.text)
    if node.view_id_resource_name:
        params.update(viewIdResName=node.view_id_resource_name)

    return tag, params
