import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from utils.decode_prune import process_node


def collect_all_child_ids(node_id, node_dict, node_dict_all):
    """
    递归收集节点的所有子节点及其子节点的ID，仅保留在 node_dict 中存在的ID
    :param node_id: 当前节点的ID
    :param node_dict: 只包含需要保留节点的字典
    :param node_dict_all: 包含所有节点（包括被过滤掉的节点）的字典
    :return: 包含所有在 node_dict 中的子节点及其子节点ID的列表
    """
    all_child_ids = []
    if node_id in node_dict_all:
        node = node_dict_all[node_id]
        for child_id in node.child_ids:
            if child_id in node_dict:  # 仅当子节点在 node_dict 中时才添加
                all_child_ids.append(child_id)
                all_child_ids.extend(collect_all_child_ids(child_id, node_dict, node_dict_all))
    return all_child_ids

def build_hierarchy_paths(parent_hierarchy, child_ids, node_dict):
    """
    为每个子节点生成hierarchy路径，并递归为其子节点生成路径
    :param parent_hierarchy: 父节点的hierarchy路径
    :param child_ids: 当前节点的所有子节点ID
    :param node_dict: 包含需要保留节点的字典
    :return: 每个子节点的hierarchy路径字典
    """
    hierarchy_paths = {}
    for i, child_id in enumerate(child_ids):
        child_hierarchy = f"{parent_hierarchy}.{i}"
        hierarchy_paths[child_id] = child_hierarchy
        if child_id in node_dict:
            # 递归生成子节点的hierarchy路径
            child_paths = build_hierarchy_paths(child_hierarchy, node_dict[child_id].child_ids, node_dict)
            hierarchy_paths.update(child_paths)
    return hierarchy_paths

def build_xml_tree(et_node, child_ids, node_dict, node_dict_all, parent_hierarchy="0"):
    hierarchy_paths = build_hierarchy_paths(parent_hierarchy, child_ids, node_dict_all)

    for nid in child_ids:
        tag, params = process_node(node_dict[nid])

        # 使用 node_dict_all 来收集子孙节点，但仅保留在 node_dict 中存在的ID
        all_child_ids = collect_all_child_ids(nid, node_dict, node_dict_all)
        if all_child_ids:
            params['all_child_ids'] = str(all_child_ids)

        # 添加 hierarchy 属性
        hierarchy = hierarchy_paths.get(nid, parent_hierarchy)
        params['hierarchy'] = hierarchy

        # 构建XML节点
        child_et_node = ET.SubElement(et_node, tag, **params)

        # 只保留 node_dict 中的节点来构建树
        filtered_child_ids = [cid for cid in node_dict[nid].child_ids if cid in node_dict]
        build_xml_tree(child_et_node, filtered_child_ids, node_dict, node_dict_all, hierarchy)


def dump_nodes_to_xml(node_list_all, node_list, xml_path, light_xml_path=None):
    keep_node_ids = [x.unique_id for x in node_list]
    node_dict = {node.unique_id: node for node in node_list}
    node_dict_all = {node.unique_id: node for node in node_list_all}
    child_ids = []
    for node in node_list:  # 先找出所有根节点
        child_ids += [nid for nid in node.child_ids if nid in keep_node_ids]
    root_ids = list(set(keep_node_ids) - set(child_ids))
    xml_root = ET.Element('Root')
    build_xml_tree(xml_root, root_ids, node_dict, node_dict_all)
    tree = ET.ElementTree(xml_root)
    xml_str = ET.tostring(xml_root, encoding="utf-8").decode("utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="\t")
    assert len([x for x in tree.iter()]) == len(node_list) + 1, f'tree node error'
    if xml_path:
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    # print(pretty_xml)
    # check available
    xml_root = ET.fromstring(xml_str)
    tree = ET.ElementTree(xml_root)
    for node in tree.iter():
        light_node(node)
    xml_str = ET.tostring(xml_root, encoding="utf-8").decode("utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="\t")
    if light_xml_path:
        tree.write(light_xml_path, encoding='utf-8', xml_declaration=True)
    return pretty_xml


def is_point_in_node(pt, node):
    b = node.bounds_in_screen
    box = [b.left, b.top, b.right, b.bottom]
    if box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]:
        return 1
    return 0


def light_node(node):
    del_key = ['boundsInScreen', 'package_name', 'drawing_order', 'enabled', 'visible']
    replace_key = {'contentDescription': 'content-desc', 'viewIdResName': 'resource-id'}
    for key in del_key:
        if key in node.attrib:
            node.attrib.pop(key)
    for old, new in replace_key.items():
        if old in node.attrib:
            value = node.attrib[old]
            del node.attrib[old]
            node.set(new, value)
    if 'resource-id' in node.attrib:
        resource_id = node.attrib['resource-id']
        node.attrib['resource-id'] = "/".join(resource_id.split("/")[1:])
    if 'content-desc' in node.attrib:
        node.attrib['content-desc'] = node.attrib['content-desc'][:50]
    if 'text' in node.attrib:
        node.attrib['text'] = node.attrib['text'][:50]


def reindex_nodes(node_list, start_idx=0):
    new_id = start_idx
    nid_map = {}
    for node in node_list:
        nid_map[node.unique_id] = new_id
        new_id += 1
    for node in node_list:
        node.unique_id = nid_map[node.unique_id]
        node.child_ids[:] = [nid_map[cid] for cid in node.child_ids]


def extract_nodes(windows, h, w):
    # 不同 window 的 id 会有重复!
    all_node_list = []
    visible_mask = np.ones((h, w))
    win_id_list = np.argsort([win.layer for win in windows])[::-1]
    start_node_id = 0
    for win_id in win_id_list:
        win = windows[win_id]
        b = win.bounds_in_screen
        win_box = [b.left, b.top, b.right, b.bottom]
        win_layer = win.layer
        win_type = win.window_type
        win_is_accessibility_focused = win.is_accessibility_focused
        win_is_active = win.is_active
        win_is_focused = win.is_focused
        node_list = win.tree.nodes
        reindex_nodes(node_list, start_idx=start_node_id)
        start_node_id += len(node_list)
        node_list_all = node_list
        node_list = filter_node_by_attribute(node_list)
        node_list = filter_node_by_window(node_list, visible_mask)
        visible_mask[b.top:b.bottom, b.left:b.right] = 0
        # 特殊窗口处理
        if win_type == 2:  # TYPE_INPUT_METHOD 输入法只保留回车键,
            all_node_list += [node for node in node_list
                              if node.view_id_resource_name.endswith('key_pos_ime_action')]
            continue
        if win_is_active:
            node_list = filter_node_by_bounds(node_list, h, w)
            node_list = filter_node_by_hierarchy(node_list)
            all_node_list += node_list
    return all_node_list, node_list_all
    # return node_list_all, node_list_all

def filter_node_by_window(node_list, visible_mask):
    filter_nodes = []
    for node in node_list:
        b = node.bounds_in_screen
        box = [b.left, b.top, b.right, b.bottom]
        visible_area = np.sum(visible_mask[b.top:b.bottom, b.left:b.right])
        if visible_area > 0:
            filter_nodes.append(node)
    return filter_nodes

def is_same_midpoint(parent_node, child_node):
    parent_box = parent_node.bounds_in_screen
    child_box = child_node.bounds_in_screen
    
    # 计算父节点的中心点
    parent_center = (
        (parent_box.left + parent_box.right) // 2,
        (parent_box.top + parent_box.bottom) // 2
    )
    
    # 判断父节点中心点是否在子节点的边框内
    if (child_box.left <= parent_center[0] <= child_box.right) and \
    (child_box.top <= parent_center[1] <= child_box.bottom):
        # 如果在子节点边框内,返回1
        return 1
    return 0

def filter_node_by_hierarchy(node_list):
    keep_node_ids = [x.unique_id for x in node_list]
    node_dict = {node.unique_id: node for node in node_list}
    num_keep = 0
    while len(keep_node_ids) != num_keep:
        num_keep = len(keep_node_ids)
        for node in node_list:
            # 只有一个有效的子节点的话 将合并父子节点
            # TODO: 节点的属性合并
            child_ids = [nid for nid in node.child_ids if nid in keep_node_ids]
            if len(child_ids) == 1:
                cid = child_ids[0]

                parent_node = node_dict[node.unique_id]
                child_node = node_dict[cid]
                if is_same_midpoint(parent_node, child_node):
                    differing_attributes = compare_nodes(parent_node, child_node)

                    # 合并属性：如果父节点的属性值为空且子节点的属性不为空，将子节点的属性值赋给父节点
                    for attr, values in differing_attributes.items():
                        if not values["parent_value"]:  # parent is empty
                            setattr(parent_node, attr, values["child_value"])
                        else:  # parent is not empty
                            pass  # print(attr, values)  # TODO: merge
                    # 删除子节点
                    keep_node_ids.pop(keep_node_ids.index(cid))
                    node.child_ids[:] = node_dict[cid].child_ids
    return [node_dict[nid] for nid in keep_node_ids]

def filter_node_by_bounds(node_list, h , w):
    """
    根据bound筛选节点，返回合法的节点列表-最终方案
    1.边界点相对位置不合规，直接删除
    2.边框完全超出mask，直接删除
    3.最终只保留超出mask面积10%以内的node，其他的都删除
    :param node_list: 节点列表
    :param h: 屏幕高度
    :param w: 屏幕宽度
    :return: 合法的节点列表,被筛选掉的节点列表
    """
    # 初始化四个类别的列表
    bounds_filter_node = []
    all_node_list = []
    
    for node in node_list:
        b = node.bounds_in_screen
        box = [b.left, b.top, b.right, b.bottom]

        # 计算节点的屏幕外占比
        node_area = (b.right - b.left) * (b.bottom - b.top)
        out_of_bounds_area = 0
        if b.left < 0:
            out_of_bounds_area += abs(b.left) * (b.bottom - b.top)
        if b.top < 0:
            out_of_bounds_area += abs(b.top) * (b.right - b.left)
        if b.right > w:
            out_of_bounds_area += (b.right - w) * (b.bottom - b.top)
        if b.bottom > h:
            out_of_bounds_area += (b.bottom - h) * (b.right - b.left)

        out_of_bounds_ratio = (out_of_bounds_area / node_area) * 100 if node_area > 0 else 0

        # 检查是否是合法的框
        if 0 <= b.left < b.right <= w and 0 <= b.top < b.bottom <= h:
            all_node_list.append(node)
        elif 0< out_of_bounds_ratio <10:
            all_node_list.append(node)
        else:
            bounds_filter_node.append(node)
    
    return all_node_list

def compare_nodes(parent_node, child_node):
    """比较父节点和子节点的属性，找出不同的属性"""
    attributes_to_compare = [
            # 文本类
            "content_description",
            "hint_text",
            "text",
            "tooltip_text",

            # view_id_resource_name
            "view_id_resource_name",

            # 操作类
            "is_checkable",
            "is_clickable",
            "is_editable",
            "is_long_clickable",
            "is_scrollable",

            # 状态
            "is_checked",
            "is_enabled",
            "is_focused",
            "is_selected",

            # 其他
            "package_name",
            "is_focusable",
            "is_password",
            "is_visible_to_user",
            "labeled_by_id",
            "label_for_id"
        ]
    differing_attributes = {}
    for attr in attributes_to_compare:
        parent_attr = getattr(parent_node, attr)
        child_attr = getattr(child_node, attr)
        if parent_attr != child_attr and child_attr:
            differing_attributes[attr] = {"parent_value": parent_attr, "child_value": child_attr, }

    return differing_attributes


def filter_node_by_attribute(node_list):
    filter_nodes = []
    for node in node_list:
        b = node.bounds_in_screen
        box = [b.left, b.top, b.right, b.bottom]
        text = node.text
        content_desc = node.content_description
        hint_text = node.hint_text
        preserve = False
        # if text or content_desc:    # 388
        #     preserve = True
        if text or content_desc or hint_text:  # 389
            preserve = True
        # if node.is_focusable and node.is_clickable:   # 614 text or content_desc or hint_text
        #     preserve = True
        if node.is_clickable:  # 65186 688 97.58 4   text or content_desc or hint_text
            preserve = True
        # if node.is_editable or node.is_checkable:  # node.is_clickable or node.is_long_clickable or node.is_checked or node.is_scrollable
        #     preserve = True
        class_name = node.class_name
        resource_name = node.view_id_resource_name
        if preserve:
            filter_nodes.append(node)
    return filter_nodes
