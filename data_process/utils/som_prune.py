import cv2
import io
import os
import base64
from PIL import Image
import numpy as np

ACTION_MISSED = ''
import pyshine as ps

def put_bounded_text(imgcv, label, text_offset_x, text_offset_y, vspace, hspace,
                     font_scale, thickness, background_RGB, text_RGB, alpha):
    # appagent 形式
    # 获取图像的高度和宽度
    img_height, img_width, _ = imgcv.shape

    # 获取文本的大小
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # 检查文本框是否超出图像范围，并进行相应调整
    if text_offset_x + text_size[0] + hspace > img_width:
        text_offset_x = img_width - text_size[0] - hspace
    if text_offset_y + text_size[1] + vspace > img_height:
        text_offset_y = img_height - text_size[1] - vspace

    # 绘制文本框
    imgcv = ps.putBText(imgcv, label, text_offset_x=text_offset_x, text_offset_y=text_offset_y,
                        vspace=vspace, hspace=hspace, font_scale=font_scale,
                        thickness=thickness, background_RGB=background_RGB,
                        text_RGB=text_RGB, alpha=alpha)

    return imgcv


def convert_image_base64(image, max_size_mb, save_path):
    # Reduce size iteratively until the size is below the limit
    scale_factor = 1.0
    max_bytes = max_size_mb * 1000 * 1000
    buffer = io.BytesIO()
    # the openai-gpt4o needs the image to be transfered to base64, and no larger than 20MB
    while True:
        buffer.seek(0)
        image.save(buffer, format='PNG', optimize=True)
        if buffer.tell() <= max_bytes:
            break
        scale_factor *= 0.95
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.ANTIALIAS)

    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    if save_path:
        image.save(save_path)
    return image_base64


def calculate_iou(boxA, boxB):
    # 计算两个边界框的交集
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集的宽度和高度
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个边界框的面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集
    unionArea = boxAArea + boxBArea - interArea + 1e-6

    # 计算IoU
    iou = interArea / float(unionArea)

    return iou


def calculate_overlap_area(box1, box2):
    # 计算两个矩形的重叠区域
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    return x_overlap * y_overlap


def calculate_overlap_rate_and_max_subbox(boxes, index, valid_indices):
    boxes = boxes.astype(np.int32)
    original_box = boxes[index]
    # 计算original_box的宽度和高度
    width = original_box[2] - original_box[0]
    height = original_box[3] - original_box[1]

    # 初始化一个与original_box大小一致的全1的NumPy数组
    overlap_array = np.ones((height, width), dtype=np.uint8)

    # 遍历每个other_box，更新overlap_array
    for ind, other_box in enumerate(boxes):
        if ind == index or ind not in valid_indices:
            continue
        # 计算重叠区域的坐标
        overlap_x1 = max(original_box[0], other_box[0])
        overlap_y1 = max(original_box[1], other_box[1])
        overlap_x2 = min(original_box[2], other_box[2])
        overlap_y2 = min(original_box[3], other_box[3])

        # 在overlap_array中将重叠区域设置为0
        overlap_array[overlap_y1 - original_box[1]:max(overlap_y2 - original_box[1], 0),
                      overlap_x1 - original_box[0]:max(overlap_x2 - original_box[0], 0)] = 0

    # 计算重叠率
    overlap_rate = 1 - (np.sum(overlap_array == 1) / (width * height))
    x1, y1, x2, y2 = find_max_sub_box(overlap_array)

    # cv2.imshow('Overlap Array', cv2.resize(overlap_array * 255, None, fx=0.4, fy=0.4))  # 乘以255转为可视化的灰度图
    # overlap_img = np.stack([overlap_array] * 3, -1)
    # cv2.rectangle(overlap_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    # cv2.imshow('Contours', cv2.resize(overlap_img, None, fx=0.4, fy=0.4))
    # cv2.waitKey()

    max_subbox = [original_box[0] + x1, original_box[1] + y1,
                  original_box[0] + x2, original_box[1] + y2]
    return overlap_rate, max_subbox


def find_max_histogram_area(heights):
    stack = []
    max_area = 0
    max_rect = (0, 0, 0, 0)  # (start index, end index, height)

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            max_area = max(max_area, height * (i - index))
            start = index
            if max_area == height * (i - index):
                max_rect = (index, i - 1, height)
        stack.append((start, h))

    for i, h in stack:
        max_area = max(max_area, h * (len(heights) - i))
        if max_area == h * (len(heights) - i):
            max_rect = (i, len(heights) - 1, h)

    return max_area, max_rect


# Main function to find the largest rectangle of 1's in a binary matrix.
def find_max_sub_box(matrix, f=0.1):
    matrix = cv2.resize(matrix, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    if not matrix.size:
        return 0, (0, 0, 0, 0)

    n, m = matrix.shape
    height = np.zeros(m, dtype=int)
    max_area = 0
    max_rect = (0, 0, 0, 0)  # (top left corner x, y, bottom right corner x, y)

    for i in range(n):
        # Update the height array to reflect the number of consecutive ones in the column.
        height[matrix[i] == 0] = 0
        height[matrix[i] == 1] += 1

        # Find the largest rectangle in the histogram.
        area, rect = find_max_histogram_area(height)
        if area > max_area:
            max_area = area
            # rect is (start index, end index, height), convert it to coordinates.
            max_rect = (rect[0], i - rect[2] + 1, rect[1], i)
    max_rect = (np.array(max_rect) / f).astype(np.int32)
    return max_rect


def calculate_iof(boxA, boxB):
    # 计算两个边界框的交集
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集的面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算参考边界框的面积
    refArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) + 1e-6

    # 计算IoF
    iof = interArea / float(refArea)
    return iof


def calculate_pairwise_iof(boxes):
    # boxes 是一个二维列表，其中每个元素是一个表示边界框的列表 [x1, y1, x2, y2]
    num_boxes = len(boxes)
    iofs = np.zeros([num_boxes, num_boxes])
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                iofs[i][j] = calculate_iof(boxes[i], boxes[j])
    return iofs

def generate_colormaps(num_colors=100, seed=42):
    assert num_colors >= 100, "The number of colors should be at least 100."
    
    # 设置随机数生成器的种子，以确保每次生成的颜色是相同的
    np.random.seed(seed)
    
    # 生成随机颜色（RGB）
    random_colors_rgb = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    
    # 创建color_map_text字典（先加入None键）
    color_map_text = {
        None: (0, 250, 0)  # 默认颜色
    }
    
    # 将随机生成的颜色加入color_map_text字典
    color_map_text.update({str(i): tuple(color) for i, color in enumerate(random_colors_rgb)})
    
    # 创建color_map_bound字典，并转换为BGR
    color_map_bound = {
        None: cv2.cvtColor(np.uint8([[color_map_text[None]]]), cv2.COLOR_RGB2BGR)[0][0]
    }
    color_map_bound.update({
        str(i): tuple(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2BGR)[0][0])
        for i, color in enumerate(random_colors_rgb)
    })
    
    return color_map_text, color_map_bound

# 构建一个完整的层次结构列表，确保所有父节点都包括在内
def get_full_hierarchy(hierarchy):
    all_hierarchies = set(hierarchy)
    for h in hierarchy:
        parts = h.split('.')
        for i in range(1, len(parts)):
            parent_hierarchy = '.'.join(parts[:i])
            all_hierarchies.add(parent_hierarchy)
    return sorted(list(all_hierarchies))
def draw_som_image(image_path, bounds_infos, hierarchy, bounds_keys, save_path=None, color=(0, 250, 0), max_size_mb=20):
    """
    :param image_path: 输入图像的路径
    :param bounds_infos: 所有节点的边界信息
    :param hierarchy: 节点所属层次
    :param bounds_keys: 节点的键值
    :param save_path: 保存整体图像的路径
    :param color: 边框颜色
    :param max_size_mb: 保存图像的最大大小
    :return: 返回base64格式的图像字符串
    """
    # 生成至少100种颜色的colormap
    color_map_text, color_map_bound = generate_colormaps(100)

    imgcv = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = imgcv.shape
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    xyxy = np.array(bounds_infos).astype(np.float32).reshape(-1, 4)

    # 按面积从大到小排序
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    sorted_indices = np.argsort(areas)[::-1]
    xyxy = xyxy[sorted_indices]
    bounds_keys = [bounds_keys[ind] for ind in sorted_indices]
    hierarchy = [hierarchy[ind] for ind in sorted_indices]  # 重新排序hierarchy

    # 去除所有没有 index 的 boxes
    valid_indices = [ind for ind, label in enumerate(bounds_keys) if label != '']
    xyxy = xyxy[valid_indices]
    bounds_keys = [bounds_keys[ind] for ind in valid_indices]
    hierarchy = [hierarchy[ind] for ind in valid_indices]  # 过滤有效hierarchy
    valid_indices = list(range(len(bounds_keys)))

    # 计算 IoF
    iofs = calculate_pairwise_iof(xyxy)

    # 适当缩小 boxes
    wh = xyxy[:, 2:] - xyxy[:, :2]
    cxcy = (xyxy[:, 2:] + xyxy[:, :2]) / 2
    wh[:, :1] = wh[:, :1] * 0.95
    wh[:, 1:] = wh[:, 1:] * 0.98
    xyxy[:, :2] = cxcy - wh / 2
    xyxy[:, 2:] = cxcy + wh / 2
    bounds_infos = xyxy.reshape(-1, 2, 2).astype(np.int32).tolist()

    # 获取完整的hierarchy列表
    full_hierarchy = get_full_hierarchy(hierarchy)

    # 保存每个 hierarchy 的单独图像
    for parent_hierarchy in full_hierarchy:
        # 查找所有以当前 parent_hierarchy 开头的子节点
        child_indices = [j for j, h in enumerate(hierarchy) if h.startswith(f"{parent_hierarchy}.")]
        
        # 计算父节点和所有子孙节点的总数
        total_nodes = len(child_indices) + (1 if parent_hierarchy in hierarchy else 0)
        
        # 如果总数大于5，进行可视化
        if total_nodes > 1:
            print(f"Visualizing hierarchy: {parent_hierarchy} with {total_nodes} nodes (including parent).")
            hierarchy_imgcv = imgcv.copy()
            indices_to_draw = ([hierarchy.index(parent_hierarchy)] if parent_hierarchy in hierarchy else []) + child_indices  # 包括父节点和其子节点

            for idx in indices_to_draw:
                bbox = bounds_infos[idx]
                current_color_text = color_map_text.get(hierarchy[idx], color_map_text[None])
                current_color_bound = color_map_bound.get(hierarchy[idx], color_map_bound[None])
                if current_color_bound is not None:
                    current_color_bound = tuple(map(int, current_color_bound))
                if bounds_keys is None:
                    label = str(idx + 1)
                else:
                    label = bounds_keys[idx]

                bbox[0][0] = min(max(20, bbox[0][0]), w - 20)  # pad on x axis
                bbox[1][0] = min(max(20, bbox[1][0]), w - 20)
                bbox[0][1] = min(max(20, bbox[0][1]), h - 20)  # pad on y axis
                bbox[1][1] = min(max(20, bbox[1][1]), h - 20)
                text_offset_x = int((bbox[0][0] + bbox[1][0]) // 2)
                text_offset_y = int((bbox[0][1] + bbox[1][1]) // 2)

                # 绘制矩形边框和透明填充矩形
                cv2.rectangle(hierarchy_imgcv, bbox[0], bbox[1], color=current_color_bound, thickness=2)
                bbox = np.array([[[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[1][0]), int(bbox[0][1])],
                                  [int(bbox[1][0]), int(bbox[1][1])], [int(bbox[0][0]), int(bbox[1][1])]]], dtype=np.int32)
                overlay = hierarchy_imgcv.copy()
                cv2.fillPoly(overlay, bbox, color=current_color_bound)
                alpha = 0.1
                hierarchy_imgcv = cv2.addWeighted(overlay, alpha, hierarchy_imgcv, 1 - alpha, 0)

                try:
                    put_bounded_text(hierarchy_imgcv, label, text_offset_x=text_offset_x - 20,
                                     text_offset_y=text_offset_y + 30,
                                     vspace=10, hspace=10, font_scale=1, thickness=2,
                                     background_RGB=current_color_text, text_RGB=(255, 250, 250), alpha=0.5)
                except Exception as e:
                    print(f"Error in putBText: {e}")

            # 保存每个 hierarchy 的图像
            save_dir, save_filename = os.path.split(save_path)
            hierarchy_save_dir = save_dir.replace('som_all', 'hierarchy')
            os.makedirs(hierarchy_save_dir, exist_ok=True)
            hierarchy_save_path = os.path.join(hierarchy_save_dir, f"{os.path.splitext(save_filename)[0]}_hierarchy_{parent_hierarchy}.jpg")
            hierarchy_image = Image.fromarray(hierarchy_imgcv)
            hierarchy_image.save(hierarchy_save_path)
            print(f"Saved hierarchy {parent_hierarchy} visualization to {hierarchy_save_path}")

    # 绘制所有节点到一张图上（不区分group）
    for idx, bbox in enumerate(bounds_infos):
        # 获取颜色并确保为数值型 BGR 颜色元组
        current_color_bound = color_map_bound.get(None, (0, 250, 0))  # 使用默认颜色
        if current_color_bound is not None:
            current_color_bound = tuple(map(int, current_color_bound))

        label = bounds_keys[idx] if bounds_keys else str(idx + 1)

        bbox[0][0] = min(max(20, bbox[0][0]), w - 20)  # pad on x axis
        bbox[1][0] = min(max(20, bbox[1][0]), w - 20)
        bbox[0][1] = min(max(20, bbox[0][1]), h - 20)  # pad on y axis
        bbox[1][1] = min(max(20, bbox[1][1]), h - 20)
        text_offset_x = int((bbox[0][0] + bbox[1][0]) // 2)
        text_offset_y = int((bbox[0][1] + bbox[1][1]) // 2)

        # 绘制矩形边框和透明填充矩形
        cv2.rectangle(imgcv, bbox[0], bbox[1], color=current_color_bound, thickness=2)
        bbox = np.array([[[int(bbox[0][0]), int(bbox[0][1])], [int(bbox[1][0]), int(bbox[0][1])],
                          [int(bbox[1][0]), int(bbox[1][1])], [int(bbox[0][0]), int(bbox[1][1])]]], dtype=np.int32)
        overlay = imgcv.copy()
        cv2.fillPoly(overlay, bbox, color=current_color_bound)
        alpha = 0.1
        imgcv = cv2.addWeighted(overlay, alpha, imgcv, 1 - alpha, 0)

        try:
            put_bounded_text(imgcv, label, text_offset_x=text_offset_x - 20,
                             text_offset_y=text_offset_y + 30,
                             vspace=10, hspace=10, font_scale=1, thickness=2,
                             background_RGB=current_color_bound, text_RGB=(255, 250, 250), alpha=0.5)
        except Exception as e:
            print(f"Error in putBText: {e}")

    # 保存最终的完整图像
    image = Image.fromarray(imgcv)
    image = image.convert('RGB')
    image_base64 = convert_image_base64(image, max_size_mb, save_path)
    return image_base64
