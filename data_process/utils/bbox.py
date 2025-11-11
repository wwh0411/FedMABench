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