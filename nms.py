import numpy as np
import config as cfg


def py_cpu_nms(dets, thresh=cfg.NMS_THRESHOLD):

    y1 = dets[:, 0]

    x1 = dets[:, 1]

    y2 = dets[:, 2]

    x2 = dets[:, 3]

    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index

    order = scores.argsort()[::-1]

    # keep为最后保留的边框

    keep = []

    while order.size > 0:

        # order[0]是当前分数最大的窗口，肯定保留

        i = order[0]

        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积

        xx1 = np.maximum(x1[i], x1[order[1:]])

        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])

        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)

        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep
