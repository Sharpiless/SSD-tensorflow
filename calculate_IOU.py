import numpy as np
import config as cfg


def encode_targets(true_box, anchors, prior_scaling=cfg.PRIOT_SCALING):

    anchor_y_min = anchors[:, 0]
    anchor_x_min = anchors[:, 1]
    anchor_y_max = anchors[:, 2]
    anchor_x_max = anchors[:, 3]

    anchor_ctr_y = (anchor_y_max + anchor_y_min) / 2
    anchor_ctr_x = (anchor_x_max + anchor_x_min) / 2
    anchor_h = anchor_y_max - anchor_y_min
    anchor_w = anchor_x_max - anchor_x_min

    true_box_y_min = true_box[:, 0]
    true_box_x_min = true_box[:, 1]
    true_box_y_max = true_box[:, 2]
    true_box_x_max = true_box[:, 3]

    true_box_ctr_y = (true_box_y_max + true_box_y_min) / 2
    true_box_ctr_x = (true_box_x_max + true_box_x_min) / 2
    true_box_h = true_box_y_max - true_box_y_min
    true_box_w = true_box_x_max - true_box_x_min

    target_dy = (true_box_ctr_y-anchor_ctr_y)/anchor_h
    target_dx = (true_box_ctr_x-anchor_ctr_x)/anchor_w
    target_dh = np.log(true_box_h/anchor_h)
    target_dw = np.log(true_box_w/anchor_w)

    targets = np.stack([target_dy, target_dx, target_dh, target_dw], axis=1)

    return np.reshape(targets, (-1, 4)) / prior_scaling


def decode_targets(anchors, targets, image_shape, prior_scaling=cfg.PRIOT_SCALING):

    y_min = anchors[:, 0]
    x_min = anchors[:, 1]
    y_max = anchors[:, 2]
    x_max = anchors[:, 3]

    height, width = image_shape[:2]

    ctr_y = (y_max + y_min) / 2
    ctr_x = (x_max + x_min) / 2
    h = y_max - y_min
    w = x_max - x_min

    targets = targets * prior_scaling

    dy = targets[:, 0]
    dx = targets[:, 1]
    dh = targets[:, 2]
    dw = targets[:, 3]

    pred_ctr_y = dy*h + ctr_y
    pred_ctr_x = dx*w + ctr_x
    pred_h = h*np.exp(dh)
    pred_w = w*np.exp(dw)

    y_min = pred_ctr_y - pred_h/2
    x_min = pred_ctr_x - pred_w/2
    y_max = pred_ctr_y + pred_h/2
    x_max = pred_ctr_x + pred_w/2

    y_min = np.clip(y_min, 0, height)
    y_max = np.clip(y_max, 0, height)
    x_min = np.clip(x_min, 0, width)
    x_max = np.clip(x_max, 0, width)

    boxes = np.stack([y_min, x_min, y_max, x_max], axis=1)

    return boxes


def fast_bbox_overlaps(holdon_anchor, true_boxes):

    num_true = true_boxes.shape[0]  # 真值框的个数 m
    num_holdon = holdon_anchor.shape[0]  # 候选框的个数（已删去越界的样本）n

    true_y_max = true_boxes[:, 2]
    true_y_min = true_boxes[:, 0]
    true_x_max = true_boxes[:, 3]
    true_x_min = true_boxes[:, 1]

    anchor_y_max = holdon_anchor[:, 2]
    anchor_y_min = holdon_anchor[:, 0]
    anchor_x_max = holdon_anchor[:, 3]
    anchor_x_min = holdon_anchor[:, 1]

    true_h = true_y_max - true_y_min
    true_w = true_x_max - true_x_min

    true_h = np.expand_dims(true_h, axis=1)
    true_w = np.expand_dims(true_w, axis=1)

    anchor_h = holdon_anchor[:, 2] - holdon_anchor[:, 0]
    anchor_w = holdon_anchor[:, 3] - holdon_anchor[:, 1]

    true_area = true_w * true_h
    anchor_area = anchor_w * anchor_h

    min_y_up = np.expand_dims(true_y_max, axis=1) < anchor_y_max
    min_y_up = np.where(min_y_up, np.expand_dims(
        true_y_max, axis=1), np.expand_dims(anchor_y_max, axis=0))

    max_y_down = np.expand_dims(true_y_min, axis=1) > anchor_y_min
    max_y_down = np.where(max_y_down, np.expand_dims(
        true_y_min, axis=1), np.expand_dims(anchor_y_min, axis=0))

    lh = min_y_up - max_y_down

    min_x_up = np.expand_dims(true_x_max, axis=1) < anchor_x_max
    min_x_up = np.where(min_x_up, np.expand_dims(
        true_x_max, axis=1), np.expand_dims(anchor_x_max, axis=0))

    max_x_down = np.expand_dims(true_x_min, axis=1) > anchor_x_min
    max_x_down = np.where(max_x_down, np.expand_dims(
        true_x_min, axis=1), np.expand_dims(anchor_x_min, axis=0))

    lw = min_x_up - max_x_down

    pos_index = np.where(
        np.logical_and(
            lh > 0, lw > 0
        )
    )

    overlap_area = lh * lw  # (n, m)

    overlap_weight = np.zeros(shape=lh.shape, dtype=np.int)

    overlap_weight[pos_index] = 1

    all_area = true_area + anchor_area

    dialta_S = all_area - overlap_area

    dialta_S = np.where(dialta_S > 0, dialta_S, all_area)

    IOU = np.divide(overlap_area, dialta_S)

    IOU = np.where(overlap_weight, IOU, 0)

    IOU_s = np.transpose(IOU)

    return IOU_s.astype(np.float32)  # (n, m) 转置矩阵


if __name__ == "__main__":

    pass
