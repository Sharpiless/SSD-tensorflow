import numpy as np
import config as cfg
from calculate_IOU import encode_targets, decode_targets, fast_bbox_overlaps


def ssd_bboxes_encode(anchors, true_boxes, true_labels, num_classes=len(cfg.CLASSES), threshold=cfg.THRESHOLD):

    labels, scores, loc = ssd_bboxes_encode_layer(
        anchors, true_boxes, true_labels, threshold=threshold
    )

    return labels.astype(np.float32), scores.astype(np.float32), loc.astype(np.float32)


def ssd_bboxes_encode_layer(anchors, true_boxes, true_labels, threshold=0.5):

    anchors = np.reshape(anchors, (-1, 4))
    true_boxes = np.reshape(true_boxes, (-1, 4))

    IOUs = fast_bbox_overlaps(anchors, true_boxes)

    max_arg = np.argmax(IOUs, axis=-1)
    index = np.arange(0, max_arg.shape[0])

    target_labels = true_labels[max_arg]
    target_scores = IOUs[index, max_arg]

    pos_index = np.where(target_scores > threshold)[0]
    pos_anchors = anchors[pos_index]
    pos_boxes = true_boxes[max_arg[pos_index]]

    target_loc = np.zeros(shape=anchors.shape)

    if pos_index.shape[0]:

        pos_targets = encode_targets(pos_boxes, pos_anchors)
        target_loc[pos_index, :] = pos_targets

    return target_labels, target_scores, target_loc


if __name__ == "__main__":

    from anchors import ssd_anchor_all_layers
    import matplotlib.pyplot as plt
    from read_data import Reader
    import cv2

    reader = Reader(is_training=True)

    while(True):

        value = reader.generate()

        image = value['image'].astype(np.int)
        image_copy = image.copy()
        true_labels = value['classes']
        true_boxes = value['boxes']

        image_shape = image.shape

        layers_anchors = ssd_anchor_all_layers(image)

        target_labels = []
        target_scores = []
        target_loc = []

        t = 0

        for anchors in layers_anchors:

            tmp = ssd_bboxes_encode(
                anchors, true_boxes, true_labels)

            target_labels.append(tmp[0])
            target_scores.append(tmp[1])
            target_loc.append(tmp[2])

        result_anchors = []
        result_labels = []
        result_loc = []

        for i in range(len(layers_anchors)):

            anchors = layers_anchors[i].reshape((-1, 4))
            scores = target_scores[i]
            labels = target_labels[i]
            loc = target_loc[i]

            pos_index = np.where(scores > 0.5)[0]

            if pos_index.shape[0]:

                result_anchors.append(anchors[pos_index])
                result_labels.append(labels[pos_index])
                result_loc.append(loc[pos_index])

        if not len(result_anchors)==0:
    
            result_anchors = np.vstack(result_anchors)
            result_loc = np.vstack(result_loc)
            result_labels = np.hstack(result_labels).astype(np.int)

            result_bboxes = decode_targets(result_anchors, result_loc, image_shape)

            font = cv2.FONT_HERSHEY_SIMPLEX

            for i in range(result_anchors.shape[0]):

                anchor = result_anchors[i]
                label = result_labels[i]
                label = cfg.CLASSES[label]

                y_min, x_min, y_max, x_max = anchor.astype(np.int)

                cv2.rectangle(image, (x_min, y_min),
                            (x_max, y_max), (0, 0, 255), thickness=2)

                bbox = result_bboxes[i]

                y_min, x_min, y_max, x_max = bbox.astype(np.int)

                cv2.rectangle(image_copy, (x_min, y_min),
                            (x_max, y_max), (255, 0, 0), thickness=2)

                cv2.putText(image_copy, label, (x_min+20, y_min+20),
                                font, 1, (0, 0, 255), thickness=2)

        num = true_boxes.shape[0]

        for i in range(num):

            box = true_boxes[i]

            y_min, x_min, y_max, x_max = box.astype(np.int)

            cv2.rectangle(image, (x_min, y_min),
                        (x_max, y_max), (0, 255, 0), thickness=2)

        final_image = np.vstack(
            [image[:, :, [2, 1, 0]], image_copy[:, :, [2, 1, 0]]]
        )

        plt.imshow(final_image)
        plt.show()
        # '../VOC2012\\JPEGImages\\2008_003621.jpg'