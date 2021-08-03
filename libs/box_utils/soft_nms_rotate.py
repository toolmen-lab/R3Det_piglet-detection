# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from libs.configs import cfgs
import tensorflow as tf


def softnms_rotate(decode_boxes, scores, iou_threshold, final_threshold=0.001):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    keep = tf.py_func(softnms_rotate_cpu,
                      inp=[decode_boxes, scores, iou_threshold, final_threshold],
                      Tout=tf.int64)
    return keep


def softnms_rotate_cpu(boxes, scores, iou_threshold, final_threshold=0.001):
    """
    :param boxes: format [x_c, y_c, w, h, theta(degrees)]
    :param scores: scores of boxes
    :param iou_threshold: iou threshold (usually 0.7 or 0.3)
    :param final_threshold: usually 0.001, if weighted score less than this value, discard the box
    :return: the remaining INDEX of boxes
    Note that this function changes 
    """

    pos = 0             # a position index

    N = boxes.shape[0]  # number of input bounding boxes
    
    for i in range(N):

        maxscore = scores[i]
        maxpos   = i

        tbox   = boxes[i,:]    
        tscore = scores[i]

        pos = i + 1

        # get bounding box with maximum score
        while pos < N:
            if maxscore < scores[pos]:
                maxscore = scores[pos]
                maxpos = pos
            pos = pos + 1

        # Add max score bounding box as a detection result
        boxes[i,:] = boxes[maxpos,:]
        scores[i]  = scores[maxpos]
        # swap ith box with position of max box
        boxes[maxpos,:] = tbox
        scores[maxpos]  = tscore

        tbox   = boxes[i,:]
        tscore = scores[i]
        tarea  = tbox[2] * tbox[3]

        pos = i + 1

        # NMS iterations, note that N changes if detection boxes fall below final_threshold
        while pos < N:
            box   = boxes[pos, :]
            area  = box[2] * box[3]
            try:
                int_pts = cv2.rotatedRectangleIntersection(((tbox[0], tbox[1]), (tbox[2], tbox[3]), tbox[4]), ((box[0], box[1]), (box[2], box[3]), box[4]))[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area  = cv2.contourArea(order_pts)
                    inter = int_area * 1.0 / (tarea + area - int_area + cfgs.EPSILON)  # compute IoU
                else:
                    inter = 0
            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                inter = 0.9999

            # Soft NMS, weight computation.
            if inter > iou_threshold:
                weight = 1 - inter
            else:
                weight = 1
            scores[pos] = weight * scores[pos]

            # if box score fall below final_threshold, discard it by swapping with last box
            # also, update N
            if scores[pos] < final_threshold:
                boxes[pos, :] = boxes[N-1, :]
                scores[pos]   = scores[N-1]
                N = N - 1
                pos = pos - 1 

            pos = pos + 1

    keep = [i for i in range(N)]
    return np.array(keep, np.int64)


if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    scores = np.array([0.99, 0.88, 0.66, 0.77])

    keep = nms_rotate(tf.convert_to_tensor(boxes, dtype=tf.float32), tf.convert_to_tensor(scores, dtype=tf.float32),
                      0.7, 5)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with tf.Session() as sess:
        print(sess.run(keep))
