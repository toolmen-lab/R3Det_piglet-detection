# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import argparse
import numpy as np
import imageio
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_r3det
from libs.box_utils import draw_box_in_img
from help_utils import tools


def detect(det_net, inference_save_path, real_test_imgname_list):

    cfgs.NMS_IOU_THRESHOLD = args.nms_thresh
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)  # [1, None, None, 3]

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch_h=None,
        gtboxes_batch_r=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer(checkpoint_path=args.model_weight)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # reduce gpu memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        predictions = {}
        inference_time = []
        all_boxes_r = {}
        for i, a_img_name in enumerate(real_test_imgname_list):

            raw_img = cv2.imread(a_img_name)
            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            end = time.time()

            show_indices = detected_scores >= args.vis_score
            show_scores = detected_scores[show_indices]
            show_boxes = detected_boxes[show_indices]
            show_categories = detected_categories[show_indices]

            draw_img = np.squeeze(resized_img, 0)

            if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
                draw_img = (draw_img * np.array(cfgs.PIXEL_STD) + np.array(cfgs.PIXEL_MEAN_)) * 255
            else:
                draw_img = draw_img + np.array(cfgs.PIXEL_MEAN)
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                boxes=show_boxes,
                                                                                labels=show_categories,
                                                                                scores=show_scores,
                                                                                method=1,
                                                                                in_graph=False)

            nake_name = a_img_name.split('/')[-1]
            predictions[nake_name] = final_detections
            tools.view_bar('{} image cost {:.3f}s'.format(nake_name, (end - start)), i+1, len(real_test_imgname_list))
            inference_time.append(end - start)
            x_c, y_c, w, h, theta = show_boxes[:, 0], show_boxes[:, 1], show_boxes[:, 2], \
                                    show_boxes[:, 3], show_boxes[:, 4]
            boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
            dets_r = np.hstack((show_categories.reshape(-1, 1),
                                show_scores.reshape(-1, 1),
                                boxes_r))
            all_boxes_r[nake_name] = dets_r
            if args.show:
                copy = final_detections[:,:,::-1].copy()
                cv2.imshow('result', copy)
                cv2.waitKey(1)
        speed = len(inference_time) / sum(inference_time)
        print("Inference speed: {} fps".format(speed))
        if args.save_video:
            writer = imageio.get_writer(os.path.join(inference_save_path, "detection_result.mp4"), fps = 10)
            for _, pred in predictions.items():
                writer.append_data(pred)
            writer.close()
        if args.write_csv:
            write_csv(inference_save_path, all_boxes_r)


def write_csv(inference_save_path, all_boxes_r):
    import csv
    with open(os.path.join(inference_save_path, "rotated_boxes.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        for image_id, dets_r in all_boxes_r.items():
            for r in dets_r:
                _, score, cx, cy, w, h, theta = r
                row = [int(image_id[:-4]), -1, cx, cy, w, h, theta, score]
                writer.writerow(row)
    with open(os.path.join(inference_save_path, "horizontal_boxes.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        for image_id, dets_r in all_boxes_r.items():
            for r in dets_r:
                _, score, cx, cy, w, h, theta = r
                box = cv2.boxPoints(((cx, cy), (w, h), theta))
                box = np.reshape(box, [-1, ])
                xmin, ymin, xmax, ymax = min(box[0::2]), min(box[1::2]), max(box[0::2]), max(box[1::2])
                row = [int(image_id[:-4]), -1, xmin, ymin, xmax, ymax, score]
                writer.writerow(row)

def inference(inference_save_path):

    test_imgname_list = [os.path.join(inference_save_path, img_name) for img_name in os.listdir(inference_save_path)
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    retinanet = build_whole_network_r3det.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                             is_training=False)
    detect(det_net=retinanet, inference_save_path=inference_save_path, real_test_imgname_list=sorted(test_imgname_list))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('--save_dir', dest='save_dir',
                        help='demo imgs to save',
                        default='../output/inference', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu id ',
                        default='0', type=str)
    parser.add_argument('--show', dest='show', default=False,
                        action='store_true' ,help='show result')
    parser.add_argument('--model_weight', dest='model_weight',
                        help='model_weight',
                        default='../output/trained_weights/RetinaNet_PIGLET_20200603/piglet_140000_model.ckpt', type=str)
    parser.add_argument('--vis_score', type=float, dest='vis_score', default=0.4,
                        help='visualize score')
    parser.add_argument('--save_video', dest='save_video', default=False,
                        action='store_true', help='save video')
    parser.add_argument('--write_csv', dest='write_csv', default=True,
                        action='store_true', help='write csv')
    parser.add_argument('--nms_thresh', type=float, dest='nms_thresh', default=0.2,
                        help='nms threshold')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    inference(inference_save_path=args.save_dir)
