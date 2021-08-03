from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import xml.etree.cElementTree as ET
from PIL import Image, ImageDraw
sys.path.append("../")

from data.io.image_preprocess_multi_gpu import short_side_resize
from libs.configs import cfgs
from libs.networks import build_whole_network_r3det
from libs.box_utils.coordinate_convert import backward_convert, get_horizen_minAreaRectangle
from help_utils import tools
from libs.box_utils import draw_box_in_img
from libs.detection_oprations.refinebox_target_layer_without_boxweight import refinebox_target_layer
from libs.box_utils import bbox_transform
from libs.label_name_dict.label_dict import NAME_LABEL_MAP
import math

class show_anchor_detector(build_whole_network_r3det.DetectionNetwork):
    def __init__(self, base_network_name, is_training):
        super().__init__(base_network_name, is_training)
        self.base_network_name = base_network_name
        self.is_training = is_training
        if cfgs.METHOD == 'H':
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)
        self.method = cfgs.METHOD

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch_h, gtboxes_batch_r, gpu_id=0):
        gtboxes_batch_h = tf.reshape(gtboxes_batch_h, [-1, 5])
        gtboxes_batch_h = tf.cast(gtboxes_batch_h, tf.float32)

        gtboxes_batch_r = tf.reshape(gtboxes_batch_r, [-1, 6])
        gtboxes_batch_r = tf.cast(gtboxes_batch_r, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        feature_pyramid = self.build_base_network(input_img_batch)

        # 2. build rpn
        rpn_box_pred_list, rpn_cls_score_list, rpn_cls_prob_list = self.rpn_net(feature_pyramid, 'rpn_net')

        # 3. generate_anchors
        anchor_list = self.make_anchors(feature_pyramid)

        rpn_box_pred = tf.concat(rpn_box_pred_list, axis=0)
        rpn_cls_score = tf.concat(rpn_cls_score_list, axis=0)
        # rpn_cls_prob = tf.concat(rpn_cls_prob_list, axis=0)
        anchors = tf.concat(anchor_list, axis=0)
        box_pred_list, cls_prob_list, proposal_list = rpn_box_pred_list, rpn_cls_prob_list, anchor_list
        for i in range(cfgs.NUM_REFINE_STAGE):
            refine_box_states, refine_boxes = self.refine_stage(input_img_batch,
                                                                gtboxes_batch_r,
                                                                box_pred_list,
                                                                cls_prob_list,
                                                                proposal_list,
                                                                feature_pyramid,
                                                                gpu_id,
                                                                pos_threshold=cfgs.REFINE_IOU_POSITIVE_THRESHOLD[i],
                                                                neg_threshold=cfgs.REFINE_IOU_NEGATIVE_THRESHOLD[i],
                                                                stage='' if i == 0 else '_stage{}'.format(i + 2),
                                                                proposal_filter=True if i == 0 else False)

        return refine_box_states, refine_boxes

    def refine_stage(self, input_img_batch, gtboxes_batch_r, box_pred_list, cls_prob_list, proposal_list,
                     feature_pyramid, gpu_id, pos_threshold, neg_threshold,
                     stage, proposal_filter=False):
        with tf.variable_scope('refine_feature_pyramid{}'.format(stage)):
            refine_feature_pyramid = {}
            refine_boxes_list = []
            for box_pred, cls_prob, proposal, stride, level in \
                    zip(box_pred_list, cls_prob_list, proposal_list,
                        cfgs.ANCHOR_STRIDE, cfgs.LEVEL):

                if proposal_filter:
                    box_pred = tf.reshape(box_pred, [-1, self.num_anchors_per_location, 5])
                    proposal = tf.reshape(proposal, [-1, self.num_anchors_per_location, 5 if self.method == 'R' else 4])
                    cls_prob = tf.reshape(cls_prob, [-1, self.num_anchors_per_location, cfgs.CLASS_NUM])

                    cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
                    box_pred_argmax = tf.cast(tf.reshape(tf.argmax(cls_max_prob, axis=-1), [-1, 1]), tf.int32)
                    indices = tf.cast(tf.cumsum(tf.ones_like(box_pred_argmax), axis=0), tf.int32) - tf.constant(1, tf.int32)
                    indices = tf.concat([indices, box_pred_argmax], axis=-1)

                    box_pred = tf.reshape(tf.gather_nd(box_pred, indices), [-1, 5])
                    proposal = tf.reshape(tf.gather_nd(proposal, indices), [-1, 5 if self.method == 'R' else 4])

                    if cfgs.METHOD == 'H':
                        x_c = (proposal[:, 2] + proposal[:, 0]) / 2
                        y_c = (proposal[:, 3] + proposal[:, 1]) / 2
                        h = proposal[:, 2] - proposal[:, 0] + 1
                        w = proposal[:, 3] - proposal[:, 1] + 1
                        theta = -90 * tf.ones_like(x_c)
                        proposal = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
                else:
                    box_pred = tf.reshape(box_pred, [-1, 5])
                    proposal = tf.reshape(proposal, [-1, 5])

                bboxes = bbox_transform.rbbox_transform_inv(boxes=proposal, deltas=box_pred)
                refine_boxes_list.append(bboxes)
                center_point = bboxes[:, :2] / stride

                refine_feature_pyramid[level] = self.refine_feature_op(points=center_point,
                                                                       feature_map=feature_pyramid[level],
                                                                       name=level)

            refine_box_pred_list, refine_cls_score_list, refine_cls_prob_list = self.refine_net(refine_feature_pyramid,
                                                                                                'refine_net{}'.format(stage))

            refine_box_pred = tf.concat(refine_box_pred_list, axis=0)
            refine_cls_score = tf.concat(refine_cls_score_list, axis=0)
            refine_boxes = tf.concat(refine_boxes_list, axis=0)

            refine_labels, refine_target_delta, refine_box_states, refine_target_boxes = tf.py_func(
                        func=refinebox_target_layer,
                        inp=[gtboxes_batch_r, refine_boxes, pos_threshold, neg_threshold, gpu_id],
                        Tout=[tf.float32, tf.float32,
                              tf.float32, tf.float32])
            return refine_box_states, refine_boxes


def main():
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)
    gtboxes_and_label = tf.placeholder(dtype=tf.int32, shape=[None, 9])


    img_batch, gtboxes_and_label, img_h, img_w = short_side_resize(img_tensor=img_batch,
                                 gtboxes_and_label=gtboxes_and_label,
                                 target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                 length_limitation=cfgs.IMG_MAX_LENGTH)

    gtboxes_and_label_r = tf.py_func(backward_convert,
                                     inp=[gtboxes_and_label],
                                     Tout=tf.float32)
    gtboxes_and_label_r = tf.reshape(gtboxes_and_label_r, [-1, 6])

    gtboxes_and_label_h = get_horizen_minAreaRectangle(gtboxes_and_label)
    gtboxes_and_label_h = tf.reshape(gtboxes_and_label_h, [-1, 5])

    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    det_net = show_anchor_detector(base_network_name=cfgs.NET_NAME, is_training=False)
    refine_box_states, refine_boxes = \
                                det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                        gtboxes_batch_h=gtboxes_and_label_h,
                                                                        gtboxes_batch_r=gtboxes_and_label_r)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer(checkpoint_path =args.model_weight)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        imgs = sorted(os.listdir(args.img_dir))
        pbar = tqdm(imgs)
        for a_img_name in pbar:
            raw_img = cv2.imread(os.path.join(args.img_dir, a_img_name))
            gtbox_label = read_xml_gtbox_and_label(os.path.join(args.annotation_path, "{}.xml".format(a_img_name[:-4])))
            refine_box_states_, refine_boxes_ = \
                sess.run(
                    [refine_box_states, refine_boxes],
                    feed_dict={img_plac: raw_img[:, :, ::-1], gtboxes_and_label: gtbox_label})
            resized_img = cv2.resize(raw_img, (512,512))
            draw_anchors(a_img_name, resized_img, refine_boxes_, refine_box_states_, 1, "refine anchor")

def draw_anchors(img_name, img, anchors, anchor_states, method, save_dir):
    positive_anchor_indices = np.where(anchor_states == 1)
    print(positive_anchor_indices)
    positive_anchor = anchors[positive_anchor_indices]
    img_obj = Image.fromarray(img)
    raw_img_obj = img_obj.copy()
    draw_obj = ImageDraw.Draw(img_obj)
    for box in positive_anchor:
        draw_box_in_img.draw_a_rectangel_in_img(draw_obj,
                                                box = box,
                                                color = (0,0,255),
                                                method = method,
                                                width = 3,
                                                )
    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)
    pos_in_img = np.array(out_img_obj)
    save_dir = "../output/positive anchors/{}/{}".format(cfgs.VERSION, save_dir)
    tools.mkdir(save_dir)
    cv2.imwrite("{}/{}".format(save_dir, img_name), pos_in_img)

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                # if child_item.tag == 'bndbox':
                #     tmp_box = []
                #     for node in child_item:
                #         tmp_box.append(int(float(node.text)))
                #     assert label is not None, 'label is none, error'
                #     tmp_box.append(label)
                #     box_list.append(tmp_box)

                    # appendix
                if child_item.tag == 'robndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    cnt_x, cnt_y, w, h, theta = tmp_box
                    c = math.cos(-theta)
                    s = math.sin(-theta)
                    rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
                    # x: left->right ; y: top->down
                    rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
                    rotated_rect = [item for sub in rotated_rect for item in sub]
                    assert label is not None, 'label is none, error'
                    rotated_rect.append(label)
                    box_list.append(rotated_rect)

    gtbox_label = np.array(box_list, dtype=np.int32)
    return gtbox_label

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Draw training positive anchors')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='../data/io/PIGLET/train/images', type=str)
    parser.add_argument('--annotation_path', dest='annotation_path',
                        help='train annotate path',
                        default='../data/io/PIGLET/train/xmls', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--model_weight', dest='model_weight',
                        help='model_weight',
                        default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main()
