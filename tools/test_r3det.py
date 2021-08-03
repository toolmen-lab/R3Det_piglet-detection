# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import pickle
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_r3det
from libs.val_libs import voc_eval_r
from libs.val_libs import rotated_coco_eval
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
from help_utils import tools
from libs.configs import cfgs

from pycocotools.coco import COCO
import json
import itertools


def eval_with_plac(img_dir, det_net, num_imgs, image_ext, save_dir, draw_imgs=True):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

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

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        predictions = []
        all_boxes_r = []
        imgs = sorted(os.listdir(img_dir))
        pbar = tqdm(imgs)
        for image_id, a_img_name in enumerate(pbar):
            prediction = {"image_id": image_id}
            a_img_name = a_img_name.split(image_ext)[0]

            raw_img = cv2.imread(os.path.join(img_dir,
                                              a_img_name + image_ext))
            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            resized_img, det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}
                )

            detected_indices = det_scores_r_ >= args.vis_score
            detected_scores = det_scores_r_[detected_indices]
            detected_boxes = det_boxes_r_[detected_indices]
            detected_categories = det_category_r_[detected_indices]
            
            det_detections_r = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                boxes=detected_boxes,
                                                                                labels=detected_categories,
                                                                                scores=detected_scores,
                                                                                method=1,
                                                                                in_graph=True)

            cv2.imwrite('{}/{}.jpg'.format(save_dir, a_img_name),
                        det_detections_r[:, :, ::-1])

            if detected_boxes.shape[0] != 0:
                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                detected_boxes = forward_convert(detected_boxes, False)
                detected_boxes[:, 0::2] *= (raw_w / resized_w)
                detected_boxes[:, 1::2] *= (raw_h / resized_h)
                detected_boxes = backward_convert(detected_boxes, False)


            x_c, y_c, w, h, theta = detected_boxes[:, 0], detected_boxes[:, 1], detected_boxes[:, 2], \
                                    detected_boxes[:, 3], detected_boxes[:, 4]

            boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
            dets_r = np.hstack((detected_categories.reshape(-1, 1),
                                detected_scores.reshape(-1, 1),
                                boxes_r))
            all_boxes_r.append(dets_r)
            prediction["instances"] = instances_to_coco_json(dets_r , image_id)
            predictions.append(prediction)
            # print(predictions)

            pbar.set_description("Eval image %s" % a_img_name)

        with open(os.path.join(save_dir, "coco_instances_results.json"), "w") as f:
            f.write(json.dumps(predictions))

        if args.use_voc:
            return all_boxes_r
        else:
            return predictions

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = instances.shape[0]
    if num_instance == 0:
        return []
    results = []
    for k in range(num_instance):
        # print(instances[k])
        result = {
            "image_id": img_id,
            "category_id": int(instances[k,0] - 1),
            "bbox": instances[k,2:7].tolist(),
            "score": float(instances[k,1]),
        }
        results.append(result)
    return results

def evaluation(num_imgs, img_dir, image_ext, test_annotation_path, draw_imgs):
    print('Evaluating')
    save_dir = os.path.join('../output/predictions/piglet')
    tools.mkdir(save_dir)
    cfgs.NMS_IOU_THRESHOLD = args.nms_thresh

    if args.use_voc:
        test_annotation_path = os.path.join(test_annotation_path, "xmls")
    else:
        test_annotation_path = os.path.join(test_annotation_path, "test.json")

    retinanet = build_whole_network_r3det.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                           is_training=False)

    predictions = eval_with_plac(img_dir=img_dir, det_net=retinanet,
                                 num_imgs=num_imgs, image_ext=image_ext, save_dir=save_dir, draw_imgs=draw_imgs)

    dataset_type = "voc" if args.use_voc else "coco"
    with open("{} prediction.pkl".format(dataset_type),  "wb") as f:
        pickle.dump(predictions, f)


    # print(10 * "**")
    # print('rotation eval:')

    # if args.use_voc:
    #     imgs = os.listdir(img_dir)
    #     real_test_imgname_list = [i.split(image_ext)[0] for i in imgs]
    #     voc_eval_r.voc_evaluate_detections(all_boxes=predictions,
    #                                        test_imgid_list=real_test_imgname_list,
    #                                        test_annotation_path=test_annotation_path)

    # else:
    #     coco_gt = COCO(test_annotation_path)
    #     coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    #     coco_eval = rotated_coco_eval.evaluate_predictions_on_coco(coco_gt, coco_results)
    #     with open(os.path.join(save_dir, "{}.pkl".format(int(args.nms_thresh*100))), "wb") as f:
    #         pickle.dump(coco_eval.eval, f)



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='../data/io/PIGLET/test/images', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.jpg', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='../data/io/PIGLET/test', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)
    parser.add_argument('--draw_imgs', '-s', default=False,
                        action='store_true')
    parser.add_argument('--model_weight', dest='model_weight',
                        help='model_weight',
                        default='../output/trained_weights/RetinaNet_PIGLET_20200603/piglet_140000_model.ckpt', type=str)
    parser.add_argument('--use_voc', dest='use_voc', default=False, action='store_true')
    parser.add_argument('--vis_score', type=float, dest='vis_score', default=0.3,
                        help='visualize score')
    parser.add_argument('--nms_thresh', type=float, dest='nms_thresh', default=0.2,
                        help='nms threshold')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    evaluation(np.inf, args.img_dir, args.image_ext, args.test_annotation_path, args.draw_imgs)

