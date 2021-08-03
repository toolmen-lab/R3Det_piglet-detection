import json
import numpy as np
import itertools
from tabulate import tabulate
import math
import matplotlib.pyplot as plt
#import pandas as pd
import cv2
import sys
sys.path.append("../../../libs")
from box_utils import generate_anchors
from configs import cfgs

def calculate_instance_histogram(dataset):
    with open('classes.txt') as f:
        class_names = f.read().split(',')
    with open(dataset) as f:
        gt = json.load(f)
        
    num_classes = len(gt['categories'])
    hist_bins = np.arange(num_classes + 1)
    classes = [ann["category_id"] for ann in gt['annotations'] if not ann.get("iscrowd", 0)]
    histogram = np.histogram(classes, bins=hist_bins)[0]
    N_COLS = min(6, len(class_names) * 2)

    data = list(
        itertools.chain(*[[class_names[i], int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)
    
def calculate_horizontal_boxes_histogram(dataset, input_imgsize):
    plt.style.use('seaborn')
    with open(dataset) as f:
        gt = json.load(f)
    w_h = {"width": [], "height": []}
    for ann in gt['annotations']:
        cx, cy, w, h, angle = ann['bbox']
        theta = angle / 180.0 * math.pi
        c = math.cos(-theta)
        s = math.sin(-theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        rotated_rect = [(s * yy + c * xx + cx, c * yy - s * xx + cy) for (xx, yy) in rect]
        rotated_rect = [item for sub in rotated_rect for item in sub]
        xmin = min(rotated_rect[0::2])
        xmax = max(rotated_rect[0::2])
        ymax = max(rotated_rect[1::2])
        ymin = min(rotated_rect[1::2])
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        image_id = ann['image_id']
        img_width = gt['images'][image_id]['width']
        img_height = gt['images'][image_id]['height']
        w_h["width"].append(w / img_width * input_imgsize)
        w_h["height"].append(h / img_height * input_imgsize)
        
        
#    df = pd.DataFrame(data = w_h)
    ratio = [w_h['width'][i] / w_h['height'][i] for i in range(len(w_h['width']))]
    size = [w_h['width'][i] * w_h['height'][i] for i in range(len(w_h['height']))]
    sort_ratio = sorted(ratio)
    sort_size = sorted(size)    
    
    ratio_x = [1/i for i in range(int(np.ceil(1/sort_ratio[0])), 1 ,-1)] + [i for i in range(1,int(np.ceil(sort_ratio[-1]))+1)]
    ratio_ticks = [r'$\frac{{1}}{{{}}}$'.format(i) for i in range(int(np.ceil(1/sort_ratio[0])), 1 ,-1)] \
                + [str(i) for i in range(1,int(np.ceil(sort_ratio[-1])) + 1)]
    ratio_x = np.log10(ratio_x)
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize = (12.0,8.0))
    ratio_n, ratio_bins, patches = ax1.hist(np.log10(sort_ratio), bins = 101, alpha = 0.5)
#    ax1.plot(bins[:-1] + patches[0]._width / 2, n, color = (0,0,139/255), alpha = 0.5)
    ax1.set_xticks(ratio_x)
    ax1.set_xticklabels(ratio_ticks)
    ax1.set_xlabel("Anchor Ratio at logarithm base 10", fontsize = 8)
    ax1.set_title("Object horizontal boxes ratio", fontsize = 10)
    ax1.set_ylabel("Count")
    ax1.set_xlim(min(ratio_x) - 0.01, max(ratio_x) + 0.01)
    
    sort_size = np.log2(sort_size)
    size_n, size_bins, patches = ax2.hist(sort_size, bins = 101, alpha = 0.5)
#    ax2.plot(bins[:-1] + patches[0]._width / 2, n, color = (0,0,139/255), alpha = 0.5)
    ax2.set_title("Object horizontal boxes area (pixels)" , fontsize = 10)
    ax2.set_xlabel("Anchor Size", fontsize = 8)
    ax2.set_xticks([i*2 for i in range(int(np.floor(min(sort_size)/2)), int(np.ceil(max(sort_size)/2)) + 1)])
    ax2.set_xticklabels([2**i for i in range(int(np.floor(min(sort_size)/2)), int(np.ceil(max(sort_size)/2)) + 1)])
    
    fig.suptitle("Input image size {}".format(input_imgsize), fontsize = 16)
    plt.grid(True)
    
#    sns.jointplot(x = "width", y = "height", data = df, kind = "reg")
    plt.style.use('default')
    
    
    max_ratio_image = gt['images'][gt['annotations'][ratio.index(max(ratio))]['image_id']]['file_name']
    min_ratio_image = gt['images'][gt['annotations'][ratio.index(min(ratio))]['image_id']]['file_name']
    max_size_image = gt['images'][gt['annotations'][size.index(max(size))]['image_id']]['file_name']
    min_size_image = gt['images'][gt['annotations'][size.index(min(size))]['image_id']]['file_name']
    print("minimum area {} at {}".format(np.power(2, sort_size[0]), min_size_image))
    print("maximum area {} at {}".format(np.power(2, sort_size[-1]), max_size_image))
    print("minimum ratio {} at {}".format(sort_ratio[0], min_ratio_image))
    print("maximum ratio {} at {}".format(sort_ratio[-1], max_ratio_image))
    
    return (ratio, ratio_n, ratio_bins), (size, size_n, size_bins)
    
def calculate_iou_histogram(dataset):
    plt.style.use('seaborn')
    with open(dataset) as f:
        gt = json.load(f)
    h_boxes, r_boxes = [], [] 
    index = 0 
    for image_id in range(len(gt["images"])):
        h_box, r_box = [], []
        for i in range(index, len(gt['annotations'])):
            ann = gt['annotations'][i]
            if ann["image_id"] != image_id:
                index = i
                break
            r_box.append(ann['bbox'])
            cx, cy, w, h, angle = ann['bbox']
            theta = angle / 180.0 * math.pi
            c = math.cos(-theta)
            s = math.sin(-theta)
            rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
            rotated_rect = [(s * yy + c * xx + cx, c * yy - s * xx + cy) for (xx, yy) in rect]
            rotated_rect = [item for sub in rotated_rect for item in sub]
            xmin = min(rotated_rect[0::2])
            xmax = max(rotated_rect[0::2])
            ymax = max(rotated_rect[1::2])
            ymin = min(rotated_rect[1::2])
            h_box.append([xmin, ymin, xmax, ymax])
        r_boxes.append(np.array(r_box))
        h_boxes.append(np.array(h_box))
    
    h_ious, r_ious = [], []
    for h_box in h_boxes:
        h_ious.append(iou_calculate(h_box, h_box))
    for r_box in r_boxes:
        r_ious.append(iou_rotate_calculate(r_box, r_box))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize = (12.0,8.0))
    h_n, h_bins, _ = ax1.hist(np.hstack(tuple(h_iou.reshape(-1) for h_iou in h_ious)), bins = 32, range = (0.1, 0.9), alpha = 0.5)
    ax1.set_title("Horizontal IoU histogram")
    ax1.set_xticks(np.arange(0.1, 0.95, 0.05))
    ax1.set_ylabel("Count")
    ax1.set_xlabel("IoU")
    ax1.set_xlim([0.1, 0.9])
    r_n, r_bins, _ = ax2.hist(np.hstack(tuple(r_iou.reshape(-1) for r_iou in r_ious)), bins = 32, range = (0.1, 0.9), alpha = 0.5)
    ax2.set_title("Rotated Bounding box IoU histogram")
    ax2.set_xticks(np.arange(0.1, 0.95, 0.05))
    ax2.set_xlabel("IoU")
    ax2.set_xlim([0.1, 0.9])
    plt.style.use('default')
            
    return h_ious, r_ious

    
def iou_calculate(boxes1, boxes2):   
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        for j, box2 in enumerate(boxes2):
            ixmin = np.maximum(box1[0], box2[0])
            iymin = np.maximum(box1[1], box2[1])
            ixmax = np.minimum(box1[2], box2[2])
            iymax = np.minimum(box1[3], box2[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            int_area = iw * ih
            inter = np.around(int_area * 1.0 / (area1[i] + area2[j] - int_area), decimals=5)
            temp_ious.append(inter)
        ious.append(temp_ious)        
    return np.array(ious, dtype=np.float32)
    
def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = np.around(int_area * 1.0 / (area1[i] + area2[j] - int_area), decimals=5)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)

def make_anchor(input_imgsize = 512):
    anchor_list = []
    resolution = [(input_imgsize / (2 ** i) , input_imgsize / (2 ** i)) for i in range(3,7)]
    
    for i, r in enumerate(resolution):
        featuremap_height, featuremap_width = r
        stride = cfgs.ANCHOR_STRIDE[i]
        tmp_anchors = generate_anchors.generate_anchors_pre(featuremap_height, featuremap_width, stride,
                                           np.array(cfgs.ANCHOR_SCALES) * stride, cfgs.ANCHOR_RATIOS, 4.0)
        anchor_list.append(tmp_anchors)
    anchors = np.concatenate(anchor_list, axis=0)
    return anchors

def anchor_target_layer(gt_boxes_h, anchors):
    anchor_states = np.zeros((anchors.shape[0],))
    labels = np.zeros((anchors.shape[0], cfgs.CLASS_NUM))
    overlaps = iou_calculate(np.ascontiguousarray(anchors, dtype=np.float),
                                 np.ascontiguousarray(gt_boxes_h, dtype=np.float))

    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    target_boxes = gt_boxes_h[argmax_overlaps_inds]

    positive_indices = max_overlaps >= cfgs.IOU_POSITIVE_THRESHOLD
    ignore_indices = (max_overlaps > cfgs.IOU_NEGATIVE_THRESHOLD) & ~positive_indices
    
    labels[positive_indices, target_boxes[positive_indices, -1].astype(int) - 1] = 1
    anchor_states[ignore_indices] = -1
    anchor_states[positive_indices] = 1

    return anchor_states

def calculate_positive_horizontal_anchors(input_imgsize = 512):
    anchors = make_anchor(input_imgsize = 512)
    with open('train/train.json') as f:
        gt = json.load(f)
    index = 0
    for image_id in range(len(gt["images"])):
        h_box = []
        for i in range(index, len(gt['annotations'])):
            ann = gt['annotations'][i]
            if ann["image_id"] != image_id:
                index = i
                break
            cx, cy, w, h, angle = ann['bbox']
            theta = angle / 180.0 * math.pi
            c = math.cos(-theta)
            s = math.sin(-theta)
            rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
            rotated_rect = [(s * yy + c * xx + cx, c * yy - s * xx + cy) for (xx, yy) in rect]
            rotated_rect = [item / 1000 * input_imgsize for sub in rotated_rect for item in sub]
            xmin = min(rotated_rect[0::2])
            xmax = max(rotated_rect[0::2])
            ymax = max(rotated_rect[1::2])
            ymin = min(rotated_rect[1::2])
            h_box.append([xmin, ymin, xmax, ymax, ann['category_id'] + 1])
        anchor_states = anchor_target_layer(np.array(h_box), anchors)


          
if __name__ == "__main__":
    dataset = 'train/train.json'
    input_imgsize = 512
#    calculate_instance_histogram(dataset)
#    ratio, size = calculate_horizontal_boxes_histogram(dataset, input_imgsize)
#    h_ious, r_ious = calculate_iou_histogram(dataset)
    calculate_positive_horizontal_anchors(input_imgsize = 512)