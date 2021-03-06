# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math
# data augmentation
# [-45, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40 ,45]

# ------------------------------------------------
VERSION = 'RetinaNet_PIGLET_20200414'
NET_NAME = 'resnet_v1_50'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
SAVE_WEIGHTS_INTE = 200

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
EVALUATE_R_DIR = ROOT_PATH + '/output/eval'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 0  # allow 0~3
# FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0
USE_IOU_FACTOR = False

BATCH_SIZE = 20
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-5
DECAY_STEP = [SAVE_WEIGHTS_INTE*800, SAVE_WEIGHTS_INTE*1000, SAVE_WEIGHTS_INTE*1200]
MAX_ITERATION = SAVE_WEIGHTS_INTE*1000
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE * 100)

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'piglet'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 512
IMG_MAX_LENGTH = 512
CLASS_NUM = 4

IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = False
IMAGE_PYRAMID = False

# --------------------------------------------- Network_config
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4
USE_GN = False
NUM_SUBNET_CONV = 4
NUM_REFINE_STAGE = 1
USE_RELU = False
FPN_CHANNEL = 256

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 4.0), 2 ** (2.0 / 4.0)]
ANCHOR_RATIOS = [1, 1 / 2., 2, 1 / 3., 3., 5., 1 / 5.]
# ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
# USE_CENTER_OFFSET = True
METHOD = 'H'
USE_ANGLE_COND = True
ANGLE_RANGE = 90

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4
REFINE_IOU_POSITIVE_THRESHOLD = [0.6, 0.7]
REFINE_IOU_NEGATIVE_THRESHOLD = [0.5, 0.6]

NMS = True
NMS_IOU_THRESHOLD = 0.15
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.5
EVAL_THRESHOLD = 0.7
USE_07_METRIC = True

# # --------------------------------------------MASK config
# USE_SUPERVISED_MASK = False
# MASK_TYPE = 'r'  # r or h
# BINARY_MASK = False
# SIGMOID_ON_DOT = False
# MASK_ACT_FET = True  # weather use mask generate 256 channels to dot feat.
# GENERATE_MASK_LIST = ["P3", "P4", "P5", "P6", "P7"]
# ADDITION_LAYERS = [4, 4, 3, 2, 2]  # add 4 layer to generate P2_mask, 2 layer to generate P3_mask
# ENLAEGE_RF_LIST = ["P3", "P4", "P5", "P6", "P7"]
# SUPERVISED_MASK_LOSS_WEIGHT = 1.0
