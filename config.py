#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/07 14:09
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

import os
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg
cfg = __C

# common options 公共配置文件
__C.COMMON = edict()
# windows 获取文件绝对路径, 方便 windows 在黑窗口 运行项目
__C.COMMON.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# # 获取当前窗口的路径, 当用 Linux 的时候切用这个，不然会报错。(windows也可以用这个)
# __C.COMMON.BASE_PATH = os.getcwd()

# 相对路径 当前路径
__C.COMMON.RELATIVE_PATH = "./"

# class 文件路径
__C.COMMON.CLASS_FILE_PATH = os.path.join(__C.COMMON.BASE_PATH, "infos/classes/voc_class.txt")
# anchor 文件路径
__C.COMMON.ANCHOR_FILE_PATH = os.path.join(__C.COMMON.BASE_PATH, "infos/anchors/coco_anchors.txt")

# iou 损失的 阈值
__C.COMMON.IOU_LOSS_THRESH = 0.5

# 超参
__C.COMMON.ALPHA = 1.0
__C.COMMON.GAMMA = 2.0

# 每个尺度最多允许有 几个 bounding boxes
__C.COMMON.MAX_BBOX_PER_SCALE = 150
# 衰减率的 移动平均值，用来控制模型的更新速度
# decay设置为接近1的值比较合理，
# 通常为：0.999,0.9999等，decay越大模型越稳定，
# 因为decay越大，参数更新的速度就越慢，趋于稳定
__C.COMMON.MOVING_AVE_DECAY = 0.9995

# 图像路径
__C.COMMON.IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/images")
# xml 路径
__C.COMMON.ANNOTATION_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/annotations")

# 数据划分比例
__C.COMMON.TRAIN_PERCENT = 0.7
__C.COMMON.VAL_PERCENT = 0.2
__C.COMMON.TEST_PERCENT = 0.1

# 图像后缀名
__C.COMMON.IMAGE_EXTENSION = ".jpg"

# YOLO options
__C.YOLO = edict()
# YOLOV3 的 3 个尺度
__C.YOLO.STRIDES = [8, 16, 32]
# YOLOV3 上采样的方法
__C.YOLO.UP_SAMPLE_METHOD = "resize"
# YOLOV3 每个尺度包含 3 个 anchors
__C.YOLO.ANCHOR_PER_SCALE = 3

# Train options
__C.TRAIN = edict()
# 训练集数据
__C.TRAIN.TRAIN_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/dataset/voc_train.txt")
__C.TRAIN.VAL_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/dataset/voc_val.txt")
# 训练集 input size
__C.TRAIN.INPUT_SIZE_LIST = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.TRAIN_BATCH_SIZE = 1
__C.TRAIN.VAL_BATCH_SIZE = 2
# 学习率的范围
__C.TRAIN.LEARNING_RATE_INIT = 1e-3
__C.TRAIN.LEARNING_RATE_END = 1e-6
# 第一阶段的训练 epoch
__C.TRAIN.FIRST_STAGE_EPOCHS = 16
# 第二阶段的训练 epoch 用于表述，如果是预训练的话，第一阶段训练会冻结参数
__C.TRAIN.SECOND_STAGE_EPOCHS = 32
# 预热训练，即在预热之前，learning_rate 学习率简单的 人为缩小，即 前面 [: 2] 个 epochs
# 预热之后，则 learning_rate 随着训练次数  人为在缩小，
# 即 [2: FIRST_STAGE_EPOCHS + SECOND_STAGE_EPOCHS] 个 epochs
__C.TRAIN.WARM_UP_EPOCHS = 2

# 初始化模型
__C.TRAIN.INITIAL_WEIGHT = os.path.join(__C.COMMON.RELATIVE_PATH, "checkpoint/val_loss=4.4647.ckpt-5")
# 训练日志
__C.TRAIN.TRAIN_LOG = os.path.join(__C.COMMON.RELATIVE_PATH, "log/train_log")
# 验证日志
__C.TRAIN.VAL_LOG = os.path.join(__C.COMMON.RELATIVE_PATH, "log/val_log")

# FREEZE MODEL
__C.FREEZE = edict()
# ckpt 模型文件夹
__C.FREEZE.CKPT_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "checkpoint/val_loss=4.4647.ckpt-5")
# pb 模型文件夹
__C.FREEZE.PB_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "model_info/val_loss=4.4647.pb")
# YOLOV3 节点输出
__C.FREEZE.YOLO_OUTPUT_NODE_NAME = ["input/input_data",
                                    "pred_sbbox/concat_2",
                                    "pred_mbbox/concat_2",
                                    "pred_lbbox/concat_2"
                                    ]


# TEST options
__C.TEST = edict()
# 测试数据集
__C.TEST.TEST_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/dataset/voc_test.txt")
# 测试 .pb 模型 文件路径 yolov3_model
__C.TEST.TEST_PB_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "model_info/val_loss=4.4647.pb")
# test 输入尺度
__C.TEST.INPUT_SIZE = 544
# 输出 图像 文件夹
__C.TEST.OUTPUT_IMAGE_FILE = os.path.join(__C.COMMON.RELATIVE_PATH, "output/test_image")
# 输出 预测框信息 文件夹
__C.TEST.OUTPUT_BOX_INFO_FILE = os.path.join(__C.COMMON.RELATIVE_PATH, "output/test_box_info")
# 是否对预测打框后的图像进行保存，默认保存 True
__C.TEST.SAVE_BOXES_IMAGE_FLAG = True

__C.TEST.RETURN_ELEMENTS = ["input/input_data:0",
                            "pred_sbbox/concat_2:0",
                            "pred_mbbox/concat_2:0",
                            "pred_lbbox/concat_2:0"
                            ]

__C.TEST.VEDIO_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/video/test_video.mp4")
