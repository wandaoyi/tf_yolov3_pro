#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/07 14:10
# @Author   : WanDaoYi
# @FileName : dataset.py
# ============================================

import os
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from config import cfg


class Dataset(object):

    def __init__(self, train_flag=True):
        """
        :param train_flag: 是否是训练，默认训练
        """
        self.train_flag = train_flag

        # 训练数据
        if train_flag:
            self.data_file_path = cfg.TRAIN.TRAIN_DATA_PATH
            self.batch_size = cfg.TRAIN.TRAIN_BATCH_SIZE
            pass
        # 验证数据
        else:
            self.data_file_path = cfg.TRAIN.VAL_DATA_PATH
            self.batch_size = cfg.TRAIN.VAL_BATCH_SIZE
            pass

        self.train_input_size_list = cfg.TRAIN.INPUT_SIZE_LIST
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.COMMON.CLASS_FILE_PATH)
        self.class_num = len(self.classes)
        self.anchor_list = utils.get_anchors(cfg.COMMON.ANCHOR_FILE_PATH)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = cfg.COMMON.MAX_BBOX_PER_SCALE

        self.annotations = self.read_annotations()
        self.sample_num = len(self.annotations)
        self.batch_num = int(np.ceil(self.sample_num / self.batch_size))
        self.batch_count = 0
        pass

    # 迭代器
    def __iter__(self):
        return self

    # 使用迭代器 Dataset() 进行迭代，类似于 for 循环
    def __next__(self):
        with tf.device("/gpu:0"):
            # 从 train_input_size_list 中随机获取一个数值 作为 train_input_size
            self.train_input_size = np.random.choice(self.train_input_size_list)
            self.train_output_size = self.train_input_size // self.strides

            # 构建 输入图像 计算图
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            # 构建 3 个尺度预测图
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_size[0], self.train_output_size[0],
                                          self.anchor_per_scale, 5 + self.class_num))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_size[1], self.train_output_size[1],
                                          self.anchor_per_scale, 5 + self.class_num))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_size[2], self.train_output_size[2],
                                          self.anchor_per_scale, 5 + self.class_num))

            # 构建每个尺度上最多的 bounding boxes 的图
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            # 是否还在当前的 epoch
            if self.batch_count < self.batch_num:
                # 这个 while 用于一个 epoch 中的数据一条一条凑够一个 batch_size
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    # 如果最后一个 batch 不够数据，则 从头拿数据来凑
                    if index >= self.sample_num:
                        index -= self.sample_num
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image

                    # [batch_size, x_scope, y_scope, iou_flag, 5 + classes]
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox

                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes

                    num += 1

                self.batch_count += 1

                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            # 下一个 epoch
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration
            pass
        pass

    # 可以让 len(Dataset()) 返回 self.batch_num 的值
    def __len__(self):
        return self.batch_num

    # 获取 annotations.txt 文件信息
    def read_annotations(self):
        with open(self.data_file_path) as file:
            file_info = file.readlines()
            annotation = [line.strip() for line in file_info if len(line.strip().split()[1:]) != 0]
            np.random.shuffle(annotation)
            return annotation
        pass

    # 根据 annotation 信息 获取 image 和 bounding boxes
    def parse_annotation(self, annotation):
        # 将 "./data/images\Anime_180.jpg 388,532,588,729,0 917,154,1276,533,0"
        # 根据空格键切成 ['./data/images\\Anime_180.jpg', '388,532,588,729,0', '917,154,1276,533,0']
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        # 将 bboxes 做成 [[388, 532, 588, 729, 0], [917, 154, 1276, 533, 0]]
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        # 训练数据，进行仿射变换，让训练模型更好
        if self.train_flag:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
                                               np.copy(bboxes))
        return image, bboxes

    # 随机水平翻转
    def random_horizontal_flip(self, image, bboxes):

        if np.random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    # 随机裁剪
    def random_crop(self, image, bboxes):

        if np.random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - np.random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - np.random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + np.random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + np.random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    # 随机平移: 水平和竖直 方向移动变化，被移走后的位置，数值为0，显示为黑色
    def random_translate(self, image, bboxes):

        if np.random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            # 左上角 x、y 的数值，距离上边和下边的距离长度
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            # 右下角 距离 右边和下边 的距离长度
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            # 移动的偏移量，用来确保目标还在图像中
            tx = np.random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = np.random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            # 仿射变换核函数
            M = np.array([[1, 0, tx], [0, 1, ty]])
            # 仿射变换操作
            image = cv2.warpAffine(image, M, (w, h))

            # 对 bboxes 进行相应值 处理
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    # 对 ground truth boxes 进行预处理
    def preprocess_true_boxes(self, bboxes):

        # 构建 [train_output_sizes, train_output_sizes, anchor_per_scale, 5 + num_classes] 结构 的 label 图, 全部填 0 值
        label = [np.zeros((self.train_output_size[i], self.train_output_size[i], self.anchor_per_scale,
                           5 + self.class_num)) for i in range(3)]

        # 构建 xywh 的结构图 [max_bbox_per_scale, 4, 3]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        # bbox_count = [0, 0, 0]
        bbox_count = np.zeros((3,))

        # 将 bboxes ['388,532,588,729,0', '917,154,1276,533,0'] list 进行遍历
        for bbox in bboxes:
            # 获取单个 ground truth boxes 的坐标 [xmin, ymin, xmax, ymax]
            bbox_coor = bbox[:4]
            # 获取 ground truth 类别的下标
            bbox_class_ind = bbox[4]

            # 构建一个 c 类 大小的 one_hot list 并用 0 填充
            one_hot = np.zeros(self.class_num, dtype=np.float)
            # 构建真实的 label: 将上面获取到的 ground truth 类别的下标 定义 该类别的 one_hot 值为 1
            one_hot[bbox_class_ind] = 1.0
            # 构建 class_num 长度 的 list，并均匀分布，并填充 1.0 / class_num 值，
            # 让平滑看起来更舒服点，使用倒数值，是为了下面做平滑的时候，方便将总概率凑够 100%
            uniform_distribution = np.full(self.class_num, 1.0 / self.class_num)
            deta = 0.01
            # 对 one_hot 进行平滑处理, 模拟真实预测情况，前景概率是 90+%，但不是 100%; 而背景的概率，也不是 0%
            # 不过，这个平滑也可以不做的，没什么必要，因为 调用 np.argmax() 获取最大概率下标的结果是一样的。
            smooth_one_hot = one_hot * (1 - deta) + deta * uniform_distribution

            # 转换 [xmin, ymin, xmax, ymax] --> [x, y, w, h] bounding boxes 结构
            bbox_xywh = utils.bbox_dxdy_xywh(bbox_coor)

            # 归一化处理，将 ground truth boxes 缩放到 strides=[8, 16, 32] 对应的尺度
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            # 这里的 3 表示 yolo v3 中有 3 个预测尺度
            for i in range(3):
                # 构建 anchors 结构 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                # 将 ground truth box 的中心点 设置进去, 后面 + 0.5 是给一个偏置值
                # [[x + 0.5, y + 0.5, 0, 0], [x + 0.5, y + 0.5, 0, 0], [x + 0.5, y + 0.5, 0, 0]]
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # 将 anchors box 的 w, h 设置进去, 例如下面的小尺度 anchor 值
                # [[x + 0.5, y + 0.5, 10, 13], [x + 0.5, y + 0.5, 16, 30], [x + 0.5, y + 0.5, 33, 23]]
                anchors_xywh[:, 2:4] = self.anchor_list[i]

                # 计算 ground truth box 与 anchor boxes 的 IOU
                # [x, y, w, h] --> [xmin, ymin, xmax, ymax]
                ground_truth_scaled = utils.bbox_xywh_dxdy(bbox_xywh_scaled[i][np.newaxis, :])
                anchor_boxes = utils.bbox_xywh_dxdy(anchors_xywh)

                # 缩放 再偏移 中心点，之后再计算 IOU，这样，用来比较判断 是否是正样本
                # anchor_boxes 里面有 3 个不同尺度 box，所以结果为 3 个 iou 值的 list
                iou_scale = utils.bboxes_iou(ground_truth_scaled, anchor_boxes)
                iou.append(iou_scale)
                # 这里 iou_mask 是 3 个 bool 元素的 list
                iou_mask = iou_scale > 0.3

                # np.any 为 逻辑 or 的意思，只要有一个是 True，这为 True
                if np.any(iou_mask):
                    # 获取 中心点 x、y
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    # 在 output 大小的 feature maps 中，找到映射 缩放后的中心点 对应的格子，
                    # 赋值 bbox_xywh、conf、prob
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    # 进入这个 if，则证明 IOU > 0.3, 有交集，是 前景，所以置信度为 1.0
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_one_hot

                    # 获取 bbox 对应的 下标; bbox_count = [0, 0, 0]
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    # bbox_ind 表示 一张图 第几个 ground truth box
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh

                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                # 获取 IOU 值最大 所对应的下标
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                # 获取最佳 anchor
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                # 获取 最佳 anchor 对应的 中心点
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                # bbox_ind 表示 一张图 第几个 ground truth box
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh

                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
