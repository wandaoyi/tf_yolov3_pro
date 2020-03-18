#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/07 14:10
# @Author   : WanDaoYi
# @FileName : yolov3.py
# ============================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from config import cfg


class YoloV3(object):

    def __init__(self, input_data, training_flag):

        # 超参
        self.alpha = cfg.COMMON.ALPHA
        self.gamma = cfg.COMMON.GAMMA

        # 是否是在训练模式下返回输出 True 为返回
        self.training_flag = training_flag
        # 获取 classes
        self.classes = utils.read_class_names(cfg.COMMON.CLASS_FILE_PATH)
        # 获取 classes 的种类数
        self.num_class = len(self.classes)
        # 获取 YOLOV3 的 3 个尺度
        self.strides = np.array(cfg.YOLO.STRIDES)
        # 获取 anchors
        self.anchors = utils.get_anchors(cfg.COMMON.ANCHOR_FILE_PATH)
        # 每个尺度有 3 个 anchors
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # iou 损失的 阈值
        self.iou_loss_thresh = cfg.COMMON.IOU_LOSS_THRESH
        # 上采样的方式
        self.up_sample_method = cfg.YOLO.UP_SAMPLE_METHOD

        try:
            # 获取 yolov3 网络 小、中、大 三个尺度的 feature maps
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.structure_network(input_data)
            pass
        except:
            raise NotImplementedError("Can not structure yolov3 network!")
            pass

        # anchors list [[116.,  90.], [156., 198.], [373., 326.]],
        # 小尺度的 feature maps 使用 大尺度的 anchors 值，用于检测大尺度目标
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.pred_conv_bbox(self.conv_lbbox, self.anchors[2], self.strides[2])
            pass

        # anchors list [[30.,  61.], [62., 45.], [59., 119.]], 用于检测中等尺度目标
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.pred_conv_bbox(self.conv_mbbox, self.anchors[1], self.strides[1])
            pass

        # anchors list [[10.,  13.], [16., 30.], [33., 23.]],
        # 大尺度的 feature maps 使用 相对小点尺度的 anchors 值，用于检测小尺度目标
        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.pred_conv_bbox(self.conv_sbbox, self.anchors[0], self.strides[0])
            pass
        pass

    # 构建网络
    def structure_network(self, input_data):
        # 调取 darknet53 网络，获取 3 个尺度的返回值
        route_1, route_2, input_data = backbone.darknet53(input_data, self.training_flag)

        # conv set 操作 conv: 1 x 1 -> 3 x 3 -> 1 x 1 -> 3 x 3 -> 1 x 1
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.training_flag, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.training_flag, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.training_flag, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.training_flag, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.training_flag, 'conv56')

        # scale one 小尺度的 feature maps: conv set -> conv 3 x 3 -> 1 x 1
        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024),
                                                self.training_flag, name='conv_lobj_branch')
        # 3 * (self.num_class + 5) 表示为: 每个尺度有 3 个 anchors，每个 anchor 有 5 + num_class 个值
        # 5 = 4 + 1, 4个 坐标值，1 个置信度，num_class 表示分类的数量
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                          training_flag=self.training_flag, name='conv_lbbox',
                                          activate=False, bn=False)

        # 进入另一个分支点: conv set -> 1 x 1 -> up_sampling -> concatenate
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.training_flag, 'conv57')
        input_data = common.up_sample(input_data, name='up_sample0', method=self.up_sample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        # conv set 操作 conv: 1 x 1 -> 3 x 3 -> 1 x 1 -> 3 x 3 -> 1 x 1
        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.training_flag, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.training_flag, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.training_flag, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.training_flag, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.training_flag, 'conv62')

        # scale two 中尺度的 feature maps: conv set -> conv 3 x 3 -> 1 x 1
        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),
                                                self.training_flag, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          training_flag=self.training_flag, name='conv_mbbox',
                                          activate=False, bn=False)

        # 进入另一个分支点: conv set -> 1 x 1 -> up_sampling -> concatenate
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.training_flag, 'conv63')
        input_data = common.up_sample(input_data, name='up_sample1', method=self.up_sample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        # conv set 操作 conv: 1 x 1 -> 3 x 3 -> 1 x 1 -> 3 x 3 -> 1 x 1
        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.training_flag, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.training_flag, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.training_flag, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.training_flag, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.training_flag, 'conv68')

        # scale three 大尺度的 feature maps: conv set -> conv 3 x 3 -> 1 x 1
        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256),
                                                self.training_flag, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          training_flag=self.training_flag, name='conv_sbbox',
                                          activate=False, bn=False)

        # 将 3 个尺度 小、中、大 的 feature maps 类似于 13 x 13 大小的那种 feature maps
        return conv_lbbox, conv_mbbox, conv_sbbox
        pass

    # 对 yolov3 网络 3 个尺度的 feature maps 进行预测
    def pred_conv_bbox(self, conv_bbox, anchors, strides):
        """
        :param conv_bbox: yolov3 network 返回的 feature maps
        :param anchors: anchors 例如: [[10., 13.], [16., 30.], [33., 23.]],
                        [10., 13.] 用来表示 anchor box 为 10 x 13 大小的先验值
        :param strides: 缩放步幅的尺度 例如: [8, 16, 32] 中的一个值，(原图 416 x 416)
                        如 32 表示使用 步幅为 32 的尺度进行操作，得到 13 x 13 大小 的feature maps，
                        相当于 缩放为 原图的 1/32 大小，另外的 8 和 16 的操作同理。
        :return:
        """
        # 获取 conv_bbox 的形状结构
        conv_bbox_shape = tf.shape(conv_bbox)
        # 获取 conv_bbox 的批量大小
        batch_size = conv_bbox_shape[0]
        # 获取 conv_bbox 的大小
        conv_bbox_size = conv_bbox_shape[1]
        # 获取每个尺度 anchors 的数量
        anchor_per_scale = len(anchors)

        # 将 conv_bbox 构建 目标张量，方便取值
        conv_obj = tf.reshape(conv_bbox, (batch_size, conv_bbox_size,
                                          conv_bbox_size, anchor_per_scale,
                                          5 + self.num_class))

        # 获取 中心点 坐标
        conv_raw_dxdy = conv_obj[:, :, :, :, 0:2]
        # 获取 width 和 high
        conv_raw_dwdh = conv_obj[:, :, :, :, 2:4]
        # 获取 置信度 即 前景或背景 的概率
        conv_raw_conf = conv_obj[:, :, :, :, 4:5]
        # 获取 c 类 对应的 概率值
        conv_raw_prob = conv_obj[:, :, :, :, 5:]

        # 张量操作, 构建一个 y 轴方向 (conv_bbox_size, conv_bbox_size) 大小的 张量,
        # 并填入对应的正数值，用来表示它的绝对位置
        y = tf.tile(tf.range(conv_bbox_size, dtype=tf.int32)[:, tf.newaxis], [1, conv_bbox_size])
        # 张量操作, 构建一个 x 轴方向 (conv_bbox_size, conv_bbox_size) 大小的 张量,
        # 并填入对应的正数值，用来表示它的绝对位置
        x = tf.tile(tf.range(conv_bbox_size, dtype=tf.int32)[tf.newaxis, :], [conv_bbox_size, 1])

        # 将 (conv_bbox_size, conv_bbox_size) 大小的 张量 根据 通道数 cancat 起来,
        # 得到 (conv_bbox_size, conv_bbox_size, 2) 大小的张量, 这样，就得到对应 feature maps 每个格子的 绝对位置的数值
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        # 张量操作: 构建成 (batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, 2) 结构
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        # 将数据转为浮点型
        xy_grid = tf.cast(xy_grid, tf.float32)

        # 获取 x、y 预测值 映射到 原图 的中心点 位置 坐标; (偏移量 + 左上角坐标值) * 缩放值
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides
        # 获取 w、h 预测值 映射到 原图 的 width 和 high
        # 论文中的公式为: b_w = p_w * e ^ (t_w); 然后再乘以 缩放度，则映射回原图
        # p_w 为 先验 w 的大小，即为 anchor box 中 w 的大小。
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * strides
        # 将 中心点 和 长 高 合并
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        # 计算置信度
        pred_conf = tf.sigmoid(conv_raw_conf)
        # 计算 c 类 概率
        pred_prob = tf.sigmoid(conv_raw_prob)

        # 返回  [batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, 4 + 1 + class_num] 的 feature map
        # 4 + 1 + class_num 代表为: pred_xywh + pred_conf + pred_prob
        # 靠近 anchors 的 pred_conf 值为 1，远离的则 pred_conf 值为 0
        # 靠近 anchors 的 pred_prob 值接近 1，远离的则 pred_prob 值接近 0
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
        pass

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        """
        :param label_sbbox: label 相对应的信息 包含 5 + classes
        :param label_mbbox:
        :param label_lbbox:
        :param true_sbbox: 为 batch_size image 对应 strides 尺度的 ground truth boxes
                           [batch_size, ground_truth_num, xywh]; ground_truth_num 为每张图里面打有几个框
        :param true_mbbox:
        :param true_lbbox:
        :return:
        """

        # 分别计算三个尺度的损失函数
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.layer_loss(self.conv_sbbox, self.pred_sbbox,
                                         label_sbbox, true_sbbox,
                                         stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.layer_loss(self.conv_mbbox, self.pred_mbbox,
                                         label_mbbox, true_mbbox,
                                         stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.layer_loss(self.conv_lbbox, self.pred_lbbox,
                                         label_lbbox, true_lbbox,
                                         stride=self.strides[2])

        # 对三个尺度的损失函数进行相加
        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

    def layer_loss(self, conv_bbox, pred_bbox, label_bbox, true_bbox, stride):
        """
        :param conv_bbox: yolov3 网络得到的其中一个尺度的输出 feature maps
        :param pred_bbox: 对 一个尺度输出的 feature maps 预测值
        :param label_bbox: ground truth 对应的信息
        :param true_bbox: ground truth 对应 anchor 尺度下的真实 box 值
        :param stride: 缩放尺度 stride = [8, 16, 32] 中的一个值
        :return:
        """

        conv_shape = tf.shape(conv_bbox)
        batch_size = conv_shape[0]
        conv_bbox_size = conv_shape[1]
        input_size = stride * conv_bbox_size
        conv_bbox = tf.reshape(conv_bbox, (batch_size, conv_bbox_size, conv_bbox_size,
                                           self.anchor_per_scale, 5 + self.num_class))

        conv_raw_conf = conv_bbox[:, :, :, :, 4:5]
        conv_raw_prob = conv_bbox[:, :, :, :, 5:]

        # [batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, 4 + 1 + class_num] 的 feature map
        pred_xywh = pred_bbox[:, :, :, :, 0:4]
        pred_conf = pred_bbox[:, :, :, :, 4:5]

        label_xywh = label_bbox[:, :, :, :, 0:4]
        respond_bbox = label_bbox[:, :, :, :, 4:5]
        label_prob = label_bbox[:, :, :, :, 5:]

        # 计算 预测框 与 label 框 的 GIOU
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        # 计算 giou 的损失函数，在这里 使用 1 < bbox_loss_scale < 2 为 giou_loss 的惩罚系数
        # 当 bbox 相对于整张图像较小时，这时预测的准确率相对于较大的图像要小，需要用较大的 loss 来
        # 对 目标 训练的准确率进行调整。因为当 loss 很小，而准确率不高的情况下，
        # 是很难通过降低 loss 来调高 准确率的。而如 loss 相对大些，则容易通过降低 loss 来调高准确率。
        # 这个 1 < bbox_loss_scale < 2 也是作者通过试验，测出来较好的值
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        # 在这里乘上一个置信度，因为背景是没有 giou_loss 的
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # 预测框 和 ground truth box 的 iou
        # [batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, ground_truth_box_num, xywh]
        # ground_truth_box_num: 表示一张图 打有 几个 框
        # 比如 pred_xywh 为 13 x 13 个格子，ground_truth_box_num 为 2。
        # 每个格子中的坐标 与 ground_truth_box_num 这两个框 的坐标 的 IOU 结果,
        # 这个 iou 用于 获取 下面 获取负样本 数
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                            true_bbox[:, np.newaxis, np.newaxis, np.newaxis, :, :])

        # tf.reduce_max(iou, axis=-1) 获取最后一个维度 最大的 iou 值;
        # expand_dims 可以用来增加一个维度，比如 [1, 2, 3] --> [[1], [2], [3]]
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # 获取 负样本 系数
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        # Focal loss: 为 交叉熵 的优化损失函数，减少 负样本 对损失函数对模型的影响
        # Focal_loss = -(respond_bbox - pred_conf) ^ gamma * log(pred_conf)
        # conf_focal 为 负样本 惩罚项系数
        conf_focal = self.alpha * tf.pow(tf.abs(respond_bbox - pred_conf), self.gamma)

        # respond_bbox 这里为正样本系数，因为它 的负样本 对应的值 为 0
        # respond_bgd 为负样本系数
        # 置信度损失函数
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        # c 类 概率损失函数
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        # 对各类损失函数累加再求均值
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    # bounding boxes giou
    def bbox_giou(self, boxes1, boxes2):

        # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 获取 框 的 左上角 和 右下角 的坐标值
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        # 计算 框 的面积
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算交集的 左上角 和 右下角 坐标
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 判断 两个 框 是否相交
        inter_section = tf.maximum(right_down - left_up, 0.0)
        # 计算 交集 的面积
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # 计算 并集 的面积
        union_area = boxes1_area + boxes2_area - inter_area
        # 计算 IOU
        iou = inter_area / union_area

        # 计算最小密封框 的 左上角 坐标
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        # 计算最小密封框 的 右下角 坐标
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        # 计算最小密封框 的 high 和 width
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        # 计算最小密封框 的 面积
        enclose_area = enclose[..., 0] * enclose[..., 1]
        # 计算 GIOU
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    # bounding boxes iou
    def bbox_iou(self, boxes1, boxes2):

        # 计算 框 的面积
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 计算交集的 左上角 和 右下角 坐标
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 判断 两个 框 是否相交
        inter_section = tf.maximum(right_down - left_up, 0.0)
        # 计算 交集 的面积
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # 计算 并集 的面积
        union_area = boxes1_area + boxes2_area - inter_area
        # 计算 IOU
        iou = 1.0 * inter_area / union_area

        return iou
