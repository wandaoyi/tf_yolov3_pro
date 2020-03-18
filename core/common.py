#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/07 14:09
# @Author   : WanDaoYi
# @FileName : common.py
# ============================================


import tensorflow as tf


# 卷积
def convolutional(input_data, filters_shape, training_flag, name, down_sample=False, activate=True, bn=True):
    """
    :param input_data: 输入信息
    :param filters_shape: 卷积核的形状，如 (3, 3, 32, 64) 表示：3 x 3 大小的卷积核，输入32维，输出64维
    :param training_flag: 是否是在训练模式下返回输出
    :param name: 卷积的名称
    :param down_sample: 是否下采样，默认不下采样
    :param activate: 是否使用 ReLU 激活函数
    :param bn: 是否进行 BN 处理
    :return:
    """
    with tf.variable_scope(name):
        # 下采样
        if down_sample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        # 不下采样
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        # 卷积操作
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        # BN 处理
        if bn:
            conv = tf.layers.batch_normalization(conv,
                                                 beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 training=training_flag)
        # 添加 bias
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        # 激活函数处理
        if activate:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


# 残差模块
def residual_block(input_data, input_channel, filter_num1, filter_num2, training_flag, name):
    """
    :param input_data: 输入的 feature maps
    :param input_channel: 输入的 通道
    :param filter_num1: 卷积核数
    :param filter_num2: 卷积核数
    :param training_flag: 是否是在训练模式下返回输出
    :param name:
    :return:
    """

    # 用来做短路连接的 feature maps
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   training_flag=training_flag, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   training_flag=training_flag, name='conv2')
        # 残差值和短路值相加，得到残差模块
        residual_output = input_data + short_cut

    return residual_output


# concat 操作
def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        concat_output = tf.concat([current_output, previous_output], axis=-1)

    return concat_output


# 上采样
def up_sample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            up_sample_output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
        pass
    else:
        # 输入 filter 的数量
        filter_num = input_data.shape.as_list()[-1]
        up_sample_output = tf.layers.conv2d_transpose(input_data, filter_num, kernel_size=2,
                                                      padding='same', strides=(2, 2),
                                                      kernel_initializer=tf.random_normal_initializer())
        pass

    return up_sample_output
