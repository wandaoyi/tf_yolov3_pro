#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/07 14:10
# @Author   : WanDaoYi
# @FileName : backbone.py
# ============================================


import core.common as common
import tensorflow as tf


def darknet53(input_data, training_flag):
    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32), 
                                          training_flag=training_flag, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                          training_flag=training_flag, name='conv1', down_sample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 64, 32, 64, 
                                               training_flag=training_flag, name='residual%d' % (i + 0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                          training_flag=training_flag, name='conv4', down_sample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128, 64, 128, training_flag=training_flag,
                                               name='residual%d' % (i + 1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          training_flag=training_flag, name='conv9', down_sample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, training_flag=training_flag,
                                               name='residual%d' % (i + 3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          training_flag=training_flag, name='conv26', down_sample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, training_flag=training_flag,
                                               name='residual%d' % (i + 11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          training_flag=training_flag, name='conv43', down_sample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, training_flag=training_flag,
                                               name='residual%d' % (i + 19))

        return route_1, route_2, input_data

