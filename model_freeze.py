#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/16 15:22
# @Author   : WanDaoYi
# @FileName : model_freeze.py
# ============================================

from datetime import datetime
from config import cfg
import tensorflow as tf
from core.yolov3 import YoloV3


class ModelFreeze(object):

    def __init__(self):
        pass

    # 调用 yolo 的节点，对模型进行冻结
    def yolo_model(self):

        output_node_names = cfg.FREEZE.YOLO_OUTPUT_NODE_NAME

        ckpt_model_path = cfg.FREEZE.CKPT_MODEL_PATH
        pb_model_path = cfg.FREEZE.PB_MODEL_PATH

        # 获取节点名
        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.float32, name='input_data')
        model = YoloV3(input_data, training_flag=False)

        self.freeze_model(ckpt_model_path, pb_model_path, output_node_names)
        pass

    # 模型冻结
    def freeze_model(self, ckpt_file, pb_file, output_node_names):

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver()

        saver.restore(sess, ckpt_file)
        converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                           input_graph_def=sess.graph.as_graph_def(),
                                                                           output_node_names=output_node_names)

        with tf.gfile.GFile(pb_file, "wb") as f:
            f.write(converted_graph_def.SerializeToString())
        pass


if __name__ == '__main__':
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = ModelFreeze()
    demo.yolo_model()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
