#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/18 22:26
# @Author   : WanDaoYi
# @FileName : yolo_video.py
# ============================================

from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from core import utils
from config import cfg


class YoloVideo(object):

    def __init__(self):
        # pb 模型路径
        self.pb_model_path = cfg.TEST.TEST_PB_MODEL_PATH
        # yolov3 网络 返回 3 个尺度节点
        self.return_elements = cfg.TEST.RETURN_ELEMENTS
        # class_name 字典
        self.class_name_dir = utils.read_class_names(cfg.COMMON.CLASS_FILE_PATH)
        # c 类 个数
        self.class_name_len = len(self.class_name_dir)
        # 输入尺度
        self.input_size = cfg.TEST.INPUT_SIZE
        # 视频文件路径
        self.video_path = cfg.TEST.VEDIO_PATH

        self.graph = tf.Graph()
        # 加载模型
        self.return_tensors = utils.read_pb_return_tensors(self.graph,
                                                           self.pb_model_path,
                                                           self.return_elements)
        self.sess = tf.Session(graph=self.graph)
        pass

    # 对视频流的处理
    def do_video(self):
        vid = cv2.VideoCapture(self.video_path)
        while True:
            # frame 是 RGB 颜色空间
            return_value, frame = vid.read()
            if return_value:
                # utils.image_preporcess 这个方法里面有 cv2.COLOR_BGR2RGB 方法
                # 如果自己写的模型，可以调一下，也许不需要这里
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                pass
            else:
                raise ValueError("No image!")
                pass

            frame_size = frame.shape[:2]
            # 之前训练的时候，转了一次颜色空间
            image_data = utils.image_preporcess(np.copy(frame), [self.input_size, self.input_size])
            image_data = image_data[np.newaxis, ...]

            pred_start_time = datetime.now()

            pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
                [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                feed_dict={self.return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.class_name_len)),
                                        np.reshape(pred_mbbox, (-1, 5 + self.class_name_len)),
                                        np.reshape(pred_lbbox, (-1, 5 + self.class_name_len))],
                                       axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame, bboxes)

            pred_end_time = datetime.now()
            print("一帧耗时: {}".format(pred_end_time - pred_start_time))

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            # 退出按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            pass
        pass


if __name__ == '__main__':

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = YoloVideo()
    demo.do_video()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
