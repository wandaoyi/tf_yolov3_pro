#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/17 00:01
# @Author   : WanDaoYi
# @FileName : yolo_test.py
# ============================================

from datetime import datetime
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
from core import utils
from config import cfg


class YoloTest(object):

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
        # 输出 图像 文件夹
        self.output_image_file = cfg.TEST.OUTPUT_IMAGE_FILE
        # 输出 预测框信息 文件夹
        self.output_box_info_file = cfg.TEST.OUTPUT_BOX_INFO_FILE
        # 是否保存预测打框图像，默认为 True 保存
        self.save_boxes_image_flag = cfg.TEST.SAVE_BOXES_IMAGE_FLAG

        self.graph = tf.Graph()
        # 加载模型
        self.return_tensors = utils.read_pb_return_tensors(self.graph,
                                                           self.pb_model_path,
                                                           self.return_elements)
        self.sess = tf.Session(graph=self.graph)
        pass

    def object_predict(self, data_line_list):

        for data_line in data_line_list:

            data_info_list = data_line.strip().split()
            image_path = data_info_list[0]
            image_path = image_path.replace("\\", "/")
            image_name = image_path.split("/")[-1]
            txt_name = image_name.split(".")[0] + ".txt"

            image_info = cv2.imread(image_path)

            pred_box = self.do_predict(image_info, image_name)
            print("predict result of {}".format(image_name))

            output_box_info_path = os.path.join(self.output_box_info_file, txt_name)

            # 保存预测图像信息
            with open(output_box_info_path, 'w') as f:
                for bbox in pred_box:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = self.class_name_dir[class_ind]
                    score = '%.4f' % score
                    x_min, y_min, x_max, y_max = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, x_min, y_min, x_max, y_max]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
                    pass
                pass

            pass
        pass

    # 预测操作
    def do_predict(self, image_info, image_name):
        image_shape = image_info.shape[: 2]
        # image_2_rgb = cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preporcess(np.copy(image_info),
                                            [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.class_name_len)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.class_name_len)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.class_name_len))],
                                   axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, image_shape, self.input_size, 0.3)
        pred_box = utils.nms(bboxes, 0.45, method='nms')

        if self.save_boxes_image_flag:
            new_image = utils.draw_bbox(image_info, pred_box)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            save_image_path = os.path.join(self.output_image_file, image_name)
            cv2.imwrite(save_image_path, new_image)
            pass

        # # 展示图像
        # new_image = utils.draw_bbox(image_2_rgb, pred_box)
        # cv2.imshow("predict_image", new_image)
        # new_image.show()
        # cv2.waitKey(0)

        return pred_box
        pass


if __name__ == '__main__':

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    # image data 的 路径 list
    data_path_list = utils.read_data_path(cfg.TEST.TEST_DATA_PATH)

    demo = YoloTest()

    if os.path.exists(demo.output_image_file):
        shutil.rmtree(demo.output_image_file)

    if os.path.exists(demo.output_box_info_file):
        shutil.rmtree(demo.output_box_info_file)

    os.mkdir(demo.output_image_file)
    os.mkdir(demo.output_box_info_file)

    demo.object_predict(data_path_list)

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))


