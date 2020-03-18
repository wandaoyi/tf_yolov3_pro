#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/15 02:59
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================

import os
import random
from datetime import datetime
import xml.etree.ElementTree as ET
import core.utils as utils
from config import cfg


class Prepare(object):

    def __init__(self):
        # 图像路径
        self.image_path = cfg.COMMON.IMAGE_PATH
        # 图像的后缀名
        self.image_extension = cfg.COMMON.IMAGE_EXTENSION
        # xml 路径
        self.annotation_path = cfg.COMMON.ANNOTATION_PATH
        # 获取 c 类 字典型
        self.classes_dir = utils.read_class_names(cfg.COMMON.CLASS_FILE_PATH)
        self.classes_len = len(self.classes_dir)
        # 获取 c 类 list 型
        self.classes_list = [self.classes_dir[key] for key in range(self.classes_len)]

        # 数据的百分比
        self.test_percent = cfg.COMMON.TEST_PERCENT
        self.val_percent = cfg.COMMON.VAL_PERCENT

        # 各成分数据保存路径
        self.train_data_path = cfg.TRAIN.TRAIN_DATA_PATH
        self.val_data_path = cfg.TRAIN.VAL_DATA_PATH
        self.test_data_path = cfg.TEST.TEST_DATA_PATH

        pass

    def do_prepare(self):

        xml_file_list = os.listdir(self.annotation_path)
        xml_len = len(xml_file_list)
        # 根据百分比得到各成分 数据量
        n_test = int(xml_len * self.test_percent)
        n_val = int(xml_len * self.val_percent)
        n_train = xml_len - n_test - n_val

        if os.path.exists(self.train_data_path):
            os.remove(self.train_data_path)
            pass

        if os.path.exists(self.val_data_path):
            os.remove(self.val_data_path)
            pass

        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
            pass

        # 随机划分数据
        n_train_val = n_train + n_val
        train_val_list = random.sample(xml_file_list, n_train_val)
        train_list = random.sample(train_val_list, n_train)

        train_file = open(self.train_data_path, "w")
        val_file = open(self.val_data_path, "w")
        test_file = open(self.test_data_path, "w")

        for xml_name in xml_file_list:
            # 名字信息
            name_info = xml_name[: -4]
            # 图像名
            image_name = name_info + self.image_extension

            # 如果文件名在 训练 和 验证 文件划分中
            if xml_name in train_val_list:
                # 如果文件名在 训练数据划分中
                if xml_name in train_list:
                    self.convert_annotation(xml_name, image_name, train_file)
                    train_file.write('\n')
                    pass
                # 否则文件在 验证 文件
                else:
                    self.convert_annotation(xml_name, image_name, val_file)
                    val_file.write('\n')
                    pass
                pass
            # 否则文件名在 测试 文件
            else:
                self.convert_annotation(xml_name, image_name, test_file)
                test_file.write('\n')
                pass

        pass

    def convert_annotation(self, xml_name, image_name, file):
        xml_path = os.path.join(self.annotation_path, xml_name)
        image_path = os.path.join(self.image_path, image_name)
        file.write(image_path)

        # 打开 xml 文件
        xml_file = open(xml_path)
        # 将 xml 文件 转为树状结构
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter("object"):
            diff = obj.find("difficult").text
            cls = obj.find("name").text
            if cls not in self.classes_list or int(diff) == 1:
                continue

            cls_id = self.classes_list.index(cls)
            xml_box = obj.find("bndbox")
            b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text),
                 int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
            file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            pass
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = Prepare()
    demo.do_prepare()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
