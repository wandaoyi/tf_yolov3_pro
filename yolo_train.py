#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/03/12 14:40
# @Author   : WanDaoYi
# @FileName : yolo_train.py
# ============================================

import os
from datetime import datetime
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YoloV3
from config import cfg


class YoloTrain(object):

    def __init__(self):
        # 学习率的范围
        self.learning_rate_init = cfg.TRAIN.LEARNING_RATE_INIT
        self.learning_rate_end = cfg.TRAIN.LEARNING_RATE_END
        # 两个阶段的 epochs
        self.first_stage_epochs = cfg.TRAIN.FIRST_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        # 预热 epoch
        self.warm_up_epochs = cfg.TRAIN.WARM_UP_EPOCHS
        # 模型初始化
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        # 衰变率 移动平均值
        self.moving_ave_decay = cfg.COMMON.MOVING_AVE_DECAY
        # 每个尺度最多的 boxes 数
        self.max_bbox_per_scale = cfg.COMMON.MAX_BBOX_PER_SCALE
        # 训练日志路径
        self.train_log = cfg.TRAIN.TRAIN_LOG
        # 验证日志路径
        self.val_log = cfg.TRAIN.VAL_LOG

        # 获取训练数据集
        self.train_data = Dataset()
        # 获取一个 epoch 需要训练多少次
        self.batch_num = len(self.train_data)
        # 获取验证数据集
        self.val_data = Dataset(train_flag=False)

        self.conv_bbox = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']

        self.train_loss_info = "train loss: %.2f"
        self.ckpt_info = "./checkpoint/val_loss=%.4f.ckpt"
        self.loss_info = "=> Epoch: %2d, Time: %s, Train loss: %.2f, Val loss: %.2f, Saving %s ..."

        # 加载 Session
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        # 定义 feed_dict 图
        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.training_flag = tf.placeholder(dtype=tf.bool, name='training')

        # 定义 loss
        with tf.name_scope("define_loss"):
            self.model = YoloV3(self.input_data, self.training_flag)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss
            pass

        # 定义学习率 和 衰减变化
        with tf.name_scope('learn_rate'):

            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            # 预热训练的 batch 数
            warm_up_steps = tf.constant(self.warm_up_epochs * self.batch_num,
                                        dtype=tf.float64, name='warm_up_steps')

            # 总训练的 batch 次数
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.batch_num,
                                      dtype=tf.float64, name='train_steps')

            # tf.cond() 类似于 if else 语句, if pred true_fn else false_fn
            # tf.cos() 余弦函数
            # 通过这个算法，用来在训练过程中逐渐缩小 learning_rate
            self.learn_rate = tf.cond(
                pred=self.global_step < warm_up_steps,
                true_fn=lambda: self.global_step / warm_up_steps * self.learning_rate_init,
                false_fn=lambda: self.learning_rate_end + 0.5 * (self.learning_rate_init - self.learning_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warm_up_steps) / (train_steps - warm_up_steps) * np.pi))
            )

            # 类似于 self.global_step += 1; 但是，使用这个方法的话，必须按照 tf 的规矩，
            # 先构建 变量图，再初始化，最后 run() 的时候，才会执行
            global_step_update = tf.assign_add(self.global_step, 1.0)

            pass

        # 衰变率 移动平均值
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
            pass

        # 第一阶段训练
        with tf.name_scope("define_first_stage_train"):

            self.first_stage_trainable_var_list = []

            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in self.conv_bbox:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        # 第二阶段训练
        with tf.name_scope("define_second_stage_train"):

            second_stage_trainable_var_list = tf.trainable_variables()

            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            if os.path.exists(self.train_log):
                shutil.rmtree(self.train_log)

            os.mkdir(self.train_log)

            if os.path.exists(self.val_log):
                shutil.rmtree(self.val_log)
            os.mkdir(self.val_log)

            # 汇总日志
            self.write_op = tf.summary.merge_all()
            # 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
            # 同时，将Session计算图sess.graph加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
            self.train_writer = tf.summary.FileWriter(self.train_log, graph=self.sess.graph)
            # 验证集日志
            self.val_writer = tf.summary.FileWriter(self.val_log)
            pass

        pass

    def do_train(self):
        # 初始化参数
        self.sess.run(tf.global_variables_initializer())

        try:
            # 加载已有模型
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        # 如果模型不存在，则初始化模型
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            # 并重新定义 第一阶段训练 epoch 为 0
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            # 调取进度条
            pbar = tqdm(self.train_data)

            train_epoch_loss = []
            val_epoch_loss = []

            for train_data in pbar:
                _, train_summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbboxes: train_data[4],
                        self.true_mbboxes: train_data[5],
                        self.true_lbboxes: train_data[6],
                        self.training_flag: True,
                    })

                train_epoch_loss.append(train_step_loss)
                self.train_writer.add_summary(train_summary, global_step_val)
                pbar.set_description(self.train_loss_info % train_step_loss)

            for test_data in self.val_data:
                val_summary, val_step_loss = self.sess.run([self.write_op, self.loss],
                                                           feed_dict={
                                                               self.input_data: test_data[0],
                                                               self.label_sbbox: test_data[1],
                                                               self.label_mbbox: test_data[2],
                                                               self.label_lbbox: test_data[3],
                                                               self.true_sbboxes: test_data[4],
                                                               self.true_mbboxes: test_data[5],
                                                               self.true_lbboxes: test_data[6],
                                                               self.training_flag: False,
                                                           })

                val_epoch_loss.append(val_step_loss)
                self.val_writer.add_summary(val_summary, epoch + 1)

            train_epoch_loss = np.mean(train_epoch_loss)
            val_epoch_loss = np.mean(val_epoch_loss)
            ckpt_file = self.ckpt_info % val_epoch_loss
            now_time = datetime.now()
            print(self.loss_info % (epoch, now_time, train_epoch_loss, val_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = YoloTrain()
    demo.do_train()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
