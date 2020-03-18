# [tf_yolov3_pro](https://github.com/wandaoyi/tf_yolov3_pro)
tensorflow 版本的 yolov3 目标检测项目 2020-03-18

- [论文地址](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [我的 CSDN 博客](https://blog.csdn.net/qq_38299170) 
- 环境依赖(其实版本要求并不严格，你的版本要是能跑起来，那也是OK的)：
```bashrc
pip install easydict
pip install numpy==1.16
conda install tensorflow-gpu==1.13.1
pip install tqdm
pip install opencv-python
```
- 对应数据来说，用户可以用自己的数据来跑，当然，也可以到网上去下载开源数据
- 下面是开源数据的链接：
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
- 将数据放到指定的文件目录下(config.py 文件):
```bashrc
# 图像路径
__C.COMMON.IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/images")
# xml 路径
__C.COMMON.ANNOTATION_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "data/annotations")
```
- 其实，做好依赖，拿到数据，就仔细看看 config.py 文件，里面全是配置。配好路径或一些超参，基本上，后面就是一键运行就 OK 了。
- 对 config.py 进行配置设置。

## 数据生成
训练模型之前，我们需要做数据，即是将数据做成我们网络所需要的样子
- 按上面的提示，将原始数据放到指定路径下，或将 config.py 里面的路径指向你的原始数据路径。还有，就是就是目标数据路径。然后，一键运行 prepare.py 文件，就可以生成 .txt 的目标数据。

## 训练模型
- 上面 config.py 调好了，而且数据也已经生成了，那，是驴是马，跑起来再说。还是一键运行 yolo_train.py。
- 在训练的过程中，最好 batch_size 不要太小，不然，loss 不好收敛。比方说，你 batch_size = 1 和 batch_size = 8 效果是不一样的。
- 在训练中，可以根据 loss 和 日志 进行人为的选择模型。

## 模型冻结
- 将上面训练得到的 .ckpt 模型文件，冻结成 .pb 文件。一键运行 model_freeze.py 文件
- 冻结模型，会对模型一定程度上的压缩，而且精度几乎不损。

## 图像预测
- 一键运行 yolo_test.py 文件(可以运行 prepare.py 来生成自己想要的数据，当然，前提是配置 config.py 文件)

## 视频预测
- 一键运行 yolo_video.py 文件

## 本项目的优点
- 就是方便，很多东西，我已经做成傻瓜式一键操作的方式。里面的路径，如果不喜欢用相对路径的，可以在 config.py 里面选择 绝对路径
- 本人和唠叨，里面的代码，基本都做了注解，就怕有人不理解，不懂，我只是希望能给予不同的你，一点点帮助。

## 本项目的缺点
- 没做 mAP
- 没做多线程(这些，以后有机会，会在博客中再详解)
