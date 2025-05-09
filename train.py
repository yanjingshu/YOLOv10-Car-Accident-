# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：train.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
@qq ：179958974
"""


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

if __name__ == '__main__':
    # model.load('yolov8n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLOv10(model=r'yolov10-main\ultralytics\cfg\models\v10\yolov10n.yaml')
    model.train(data=r'yolov10-main\data\data.yaml',
                imgsz=640,
                epochs=30,
                batch=8,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
