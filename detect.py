# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
@qq ：179958974
"""
from ultralytics import YOLOv10

# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt


model = YOLOv10(r'runs\train\exp5\weights\best.pt')



model.predict(source=r'yolov10-main\车祸\车祸 第一视角合集\xzg_948103.mp4', save=True)
