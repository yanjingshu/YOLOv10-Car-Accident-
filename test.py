from ultralytics import YOLO

# Load the pre-trained YOLOv10n model
model = YOLO(r"runs\detect\train\weights\best.pt")
results = model("car accident.jpg")
results[0].show()

#yolo val model=jameslahm/yolov10{n/s/m/b/l/x} data=coco.yaml batch=256


#yolo detect train data=data/data.yaml model=yolov10n.pt epochs=30 batch=8 imgsz=640 device=0