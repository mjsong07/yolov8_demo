# yolov8_demo
yolov8视频demo

# 训练模型
yolo detect train \
  model=yolov8s.pt \
  data=code.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=code_project2 \
  name=exp2



 # 验证模型
 yolo detect predict \
  model=code_project2/exp2/weights/best.pt \
  source=dataset/images/val \
  conf=0.25 \
  save=True