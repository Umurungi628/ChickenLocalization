from ultralytics import YOLO

# Load a YOLOv8 model (yolov8n.pt is a small model, consider using yolov8m.pt or yolov8l.pt for larger models)
model = YOLO('yolov8m.pt')

# Train the model
model.train(data='/Users/User/PycharmProjects/ChickenLocalization/ChickenLocalization-main/data.yaml', epochs=100, imgsz=640, batch=16)
