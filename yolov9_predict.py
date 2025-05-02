from ultralytics import YOLO

# Define the path to the model and dataset
model = YOLO(r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\yolov9_result_detect\detect\train\weights\best.pt')  # Update this with correct path to your best.pt file
sc = r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset'

# Run prediction
model.predict(
    source=sc,
    conf=0.3,
    device=0,  # Use 'cpu' if no GPU is available
    save=True,
    line_width=1,
    imgsz=640,  # Use a larger image size for better results
    max_det=1000,
    exist_ok=True,
    show=True,
    project=sc,
    name='yolov9'
)