from ultralytics import YOLO

# Load the trained RT-DETR-L model
model = YOLO(r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\runs\detect\train10\weights\best.pt')

# Define the path to the dataset or image folder for prediction
source = r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset'

# Run prediction
results = model.predict(
    source=source,      # Path to images or dataset
    conf=0.3,           # Confidence threshold for detections
    device="cpu",       # Use CPU (Change to 0 for GPU if available)
    save=True,          # Save the results
    line_width=1,       # Line thickness for bounding boxes
    imgsz=640,          # Image size for better detection performance
    max_det=1000,       # Maximum detections per image
    exist_ok=True,      # Overwrite existing results
    show=False,         # Disable visualization (avoid OpenCV errors)
    project=r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\results',  # Output directory
    name='rtdetr-l'     # Experiment name
)

# Print detection results
for result in results:
    print(result)  # This will show detection details like bounding boxes and class labels
