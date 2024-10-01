from ultralytics import YOLO

# Load the pre-trained YOLOv8-nano model
model = YOLO("yolov8n.pt")

# Perform inference on an image (URL or local path)
results = model(r"C:\Users\Admin\PycharmProjects\ChickenLocalization\zidane.jpg")

# Since 'results' is a list, we need to call 'show()' on each result
for result in results:
    result.show()  # Display each result
