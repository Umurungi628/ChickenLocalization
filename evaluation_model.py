from ultralytics import YOLO
import torch

# Load the model
model = YOLO("/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/yolov8x.pt")

# Evaluate the model on the validation dataset
# Replace "path/to/val_dataset" with the actual path to your validation dataset
results = model.val(data='/Users/User/PycharmProjects/ChickenLocalization/ChickenLocalization-main/dataset_test.yaml', imgsz=640)

# Display evaluation results
print("Evaluation Results:")
print(f"Mean Precision (mp): {results.box.mp:.4f}")
print(f"Mean Recall (mr): {results.box.mr:.4f}")
print(f"Mean AP@0.5 (map50): {results.box.map50:.4f}")
print(f"Mean AP@0.5:0.95 (map): {results.box.map:.4f}")

# For per-class results
num_classes = results.box.nc
for i in range(num_classes):
    class_precision, class_recall, class_ap50, class_map = results.box.class_result(i)
    print(f"\nClass {i} Results:")
    print(f"  Precision: {class_precision:.4f}")
    print(f"  Recall: {class_recall:.4f}")
    print(f"  AP@0.5: {class_ap50:.4f}")
    print(f"  AP@0.5:0.95: {class_map:.4f}")