import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load your trained YOLO model
model = YOLO(r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\yolov8_detect_result\detect\train5\weights\best.pt')

# Path to your images (not annotation files)
image_paths = [
    r"C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\test_images\3791_1.jpg"  # Change with actual image path

]

# Directory to save images with bounding boxes
output_dir = "output_anno"  # You can change this directory path
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Loop through each image and run inference
for img_path in image_paths:
    # Load original image (not the annotation .txt file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image: {img_path}")
        continue  # Skip this image and move to the next if it fails to load

    # Perform inference on the image
    results = model(img)  # Inference using the trained model

    # Get the result (bounding boxes, labels, etc.)
    result = results[0]  # The result for the first image in the batch

    # Check if any bounding boxes were detected
    if len(result.boxes) > 0:
        # Loop over detected boxes and draw the bounding boxes
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score

            # Set bounding box color based on class (Yellow for healthy chicken, Red for dead chicken)
            if model.names[cls] == 'healthy_chicken':
                color = (0, 255, 255)  # Yellow box (BGR format)
            elif model.names[cls] == 'dead_chicken':
                color = (0, 0, 255)  # Red box (BGR format)
            else:
                color = (0, 255, 0)  # Default Green if other classes are detected

            # Draw the bounding box and label
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the image with bounding boxes
        full_image_path = os.path.join(output_dir, os.path.basename(img_path).replace(".jpg", "_output.jpg"))
        cv2.imwrite(full_image_path, img)  # Save the image with bounding boxes
        print(f"Saved full image with detections as: {full_image_path}")

        # Display the full image with bounding boxes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    else:
        print(f"No detections found in {img_path}, skipping display.")
