import json
import os
from PIL import Image

# Path to COCO annotations (in JSON format) and the images
coco_annotation_file = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmented_annotations\augmented_annotations.json'
images_directory = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmented_images'
yolo_output_dir = 'yolo_annotations'

# Load COCO annotation JSON
with open(coco_annotation_file) as f:
    coco_data = json.load(f)

# Create output directory if it doesn't exist
if not os.path.exists(yolo_output_dir):
    os.makedirs(yolo_output_dir)

# Extract images and annotations from COCO data
images = {image['id']: image for image in coco_data['images']}
annotations = coco_data['annotations']
categories = {category['id']: category['name'] for category in coco_data['categories']}

# Function to convert COCO bbox to YOLO format
def coco_to_yolo_bbox(image_width, image_height, bbox):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width = width / image_width
    height = height / image_height
    return x_center, y_center, width, height

# Iterate over each annotation and convert to YOLO format
for annotation in annotations:
    image_id = annotation['image_id']
    image_info = images[image_id]
    image_path = os.path.join(images_directory, image_info['file_name'])

    # Get image dimensions (use PIL to open image)
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    # COCO bounding box format [x_min, y_min, width, height]
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    yolo_bbox = coco_to_yolo_bbox(image_width, image_height, bbox)

    # YOLO format: class_id x_center y_center width height
    class_id = category_id  # Map this if you need specific class labels

    # Create YOLO annotation string
    yolo_annotation = f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n"

    # Save to .txt file named after the image (but with .txt extension)
    txt_filename = os.path.splitext(image_info['file_name'])[0] + '.txt'
    txt_filepath = os.path.join(yolo_output_dir, txt_filename)

    with open(txt_filepath, 'a') as f:
        f.write(yolo_annotation)

print(f"Conversion completed! YOLO annotations saved to {yolo_output_dir}")
