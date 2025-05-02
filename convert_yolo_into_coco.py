import json
import os

# Paths to your directories
train_image_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\train\images'
valid_image_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\valid\images'
train_label_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\train\labels'
valid_label_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\valid\labels'

# Paths to the COCO annotations JSON files
train_coco_annotations_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset_coco\train\annotations\train_coco_annotations.json'
valid_coco_annotations_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset_coco\valid\annotations\valid_coco_annotations.json'

# Ensure that the directories exist
os.makedirs(os.path.dirname(train_coco_annotations_path), exist_ok=True)
os.makedirs(os.path.dirname(valid_coco_annotations_path), exist_ok=True)

# Initialize image and annotation id counters for train and valid separately
train_image_id = 1
valid_image_id = 1
train_annotation_id = 1
valid_annotation_id = 1

# Initialize train and valid COCO annotations data
train_coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "healthy_chicken", "supercategory": "chicken"},
        {"id": 1, "name": "dead_chicken", "supercategory": "chicken"}
    ]
}

valid_coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "healthy_chicken", "supercategory": "chicken"},
        {"id": 1, "name": "dead_chicken", "supercategory": "chicken"}
    ]
}

# Function to add images and annotations to COCO data
def add_images_and_annotations(image_dir, label_dir, image_set, coco_data, image_id_counter, annotation_id_counter):
    # Get the list of images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Loop through the image files
    for image_file in image_files:
        # Define the image file path
        image_path = os.path.join(image_dir, image_file)

        # Add image information
        coco_data['images'].append({
            'id': image_id_counter,
            'file_name': os.path.join(image_set, 'images', image_file),
            'width': 640,  # Update with the actual image width if necessary
            'height': 480  # Update with the actual image height if necessary
        })

        # Get the corresponding label file
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)

        # Check if label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as label_f:
                # Read the annotations in YOLO format
                for line in label_f.readlines():
                    # Format: class_id center_x center_y width height
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO coordinates to COCO format (x, y, width, height)
                    image_width = 640  # Update with actual image width
                    image_height = 480  # Update with actual image height

                    # COCO bbox format: [x, y, width, height]
                    x = (center_x - width / 2) * image_width
                    y = (center_y - height / 2) * image_height
                    bbox = [x, y, width * image_width, height * image_height]

                    # Add annotation for the object
                    coco_data['annotations'].append({
                        'id': annotation_id_counter,
                        'image_id': image_id_counter,
                        'category_id': class_id,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3],
                        'iscrowd': 0
                    })

                    annotation_id_counter += 1

        # Increment the image id counter
        image_id_counter += 1

    return coco_data, image_id_counter, annotation_id_counter


# Add images and annotations for the train and valid sets
train_coco_data, train_image_id, train_annotation_id = add_images_and_annotations(
    train_image_dir, train_label_dir, 'train', train_coco_data, train_image_id, train_annotation_id
)

valid_coco_data, valid_image_id, valid_annotation_id = add_images_and_annotations(
    valid_image_dir, valid_label_dir, 'valid', valid_coco_data, valid_image_id, valid_annotation_id
)

# Save the updated COCO annotations for both train and valid sets
with open(train_coco_annotations_path, 'w') as f:
    json.dump(train_coco_data, f, indent=4)

with open(valid_coco_annotations_path, 'w') as f:
    json.dump(valid_coco_data, f, indent=4)

print("COCO annotations for train and valid sets have been updated and saved successfully!")
