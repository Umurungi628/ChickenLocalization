import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO

# Load the combined annotations file
annotation_file = '/Users/Admin/PycharmProjects/ChickenLocalization/outputtt/annotations.json'
image_folder = '/Users/Admin/PycharmProjects/ChickenLocalization/outputtt'

# Initialize COCO API
coco = COCO(annotation_file)

# Get all image IDs
image_ids = coco.getImgIds()

# Function to display annotations for an image
def display_annotations(image_id):
    # Load image info
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_folder, image_info['file_name'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotations for the images
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # Display the image
    plt.imshow(image)
    plt.axis('off')

    # Draw annotations
    for annotation in annotations:
        bbox = annotation['bbox']
        x, y, width, height = bbox

        # Ignore invalid bounding boxes that cover the entire image
        if width >= image.shape[1] or height >= image.shape[0]:
            continue

        # Assign color based on category_id
        if annotation['category_id'] == 1:
            color = 'green'  # Healthy chicken
        elif annotation['category_id'] == 2:
            color = 'red'  # Dead chicken
        else:
            color = 'blue'  # Any other category (for debugging)

        # Draw the bounding box
        rect = plt.Rectangle((x, y), width, height, edgecolor=color, facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)

    plt.show()

# Display annotations for all images
for image_id in image_ids:
    display_annotations(image_id)
