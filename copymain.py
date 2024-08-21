import os
import numpy as np
import cv2
import random
import glob
import json


# Function to load images
def load_images(image_dir):
    return [cv2.imread(file) for file in glob.glob(os.path.join(image_dir, '*.jpg'))]


# Function to load JSON annotations
def load_json_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)
    return annotations


# Function to enhance the crop (optional)
def enhance_crop(crop, alpha=1.5, beta=40):
    # Increase the contrast (alpha) and brightness (beta) of the image
    enhanced = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)

    # Apply sharpening kernel to make the crop edges more distinct
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


# Function to transform bounding boxes (applies to both healthy and dead chickens)
def transform_bounding_boxes(images, annotations, repeat_range, max_crop_size=(80, 80)):
    transformed_bounding_boxes = []
    transformed_crops = []

    for image, annotation in zip(images, annotations):
        for _ in range(random.randint(*repeat_range)):
            bbox = annotation['bbox']
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = map(int, bbox)

                # Limit the bounding box dimensions
                new_w, new_h = min(max_crop_size[0], w), min(max_crop_size[1], h)
                center_x, center_y = x + w // 2, y + h // 2
                new_x, new_y = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)

                if new_y + new_h <= image.shape[0] and new_x + new_w <= image.shape[1]:
                    crop = image[new_y:new_y + new_h, new_x:new_x + new_w]
                    crop = cv2.resize(crop, max_crop_size)  # Resize crop to max crop size

                    # Apply enhancements
                    crop = enhance_crop(crop)

                    # Apply random rotation
                    angle = random.randint(0, 360)
                    M = cv2.getRotationMatrix2D((crop.shape[1] // 2, crop.shape[0] // 2), angle, 1)
                    crop_rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))

                    # Apply perspective transform
                    perspective_matrix = cv2.getPerspectiveTransform(
                        np.float32([[0, 0], [crop.shape[1], 0], [0, crop.shape[0]], [crop.shape[1], crop.shape[0]]]),
                        np.float32([[random.uniform(0, 10), random.uniform(0, 10)],
                                    [crop.shape[1] + random.uniform(-10, 10), random.uniform(0, 10)],
                                    [random.uniform(0, 10), crop.shape[0] + random.uniform(-10, 10)],
                                    [crop.shape[1] + random.uniform(-10, 10), crop.shape[0] + random.uniform(-10, 10)]])
                    )
                    crop_transformed = cv2.warpPerspective(crop_rotated, perspective_matrix,
                                                           (crop.shape[1], crop.shape[0]))

                    transformed_crops.append(crop_transformed)
                    transformed_bounding_boxes.append([new_x, new_y, new_w, new_h])
    return transformed_crops, transformed_bounding_boxes


# Function to place bounding boxes on empty stalls
def place_bounding_boxes_on_stalls(empty_stalls, crops, repeat_range, label, starting_id):
    annotations = []
    ann_id = starting_id
    for stall_idx, stall in enumerate(empty_stalls):
        for _ in range(random.randint(*repeat_range)):
            crop = random.choice(crops)

            if crop.shape[0] <= stall.shape[0] and crop.shape[1] <= stall.shape[1]:
                position = (random.randint(0, stall.shape[1] - crop.shape[1]),
                            random.randint(0, stall.shape[0] - crop.shape[0]))

                # Place the crop on the stall without drawing any bounding box
                stall[position[1]:position[1] + crop.shape[0], position[0]:position[0] + crop.shape[1]] = crop

                annotations.append({
                    "id": ann_id,
                    "image_id": stall_idx,
                    "category_id": label,
                    "bbox": [position[0], position[1], crop.shape[1], crop.shape[0]],
                    "area": crop.shape[1] * crop.shape[0],
                    "iscrowd": 0
                })
                ann_id += 1
    return empty_stalls, annotations, ann_id


# Function to combine healthy and dead chickens with empty stalls
def combine_images_with_stalls(empty_stalls, healthy_anot, dead_anot, healthy_repeat_range, dead_repeat_range,
                               max_crop_size=(80, 80)):
    # Transform healthy chickens
    healthy_crops, _ = transform_bounding_boxes(healthy_images, healthy_anot, healthy_repeat_range, max_crop_size)

    # Transform dead chickens
    dead_crops, _ = transform_bounding_boxes(dead_images, dead_anot, dead_repeat_range, max_crop_size)

    # Initialize the annotation ID
    ann_id = 0

    # Place healthy chickens on empty stalls
    final_images, healthy_annotations, ann_id = place_bounding_boxes_on_stalls(empty_stalls, healthy_crops,
                                                                               healthy_repeat_range,
                                                                               label=1, starting_id=ann_id)

    # Place dead chickens on empty stalls
    final_images, dead_annotations, ann_id = place_bounding_boxes_on_stalls(final_images, dead_crops, dead_repeat_range,
                                                                            label=2, starting_id=ann_id)

    # Combine annotations
    annotations = healthy_annotations + dead_annotations
    return final_images, annotations


# Main script
if __name__ == "__main__":
    # Define paths to the directories and annotation files
    healthy_image_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/healthy_images'
    dead_image_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/dead_images'
    empty_stall_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/empty_stalls'
    healthy_annotations_file = '/Users/Admin/PycharmProjects/ChickenLocalization/healthy_annotations/annotations/annotations.json'
    dead_annotations_file = '/Users/Admin/PycharmProjects/ChickenLocalization/dead_annotations/annotations/annotations.json'

    # Load images and annotations
    healthy_images = load_images(healthy_image_dir)
    dead_images = load_images(dead_image_dir)
    empty_stalls = load_images(empty_stall_dir)
    healthy_annotations = load_json_annotations(healthy_annotations_file)
    dead_annotations = load_json_annotations(dead_annotations_file)

    # Define the repeat ranges
    healthy_repeat_range = (50, 150)  # Increased the range for more healthy chickens
    dead_repeat_range = (5, 10)

    # Combine images with stalls, using a uniform max crop size for both categories
    final_images, annotations = combine_images_with_stalls(empty_stalls, healthy_annotations['annotations'],
                                                           dead_annotations['annotations'], healthy_repeat_range,
                                                           dead_repeat_range, max_crop_size=(60, 60))  # Set equal size

    # Create the output directory
    output_dir = 'resultimage'
    os.makedirs(output_dir, exist_ok=True)

    # Save the final images and annotations
    image_metadata = []
    for idx, img in enumerate(final_images):
        image_filename = os.path.join(output_dir, f'final_image_{idx}.jpg')
        cv2.imwrite(image_filename, img)
        image_metadata.append({
            "id": idx,
            "file_name": f'final_image_{idx}.jpg',
            "width": img.shape[1],
            "height": img.shape[0]
        })

    # Define the categories
    categories = [
        {"id": 1, "name": "healthy_chicken"},
        {"id": 2, "name": "dead_chicken"}
    ]

    # Save annotations
    output_annotations = {
        "images": image_metadata,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(output_annotations, f, indent=2)
