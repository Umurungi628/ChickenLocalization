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

# Function to choose random bounding boxes from annotations
def choose_random_bounding_boxes(annotations, count):
    bounding_boxes = [annotation['bbox'] for annotation in annotations if 'bbox' in annotation]
    if count > len(bounding_boxes):
        count = len(bounding_boxes)
    return random.sample(bounding_boxes, count)

def choose_random_contours(annotations, count):
    contours = [annotation["segmentation"] for annotation in annotations if "segmentation" in annotation]
    if count > len(contours):
        count = len(contours)
    return random.sample(contours, count)


# Function to place bounding boxes on empty stalls
def place_bounding_boxes_on_stalls(empty_stalls, crops, bounding_boxes, contours, repeat_range, label, starting_id):
    annotations = []
    ann_id = starting_id
    for stall_idx, stall in enumerate(empty_stalls):
        for _ in range(random.randint(*repeat_range)):
            crop = random.choice(crops)
            bbox = random.choice(bounding_boxes)
            contour = random.choice(contours)

            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                if crop.shape[0] <= stall.shape[0] and crop.shape[1] <= stall.shape[1]:
                    position = (random.randint(0, stall.shape[1] - crop.shape[1]),
                                random.randint(0, stall.shape[0] - crop.shape[0]))
                    for i in range(w):
                        for j in range(h):
                            if cv2.pointPolygonTest(contour, [j,i], True):
                                stall[position[1]+j, position[0]+i] = crop[j, i]
                    #stall[position[1]:position[1] + crop.shape[0], position[0]:position[0] + crop.shape[1]] = crop


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

# Function to transform dead chicken bounding boxes
def transform_dead_chicken_bounding_boxes(dead_chicken_images, annotations, repeat_range):
    transformed_bounding_boxes = []
    transformed_contours = []
    transformed_crops = []

    for image, annotation in zip(dead_chicken_images, annotations):
        for _ in range(random.randint(*repeat_range)):
            bbox = annotation['bbox']
            contours = annotation['segmentation']
            xc = np.array(contours, dtype=float).flatten()[::2]
            yc = np.array(contours, dtype=float).flatten()[1::2]
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = map(int, bbox)

                # Scale up the bounding box dimensions
                scale_factor = 1.3
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                center_x, center_y = x + w // 2, y + h // 2
                new_x, new_y = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)

                if new_y + new_h <= image.shape[0] and new_x + new_w <= image.shape[1]:
                    crop = image[new_y:new_y + new_h, new_x:new_x + new_w]
                    crop = cv2.resize(crop, (random.randint(100, 180), random.randint(100, 180)))

                    angle = random.randint(0, 360)
                    M = cv2.getRotationMatrix2D((crop.shape[1] // 2, crop.shape[0] // 2), angle, 1)
                    crop_rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))
                    angler = np.radians(angle)
                    xcr = (xc-w//2)*np.cos(angler) + (y-h//2)*np.sin(angler) + w//2
                    ycr = (xc-w//2)*np.sin(angler) + (y-h//2)*np.cos(angler) + w//2


                    perspective_matrix = cv2.getPerspectiveTransform(
                        np.float32([[0, 0], [crop.shape[1], 0], [0, crop.shape[0]], [crop.shape[1], crop.shape[0]]]),
                        np.float32([[random.uniform(0, 10), random.uniform(0, 10)],
                                    [crop.shape[1] + random.uniform(-10, 10), random.uniform(0, 10)],
                                    [random.uniform(0, 10), crop.shape[0] + random.uniform(-10, 10)],
                                    [crop.shape[1] + random.uniform(-10, 10), crop.shape[0] + random.uniform(-10, 10)]])
                    )
                    crop_transformed = cv2.warpPerspective(crop_rotated, perspective_matrix,
                                                           (crop.shape[1], crop.shape[0]))
                    coords = np.array([xcr,ycr]).transpose().reshape(1, len(xcr), 2)
                    coords_t = cv2.perspectiveTransform(coords, perspective_matrix)
                    coords_t = coords_t.transpose().reshape(2, len(xcr))
                    xct = coords_t[0,:]
                    yct = coords_t[1,:]

                    transformed_contours.append(np.array([xct, yct], dtype=int).transpose().reshape(1, 2 * len(xct)).tolist())
                    transformed_crops.append(crop_transformed)
                    transformed_bounding_boxes.append([new_x, new_y, new_w, new_h])
    return transformed_crops, transformed_bounding_boxes, transformed_contours

# Function to combine healthy and dead chickens with empty stalls
def combine_images_with_stalls(empty_stalls, healthy_anot, dead_anot, healthy_repeat_range, dead_repeat_range):
    healthy_crops = [healthy_images[i] for i in range(len(healthy_images))]
    healthy_bboxes = choose_random_bounding_boxes(healthy_anot, random.randint(*healthy_repeat_range))
    healthy_contours = choose_random_contours(healthy_anot, random.randint(*healthy_repeat_range))

    dead_crops, dead_bboxes, dead_contours = transform_dead_chicken_bounding_boxes(dead_images, dead_anot, dead_repeat_range)

    ann_id = 0
    final_images, healthy_annotations, ann_id = place_bounding_boxes_on_stalls(empty_stalls, healthy_crops, healthy_bboxes,
                                                                               healthy_contours, healthy_repeat_range,
                                                                               label=1, starting_id=ann_id)
    final_images, dead_annotations, ann_id = place_bounding_boxes_on_stalls(final_images, dead_crops, dead_contours, dead_contours,
                                                                            dead_repeat_range, label=2,
                                                                            starting_id=ann_id)

    annotations = healthy_annotations + dead_annotations
    return final_images, annotations

# Main script
if __name__ == "__main__":
    healthy_image_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/healthy_images'
    dead_image_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/dead_images'
    empty_stall_dir = '/Users/Admin/PycharmProjects/ChickenLocalization/empty_stalls'
    healthy_annotations_file = '/Users/Admin/PycharmProjects/ChickenLocalization/healthy_annotations/annotations/annotations.json'
    dead_annotations_file = '/Users/Admin/PycharmProjects/ChickenLocalization/dead_annotations/annotations/annotations.json'

    healthy_images = load_images(healthy_image_dir)
    dead_images = load_images(dead_image_dir)
    empty_stalls = load_images(empty_stall_dir)

    healthy_annotations = load_json_annotations(healthy_annotations_file)
    dead_annotations = load_json_annotations(dead_annotations_file)

    healthy_repeat_range = (30, 100)
    dead_repeat_range = (5, 10)

    final_images, annotations = combine_images_with_stalls(empty_stalls, healthy_annotations['annotations'],
                                                           dead_annotations['annotations'], healthy_repeat_range,
                                                           dead_repeat_range)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

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

    categories = [
        {"id": 1, "name": "healthy_chicken"},
        {"id": 2, "name": "dead_chicken"}
    ]

    output_annotations = {
        "images": image_metadata,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(output_annotations, f, indent=2)
