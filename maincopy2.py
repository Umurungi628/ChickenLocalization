import os
import numpy as np
import cv2
import random
import glob
import json


# Function to load images
def load_images(image_dir):
    imnames = []
    [imnames.append(str(os.path.split(file)[1])) for file in glob.glob(os.path.join(image_dir, '*.jpg'))]
    return [cv2.imread(file) for file in glob.glob(os.path.join(image_dir, '*.jpg'))], imnames


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

def getimbboxes(imname, annotations):
    im_part = annotations['images']
    anot_part = annotations['annotations']
    im_id = []
    bboxes = []
    segmentations = []
    [im_id.append(j['id']) for j in im_part if j['file_name'] == imname]
    [bboxes.append(k['bbox']) for k in anot_part if k['image_id'] in im_id]
    [segmentations.append(m['segmentation']) for m in anot_part if m['image_id'] in im_id]
    return bboxes, segmentations


# Function to transform bounding boxes (applies to both healthy and dead chickens)
# Function to transform bounding boxes (focus on healthy chickens)
def transform_bounding_boxes(images, imnames, annotations, repeat_range, max_crop_size=(80, 80), im_type='healthy'):
    export_bounding_boxes = []
    export_segmentations = []
    export_crops = []

    for i, image in enumerate(images):
        name = imnames[i]
        bboxes, segmentations = getimbboxes(name, annotations)

        if im_type == 'healthy':
            repeats = random.sample(range(repeat_range[0], repeat_range[1]), 1)[0] if repeat_range[1] <= len(
                bboxes) else len(bboxes)
            rand_bbox_id = random.sample(range(len(bboxes)), repeats)
            for i in rand_bbox_id:
                bbox = bboxes[i]
                segmentation = segmentations[i]
                x, y, w, h = map(int, bbox)

                # Crop the image based on the bounding box
                crop = image[y:y + h, x:x + w]

                # Resize crop to the desired max crop size
                crop = cv2.resize(crop, max_crop_size)

                # Adjust the segmentation to match the resized crop
                xc = np.array(segmentation, dtype=float).flatten()[::2] - x  # x-coordinates of the segmentation
                yc = np.array(segmentation, dtype=float).flatten()[1::2] - y  # y-coordinates of the segmentation

                # Scale the segmentation coordinates to the resized crop
                x_scale = max_crop_size[0] / w
                y_scale = max_crop_size[1] / h
                xc_rescaled = xc * x_scale
                yc_rescaled = yc * y_scale

                # Combine x and y coordinates to create the updated segmentation
                updated_segmentation = np.array([xc_rescaled, yc_rescaled], dtype=int).transpose().reshape(1, 2 * len(
                    xc_rescaled)).tolist()

                # Store the crop and its bounding box and segmentation
                export_crops.append(crop)
                export_bounding_boxes.append([x, y, w, h])
                export_segmentations.append(updated_segmentation)

    return export_crops, export_bounding_boxes, export_segmentations


# Function to place healthy bounding boxes on empty stalls with updated segmentations
# Function to place bounding boxes on empty stalls without using segmentation initially
def place_bounding_boxes_on_stalls(empty_stalls, crops, bounding_boxes, segmentations, repeat_range, label, starting_id):
    annotations = []
    ann_id = starting_id
    for stall_idx, stall in enumerate(empty_stalls):
        repeats = random.sample(range(repeat_range[0], repeat_range[1]), 1)[0] if repeat_range[1] <= len(
            bounding_boxes) else len(bounding_boxes)
        random_choose = random.sample(range(len(bounding_boxes)), repeats)
        for i in random_choose:
            crop = crops[i]
            bbox = bounding_boxes[i]
            segmentation = segmentations[i]

            w, h = crop.shape[1], crop.shape[0]
            if crop.shape[0] <= stall.shape[0] and crop.shape[1] <= stall.shape[1]:
                position = (random.randint(0, stall.shape[1] - crop.shape[1]),
                            random.randint(0, stall.shape[0] - crop.shape[0]))

                segment = np.array(segmentation, int)[0]
                segment = np.reshape(segment,(round(segment.shape[0]/2), 2))
                segment = np.reshape(segment,(-1,1,2))
                #print(segment)
                #img1 = cv2.polylines(crop, [segment], True, (0,0,255), 2)
                for i in range(w):
                    for j in range(h):
                        if cv2.pointPolygonTest(segment, [j, i], True) >= 0:
                            stall[position[1] + j, position[0] + i] = crop[j, i]
                            #cv2.imshow('test', stall)
                            #cv2.waitKey(0)

                # Place the crop on the stall without drawing any bounding box
                #stall[position[1]:position[1] + crop.shape[0], position[0]:position[0] + crop.shape[1]] = crop
                #cv2.imshow('test', stall)
                #cv2.waitKey(0)

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
                               max_crop_size=(100, 100)):
    # Transform healthy chickens with segmentation
    healthy_crops, healthy_bboxes, healthy_segmentations = transform_bounding_boxes(healthy_images, himnames,
                                                                                    healthy_anot, healthy_repeat_range,
                                                                                    max_crop_size, im_type='healthy')

    # Transform dead chickens (unchanged)
    dead_crops, dead_bboxes, dead_segmentations = transform_bounding_boxes(dead_images, dimnames, dead_anot,
                                                                           dead_repeat_range, max_crop_size,
                                                                           im_type='dead')

    # Initialize the annotation ID
    ann_id = 0

    # Place healthy chickens on empty stalls
    final_images, healthy_annotations, ann_id = place_bounding_boxes_on_stalls(empty_stalls, healthy_crops,
                                                                               healthy_bboxes,
                                                                               healthy_segmentations,
                                                                               healthy_repeat_range,
                                                                               label=1, starting_id=ann_id)

    # Place dead chickens on empty stalls (unchanged)
    final_images, dead_annotations, ann_id = place_bounding_boxes_on_stalls(final_images, dead_crops, dead_bboxes,
                                                                            dead_segmentations,
                                                                            dead_repeat_range, label=2,
                                                                            starting_id=ann_id)

    # Combine annotations
    annotations = healthy_annotations + dead_annotations
    return final_images, annotations


# Main script
if __name__ == "__main__":
    # Define paths to the directories and annotation files
    healthy_image_dir = r'C:\Users\Admin\PycharmProjects\chickenLocalization\healthy_images'
    dead_image_dir = r'C:\Users\Admin\PycharmProjects\chickenLocalization\dead_images'
    empty_stall_dir = r'C:\Users\Admin\PycharmProjects\chickenLocalization\empty_stalls'
    healthy_annotations_file = r'C:\Users\Admin\PycharmProjects\chickenLocalization\healthy_annotations\annotations\annotations.json'
    dead_annotations_file = r'C:\Users\Admin\PycharmProjects\chickenLocalization\dead_annotations\annotations\annotations.json'

    # Load images and annotations
    healthy_images, himnames = load_images(healthy_image_dir)
    dead_images, dimnames = load_images(dead_image_dir)
    empty_stalls, eimnames = load_images(empty_stall_dir)
    healthy_annotations = load_json_annotations(healthy_annotations_file)
    dead_annotations = load_json_annotations(dead_annotations_file)

    # Define the repeat ranges
    healthy_repeat_range = (50, 150)  # Increased the range for more healthy chickens
    dead_repeat_range = (5, 10)

    # Combine images with stalls, using a uniform max crop size for both categories
    final_images, annotations = combine_images_with_stalls(empty_stalls, healthy_annotations,
                                                           dead_annotations, healthy_repeat_range,
                                                           dead_repeat_range, max_crop_size=(80, 80))  # Set equal size

    # Create the output directory
    output_dir = 'result_images'
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