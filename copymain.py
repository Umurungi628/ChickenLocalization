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
def transform_bounding_boxes(images, imnames, annotations, repeat_range, max_crop_size=(80, 80), im_type='healthy'):
    export_bounding_boxes = []
    export_segmentations = []
    export_crops = []

    for i, image in enumerate(images):
        name = imnames[i]
        bboxes, segmentations = getimbboxes(name, annotations)
        if im_type == 'healthy':
            repeats = random.sample(range(repeat_range[0], repeat_range[1]), 1)[0] if repeat_range[1] <= len(bboxes) else len(bboxes)
            rand_bbox_id = random.sample(range(len(bboxes)), repeats)
            for i in rand_bbox_id:
                bbox = bboxes[i]
                segmentation = segmentations[i]
                x, y, w, h = map(int, bbox)
                crop = image[y:y + h, x:x + w]
                crop = cv2.resize(crop, max_crop_size)  # Resize crop to max crop size
                xc = np.array(segmentation, dtype=float).flatten()[::2] - x
                yc = np.array(segmentation, dtype=float).flatten()[1::2] - y
                segmentation = np.array([xc, yc], dtype=int).transpose().reshape(1, 2 * len(xc)).tolist()

                export_crops.append(crop)
                export_bounding_boxes.append(bbox)
                export_segmentations.append(segmentation)
        else:
            for _ in range(random.randint(*repeat_range)):
                bbox = bboxes[0]
                contours = segmentations[0]
                x, y, w, h = map(int, bbox)
                xc = np.array(contours, dtype=float).flatten()[::2] - x
                yc = np.array(contours, dtype=float).flatten()[1::2] - y
                if isinstance(bbox, list) and len(bbox) == 4:
                    x, y, w, h = map(int, bbox)

                    # Limit the bounding box dimensions
                    #new_w, new_h = min(max_crop_size[0], w), min(max_crop_size[1], h)
                    #center_x, center_y = x + w // 2, y + h // 2
                    #new_x, new_y = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)

                    if x > 0: #new_y + new_h <= image.shape[0] and new_x + new_w <= image.shape[1]:
                        #crop = image[new_y:new_y + new_h, new_x:new_x + new_w]
                        crop = image[y:y + h, x:x + w]
                        crop = cv2.resize(crop, max_crop_size)  # Resize crop to max crop size

                        # Apply enhancements
                        crop = enhance_crop(crop)

                        # Apply random rotation
                        angle = random.randint(0, 360)
                        M = cv2.getRotationMatrix2D((crop.shape[1] // 2, crop.shape[0] // 2), angle, 1)
                        crop_rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))
                        # Apply random rotation to polygon contours
                        angler = np.radians(angle)
                        xcr = (xc - w // 2) * np.cos(angler) + (yc - h // 2) * np.sin(angler) + w // 2
                        ycr = (xc - w // 2) * np.sin(angler) + (yc - h // 2) * np.cos(angler) + w // 2

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

                        coords = np.array([xcr, ycr]).transpose().reshape(1, len(xcr), 2)
                        coords_t = cv2.perspectiveTransform(coords, perspective_matrix)
                        coords_t = coords_t.transpose().reshape(2, len(xcr))
                        xct = coords_t[0, :]
                        yct = coords_t[1, :]


                        export_crops.append(crop_transformed)
                        export_crops.append(crop)
                        export_bounding_boxes.append([x, y, w, h])
                        export_segmentations.append(
                            np.array([xct, yct], dtype=int).transpose().reshape(1, 2 * len(xct)).tolist())
    return export_crops, export_bounding_boxes, export_segmentations


# Function to place bounding boxes on empty stalls
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
    # Transform healthy chickens
    healthy_crops, healthy_bboxes, healthy_segmentations = transform_bounding_boxes(healthy_images, himnames, healthy_anot, healthy_repeat_range, max_crop_size, im_type='healthy')

    # Transform dead chickens
    dead_crops, dead_bboxes, dead_segmentations = transform_bounding_boxes(dead_images, dimnames, dead_anot, dead_repeat_range, max_crop_size, im_type='dead')

    # Initialize the annotation ID
    ann_id = 0

    # Place healthy chickens on empty stalls
    final_images, healthy_annotations, ann_id = place_bounding_boxes_on_stalls(empty_stalls, healthy_crops,
                                                                               healthy_bboxes,
                                                                               healthy_segmentations, healthy_repeat_range,
                                                                               label=1, starting_id=ann_id)

    # Place dead chickens on empty stalls
    final_images, dead_annotations, ann_id = place_bounding_boxes_on_stalls(final_images, dead_crops, dead_bboxes, dead_segmentations,
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