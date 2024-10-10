import albumentations as A
import cv2
import os
import json
from tqdm import tqdm

# Define the augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_id'], min_area=0, min_visibility=0.3))

# Input and output paths
image_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\result_images'
annotation_file = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\result_images\annotations.json'
output_image_dir = 'augmented_images'
output_annotation_file = 'augmented_annotations/augmented_annotations.json'

# Load the COCO annotations
with open(annotation_file) as f:
    coco_annotations = json.load(f)

# Prepare output annotations
augmented_annotations = {
    'images': [],
    'annotations': [],
    'categories': coco_annotations['categories'],  # Copy categories directly
}

# Create directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)

# Keep track of IDs for augmented data
new_image_id = max([img['id'] for img in coco_annotations['images']]) + 1
new_annotation_id = max([ann['id'] for ann in coco_annotations['annotations']]) + 1

# Loop over each image and apply augmentations
for image_info in tqdm(coco_annotations['images']):
    image_id = image_info['id']
    img_name = image_info['file_name']
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)

    # Get all annotations for this image
    image_annotations = [ann for ann in coco_annotations['annotations'] if ann['image_id'] == image_id]

    # Extract bounding boxes and class labels
    bboxes = [ann['bbox'] for ann in image_annotations]
    category_ids = [ann['category_id'] for ann in image_annotations]

    # Augment image and bounding boxes
    for i in range(200):  # Create 5 augmentations per image
        augmented = augmentations(image=image, bboxes=bboxes, category_id=category_ids)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']

        # Save the augmented image
        aug_img_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(output_image_dir, aug_img_name), aug_image)

        # Update image info
        new_image_info = image_info.copy()
        new_image_info['file_name'] = aug_img_name
        new_image_info['id'] = new_image_id
        augmented_annotations['images'].append(new_image_info)

        # Update the annotations
        for bbox, category_id in zip(aug_bboxes, category_ids):
            new_annotation = {
                'id': new_annotation_id,
                'image_id': new_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],  # width * height for COCO
                'iscrowd': 0
            }
            augmented_annotations['annotations'].append(new_annotation)
            new_annotation_id += 1

        new_image_id += 1

# Save the augmented annotations
with open(output_annotation_file, 'w') as f:
    json.dump(augmented_annotations, f)

print(f"Augmentation completed. Augmented images and annotations saved to {output_image_dir} and {output_annotation_file}")
