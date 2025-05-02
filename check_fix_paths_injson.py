import json
import os

# Path to your existing coco annotations
annotations_path = "C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/valid/annotations/valid_coco_annotations.json"

# Load your annotations
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# Update file_name to only contain the filename (remove full path)
for annotation in annotations['images']:
    # Get only the filename part of the full path
    annotation['file_name'] = os.path.basename(annotation['file_name'])

# Save the updated annotations
updated_annotations_path = "C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/valid/annotations/valid_coco_annotations_updated.json"
with open(updated_annotations_path, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations have been updated and saved to {updated_annotations_path}")
