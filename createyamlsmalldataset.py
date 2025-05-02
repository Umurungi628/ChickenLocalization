import yaml

# Data for the YAML file
data = {
    'train': 'C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/train/images',
    'val': 'C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/valid/images',
    'train_coco': 'C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/train/annotations/train_coco_annotations.json',
    'val_coco': 'C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset_coco/valid/annotations/valid_coco_annotations.json',
    'nc': 2,  # Number of classes
    'names': ['healthy_chicken', 'dead_chicken']  # List of class names
}

# Save as YAML file
with open('dfine_n_coco_config.yaml', 'w') as f:
    yaml.dump(data, f)

print("YAML file created successfully!")

