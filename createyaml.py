import os

# Specify your paths
train_images_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmented_images\train'
val_images_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmented_images\valid'
# Optional test images path
test_images_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmented_images\test'

# Specify the class names
class_names = ['dead_chicken', 'healthy_chicken']  # Add your class names here

# Prepare the data.yaml content
yaml_content = f"""train: {train_images_path}
val: {val_images_path}
#test: {test_images_path} Optional

nc: {len(class_names)}  # Number of classes
names: {class_names}  # Class names
"""

# Write to data.yaml
yaml_file_path = 'data.yaml'
with open(yaml_file_path, 'w') as f:
    f.write(yaml_content)

print(f'YAML file created: {yaml_file_path}')
