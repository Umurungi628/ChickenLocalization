import yaml  # Import the yaml module

# Specify your paths
train_images = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\train\images'
val_images = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\valid\images'
test_images = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\test\images'

# Specify the class names
class_names = ['dead_chicken', 'healthy_chicken']  # Add your class names here

# Define the YAML structure
data = {
    'path': 'C:/Users/User/PycharmProjects/chickenLocalization/chickenLocalization-main/dataset',  # Base path for dataset
    'train': train_images,  # Path to train images
    'val': val_images,      # Path to validation images
    'test': test_images,    # Path to test images
    'names': class_names,   # List your classes here
    'nc': len(class_names)  # Number of classes
}

# Save the dictionary to a .yaml file
with open('dataset.yaml', 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)

print("YAML file with train, val, and test paths created successfully!")

