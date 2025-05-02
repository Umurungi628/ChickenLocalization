import os

# Define the directory for your original images and labels
original_image_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\train\images'  # Update to your original images directory
original_label_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset\train\labels'  # Update to your original labels directory

# Define class IDs (adjust these based on your dataset)
class_ids = {
    'dead_chicken': 2,
    'healthy_chicken': 1
}

# Loop through each image file in the original image directory
for image_file in os.listdir(original_image_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):  # Ensure you're checking image files
        label_file = os.path.splitext(image_file)[0] + '.txt'  # Get corresponding label file name
        label_path = os.path.join(original_label_dir, label_file)

        # Initialize counts for each image
        class_counts = {key: 0 for key in class_ids.keys()}

        # Check if the label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:

                lines = f.readlines()
                print(f"\nContents of {label_file}:")
                for line in lines:
                    print(line.strip())  # Print each line of the label file
                    class_id = int(line.split()[0])  # Get the class ID (first number in each line)

                    # Count occurrences of each class
                    for class_name, class_id_value in class_ids.items():
                        if class_id == class_id_value:
                            class_counts[class_name] += 1

            # Print the results for the specified image
            print(f"Class counts for {image_file}:")
            for class_name, count in class_counts.items():
                print(f"{class_name}: {count} instances")
        else:
            print(f"Label file for {image_file} does not exist.")