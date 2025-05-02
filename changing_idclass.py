import os

# Base path to your dataset
base_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dataset'

# Define the class ID mapping: Old ID → New ID
class_mapping = {
    1: 0,  # Healthy chicken: Change from 1 to 0
    2: 1   # Dead chicken: Change from 2 to 1
}


# Function to update label files in a specific directory
def update_labels(label_dir, mapping):
    # Ensure the directory exists
    if not os.path.exists(label_dir):
        print(f"Skipping missing folder: {label_dir}")
        return

    # Process each label file
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):  # Only process .txt files
            file_path = os.path.join(label_dir, label_file)

            # Read and update each line
            updated_lines = []
            with open(file_path, 'r') as file:
                for line in file:
                    items = line.strip().split()
                    if items:
                        old_class_id = int(items[0])
                        if old_class_id in mapping:
                            items[0] = str(mapping[old_class_id])  # Update class ID
                        updated_lines.append(" ".join(items))

            # Overwrite the file with updated content
            with open(file_path, 'w') as file:
                file.write("\n".join(updated_lines))
                file.write("\n")  # Ensure the file ends with a newline

    print(f"Updated labels in folder: {label_dir}")


# Process each dataset split (train, valid, test)
for split in ['train', 'valid', 'test']:
    label_dir = os.path.join(base_path, split, 'labels')
    update_labels(label_dir, class_mapping)

print("All label files have been updated!")
