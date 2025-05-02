import os
import subprocess
cache_file = r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\augmented\train\Images.cache'
if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"Cache file {cache_file} deleted.")
else:
    print("Cache file does not exist.")
    # Define the YOLOv8 command to run the training (assuming you have YOLOv8 installed and accessible)
    yolo_command = [
        'yolo',  # YOLOv8 executable
        'task=detect',
        'mode=train',
        'data=/Users/User/PycharmProjects/ChickenLocalization/ChickenLocalization-main/dataset.yaml',  # Path to your dataset YAML file
        'cache=False'  # Optional: disable caching if you don't want it
    ]

    # Run the YOLOv8 command
    print("Starting YOLOv8 training...")
    subprocess.run(yolo_command)