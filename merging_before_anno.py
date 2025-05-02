import cv2
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from u2net import U2NET

# Paths
stall_images_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\healthy_images'
dead_chickens_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\dead_images'
output_path = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\merging_before_ann'

os.makedirs(output_path, exist_ok=True)

# Load U²-Net model (for better background removal)
model = U2NET(3, 1)
model.load_state_dict(torch.load("u2net.pth", map_location=torch.device("cpu")))
model.eval()

# Preprocessing for U²-Net
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])


def remove_background(image_path):
    """ Remove background using U²-Net deep learning model """
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        mask = model(img_tensor)[0]

    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (image.width, image.height))

    # Threshold the mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Apply mask
    image_cv = np.array(image)
    bg_removed = cv2.bitwise_and(image_cv, image_cv, mask=mask)

    return bg_removed


def overlay_dead_chicken(stall_img, dead_img):
    """ Overlay dead chicken onto stall image with proper placement and blending """
    stall = cv2.imread(stall_img)
    dead = remove_background(dead_img)

    # Convert to grayscale & find contours
    gray = cv2.cvtColor(dead, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        dead = dead[y:y + h, x:x + w]  # Crop object region

    # Resize dead chicken based on healthy chicken size (scale proportionally)
    avg_chicken_size = (stall.shape[1] // 8, stall.shape[0] // 8)
    dead = cv2.resize(dead, avg_chicken_size, interpolation=cv2.INTER_AREA)

    # Select a random position inside the stall
    max_x = stall.shape[1] - dead.shape[1]
    max_y = stall.shape[0] - dead.shape[0]
    x_offset = random.randint(50, max_x - 50)
    y_offset = random.randint(50, max_y - 50)

    # Poisson blending for realistic merging
    center = (x_offset + dead.shape[1] // 2, y_offset + dead.shape[0] // 2)
    merged = cv2.seamlessClone(dead, stall, np.ones_like(dead[:, :, 0]) * 255, center, cv2.NORMAL_CLONE)

    return merged


# Process each stall image
for i, stall_img in enumerate(os.listdir(stall_images_path)):
    if i < len(os.listdir(dead_chickens_path)):
        stall_img_path = os.path.join(stall_images_path, stall_img)
        dead_chicken_img_path = os.path.join(dead_chickens_path, os.listdir(dead_chickens_path)[i])

        merged_img = overlay_dead_chicken(stall_img_path, dead_chicken_img_path)

        # Save merged image
        output_filename = os.path.join(output_path, f"merged_{i}.jpg")
        cv2.imwrite(output_filename, merged_img)
        print(f"Saved: {output_filename}")

print("Dataset merging complete!")
