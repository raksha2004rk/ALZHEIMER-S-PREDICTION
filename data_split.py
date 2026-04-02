import os
import shutil
import random

# Your original dataset folder
source_dir = "dataset"

# Destination folders
train_dir = "artifacts/train_data"
test_dir = "artifacts/test_data"

# Train-Test ratio
split_ratio = 0.8   # 80% train, 20% test

for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create folders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Copy training images
    for img in train_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(train_dir, category, img)
        )

    # Copy testing images
    for img in test_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(test_dir, category, img)
        )

print("✅ Dataset successfully split into train and test!")