import os
import random
import shutil

dataset_dir = "dataset"  # your original dataset folder with 'images' and 'labels' folders
train_dir = "dataset/train"
val_dir = "dataset/val"
split_ratio = 0.8  # 80% train, 20% val

os.makedirs(train_dir + "/images", exist_ok=True)
os.makedirs(train_dir + "/labels", exist_ok=True)
os.makedirs(val_dir + "/images", exist_ok=True)
os.makedirs(val_dir + "/labels", exist_ok=True)

# List all images in dataset/images
all_images = os.listdir(os.path.join(dataset_dir, "images"))
random.shuffle(all_images)

train_count = int(len(all_images) * split_ratio)

train_images = all_images[:train_count]
val_images = all_images[train_count:]

def copy_files(file_list, dest_dir):
    for img_file in file_list:
        # Copy image
        shutil.copy(os.path.join(dataset_dir, "images", img_file), os.path.join(dest_dir, "images", img_file))
        
        # Copy label with same base name and .txt extension
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.copy(os.path.join(dataset_dir, "labels", label_file), os.path.join(dest_dir, "labels", label_file))

copy_files(train_images, train_dir)
copy_files(val_images, val_dir)

print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
