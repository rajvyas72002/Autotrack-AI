import os
import random
import shutil

# ✅ Set your paths
base_path = r"C:\Users\rajvy\OneDrive\Desktop\Autotrack_ai\number_plate"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

# ✅ Create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

# ✅ Collect all image files
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# ✅ Shuffle and split
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))  # 80% for training
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# ✅ Move files
def move_files(file_list, split):
    for img_file in file_list:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        
        src_img = os.path.join(images_path, img_file)
        src_label = os.path.join(labels_path, label_file)
        
        dst_img = os.path.join(images_path, split, img_file)
        dst_label = os.path.join(labels_path, split, label_file)

        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.move(src_img, dst_img)
            shutil.move(src_label, dst_label)
        else:
            print(f"Skipping: {img_file} or {label_file} missing.")

move_files(train_files, 'train')
move_files(val_files, 'val')

print("✅ Dataset successfully split into train/val!")
