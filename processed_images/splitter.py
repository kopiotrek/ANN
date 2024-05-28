import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
base_dir = "/home/koczka/Documents/ANN/processed_images/original_dataset"
classes = ["No_findings", "Pneumonia"]
output_dirs = ["train", "val", "test"]

# Ratios for split
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

# Ensure the ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

# Create directories
for output_dir in output_dirs:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, output_dir, cls), exist_ok=True)

# Function to split and copy files
def split_and_copy(class_dir):
    files = os.listdir(class_dir)
    files = [f for f in files if os.path.isfile(os.path.join(class_dir, f))]
    
    train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)
    
    return train_files, val_files, test_files

# Split and copy files for each class
for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    train_files, val_files, test_files = split_and_copy(class_dir)
    
    for file in train_files:
        shutil.copy(os.path.join(class_dir, file), os.path.join(base_dir, 'train', cls, file))
    
    for file in val_files:
        shutil.copy(os.path.join(class_dir, file), os.path.join(base_dir, 'val', cls, file))
    
    for file in test_files:
        shutil.copy(os.path.join(class_dir, file), os.path.join(base_dir, 'test', cls, file))

print("Dataset split completed.")
