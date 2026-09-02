import os
import shutil

src_train = "plantdoc_dataset/train"
src_test = "plantdoc_dataset/test"
dest = "dataset"

# Map PlantDoc folder names to our existing PlantVillage folder names
mapping = {
    "Bell_pepper_leaf": "Pepper__bell___healthy",
    "Bell_pepper_leaf_spot": "Pepper__bell___Bacterial_spot",
    "Potato_leaf_early_blight": "Potato___Early_blight",
    "Potato_leaf_late_blight": "Potato___Late_blight",
    "Tomato_Early_blight_leaf": "Tomato_Early_blight",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_leaf": "Tomato_healthy",
    "Tomato_leaf_bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato_leaf_late_blight": "Tomato_Late_blight",
    "Tomato_leaf_mosaic_virus": "Tomato__Tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_mold_leaf": "Tomato_Leaf_Mold",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
}

def copy_files(src_dir, dest_dir, mapping):
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} not found. Skipping.")
        return

    for folder in os.listdir(src_dir):
        if folder in mapping:
            src_folder = os.path.join(src_dir, folder)
            dest_folder = os.path.join(dest_dir, mapping[folder])
            
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            for file in os.listdir(src_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(src_folder, file)
                    # Prefix to avoid name clashes with existing PlantVillage images
                    dest_file = os.path.join(dest_folder, "plantdoc_" + file)
                    # Only copy if it doesn't already exist so we can run this safely multiple times
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
            print(f"Merged {folder} into {mapping[folder]}")

print("Merging train dataset...")
copy_files(src_train, dest, mapping)
print("Merging complete!")
