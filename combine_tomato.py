import os
import shutil

dest_dir = "dataset"
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)

def copy_images(src_dir, mapping, prefix):
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} not found. Skipping...")
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
                    dest_file = os.path.join(dest_folder, f"{prefix}_{file}")
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)

# Unified standard names
C_BACTERIAL = "Tomato_Bacterial_spot"
C_EARLY = "Tomato_Early_blight"
C_LATE = "Tomato_Late_blight"
C_MOLD = "Tomato_Leaf_Mold"
C_SEPTORIA = "Tomato_Septoria_leaf_spot"
C_SPIDER = "Tomato_Spider_mites"
C_TARGET = "Tomato_Target_Spot"
C_CURL = "Tomato_Yellow_Leaf_Curl_Virus"
C_MOSAIC = "Tomato_mosaic_virus"
C_HEALTHY = "Tomato_healthy"

# PlantVillage Mapping
map_pv = {
    "Tomato_Bacterial_spot": C_BACTERIAL,
    "Tomato_Early_blight": C_EARLY,
    "Tomato_Late_blight": C_LATE,
    "Tomato_Leaf_Mold": C_MOLD,
    "Tomato_Septoria_leaf_spot": C_SEPTORIA,
    "Tomato_Spider_mites_Two_spotted_spider_mite": C_SPIDER,
    "Tomato__Target_Spot": C_TARGET,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": C_CURL,
    "Tomato__Tomato_mosaic_virus": C_MOSAIC,
    "Tomato_healthy": C_HEALTHY
}

# PlantDoc Mapping
map_pd = {
    "Tomato_leaf_bacterial_spot": C_BACTERIAL,
    "Tomato_Early_blight_leaf": C_EARLY,
    "Tomato_leaf_late_blight": C_LATE,
    "Tomato_mold_leaf": C_MOLD,
    "Tomato_Septoria_leaf_spot": C_SEPTORIA,
    "Tomato_two_spotted_spider_mites_leaf": C_SPIDER,
    "Tomato_leaf_yellow_virus": C_CURL,
    "Tomato_leaf_mosaic_virus": C_MOSAIC,
    "Tomato_leaf": C_HEALTHY
}

# Tomato_Dataset Mapping
map_td = {
    "Tomato___Bacterial_spot": C_BACTERIAL,
    "Tomato___Early_blight": C_EARLY,
    "Tomato___Late_blight": C_LATE,
    "Tomato___Leaf_Mold": C_MOLD,
    "Tomato___Septoria_leaf_spot": C_SEPTORIA,
    "Tomato___Spider_mites Two-spotted_spider_mite": C_SPIDER,
    "Tomato___Target_Spot": C_TARGET,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": C_CURL,
    "Tomato___Tomato_mosaic_virus": C_MOSAIC,
    "Tomato___healthy": C_HEALTHY
}

print("Copying PlantVillage Tomato images...")
copy_images("plantvillage_dataset", map_pv, "pv")
print("Copying PlantDoc Tomato images...")
copy_images("plantdoc_dataset/train", map_pd, "pd")
print("Copying Tomato Dataset images...")
copy_images("tomato_dataset", map_td, "td")
print("Finished combining datasets into /dataset!")
