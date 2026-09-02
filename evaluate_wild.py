import os
import random
import shutil
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_dir = "plantdoc_dataset/test"
output_dir = "wild_test_images"

# Updated Mapping to match our new Unified Tomato classes
mapping = {
    "Tomato_leaf_bacterial_spot": "Tomato_Bacterial_spot",
    "Tomato_Early_blight_leaf": "Tomato_Early_blight",
    "Tomato_leaf_late_blight": "Tomato_Late_blight",
    "Tomato_mold_leaf": "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato_two_spotted_spider_mites_leaf": "Tomato_Spider_mites",
    "Tomato_leaf_yellow_virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_leaf_mosaic_virus": "Tomato_mosaic_virus",
    "Tomato_leaf": "Tomato_healthy",
}

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading model (this might take a moment)...")
    model = tf.keras.models.load_model('plant_disease_model.keras')

    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    correct = 0
    total = 0

    print(f"\n{'-'*90}")
    print(f"{'True Disease (Internet Image)':<35} | {'Model Prediction':<35} | {'Status':<10}")
    print(f"{'-'*90}")

    for plantdoc_folder, our_class in mapping.items():
        src_folder = os.path.join(test_dir, plantdoc_folder)
        if not os.path.exists(src_folder):
            continue
            
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue
            
        img_name = random.choice(images)
        src_img = os.path.join(src_folder, img_name)
        
        dest_img = os.path.join(output_dir, f"{our_class}.jpg")
        shutil.copy2(src_img, dest_img)
        
        img = tf.keras.utils.load_img(dest_img, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        predicted_class = class_names[np.argmax(score)]
        
        status = "✅ PASS" if predicted_class == our_class else "❌ FAIL"
        if predicted_class == our_class:
            correct += 1
        total += 1
        
        display_true = our_class.replace('_', ' ')
        display_pred = predicted_class.replace('_', ' ')
        
        print(f"{display_true[:33]:<35} | {display_pred[:33]:<35} | {status}")

    print(f"{'-'*90}")
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Final Model Rating (In-The-Wild Accuracy): {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    main()
