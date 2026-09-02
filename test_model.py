import tensorflow as tf
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_image>")
        print("Example: python test_model.py dataset/Tomato___Early_blight/image1.JPG")
        return

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    model_path = 'plant_disease_model.keras'
    labels_path = 'labels.txt'

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("Error: Model or labels not found. Make sure you run train_model.py first.")
        return

    # Suppress TensorFlow logging warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("Loading model (this might take a few seconds)...")
    model = tf.keras.models.load_model(model_path)

    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    print(f"Processing image: {image_path}")
    # Load the image and resize it to the exact size the model expects (224x224)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    
    # The model expects a "batch" of images, so we add an extra dimension to our single image
    img_array = tf.expand_dims(img_array, 0) 

    print("Running prediction...")
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0] # Our model outputs softmax probabilities directly

    predicted_class = class_names[np.argmax(score)]
    
    # Clean up the dataset folder name for human readability
    # Example: 'Tomato___Septoria_leaf_spot' -> 'Tomato - Septoria Leaf Spot'
    formatted_class = predicted_class.replace('___', ' - ').replace('_', ' ').title()
    
    confidence = 100 * np.max(score)

    print("\n" + "="*50)
    print(" 🌿 PREDICTION RESULT 🌿")
    print("="*50)
    print(f"Diagnosis:  {formatted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
