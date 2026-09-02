import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_dir = "test_images"
tflite_model_path = "plant_disease_model.tflite"

def main():
    print("Loading TFLite model (simulating Raspberry Pi environment)...")
    
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor architecture
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    correct = 0
    total = 0

    print(f"\n{'-'*90}")
    print(f"{'True Disease (Filename)':<35} | {'TFLite Prediction':<35} | {'Status':<10}")
    print(f"{'-'*90}")

    for file in os.listdir(test_dir):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        true_class = file.rsplit('.', 1)[0]
        if true_class == "Tomato_Spider_mites_Two_spotted_spider_mite":
            true_class = "Tomato_Spider_mites"
            
        img_path = os.path.join(test_dir, file)
        
        # Load and preprocess image exactly as before
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Ensure it matches the expected input type (FLOAT32)
        img_array = tf.cast(img_array, tf.float32)

        # Load the image into the TFLite model memory
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference (This is what the Raspberry Pi will do)
        interpreter.invoke()

        # Extract output prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])
        score = predictions[0]
        predicted_class = class_names[np.argmax(score)]
        
        status = "✅ PASS" if predicted_class == true_class else "❌ FAIL"
        if predicted_class == true_class:
            correct += 1
        total += 1
        
        display_true = true_class.replace('_', ' ')
        display_pred = predicted_class.replace('_', ' ')
        
        print(f"{display_true[:33]:<35} | {display_pred[:33]:<35} | {status}")

    print(f"{'-'*90}")
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"TFLite Model Rating: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    main()
