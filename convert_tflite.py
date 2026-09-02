import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_to_tflite():
    print("Loading the .keras model...")
    model = tf.keras.models.load_model('plant_disease_model.keras')

    print("Initializing the TensorFlow Lite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # We apply DEFAULT optimization to compress the model (Quantization).
    # This shrinks the file size dramatically and speeds up inference on the Raspberry Pi
    # without significantly hurting accuracy.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting... (This may take a minute)")
    tflite_model = converter.convert()

    print("Saving the .tflite model...")
    with open('plant_disease_model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("\n✅ Conversion Complete!")
    print("The model has been successfully compressed and saved as 'plant_disease_model.tflite'.")
    
    original_size = os.path.getsize('plant_disease_model.keras') / (1024 * 1024)
    new_size = os.path.getsize('plant_disease_model.tflite') / (1024 * 1024)
    print(f"Original Size: {original_size:.2f} MB")
    print(f"Optimized TFLite Size: {new_size:.2f} MB")

if __name__ == "__main__":
    convert_to_tflite()
