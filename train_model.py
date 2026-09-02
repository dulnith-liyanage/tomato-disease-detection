import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import os
import ssl

# Fix for macOS SSL CERTIFICATE_VERIFY_FAILED error when downloading pre-trained weights
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
DATASET_DIR = 'dataset' # We will place the downloaded images here
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 10

def main():
    print("1. Loading dataset...")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please download the PlantVillage dataset and place the class folders inside the 'dataset/' directory.")
        return

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Save class names to a file so we can use them later on the Raspberry Pi
    with open('labels.txt', 'w') as f:
        f.write('\n'.join(class_names))
    print("Saved labels to 'labels.txt'")

    # Optimize dataset pipeline for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n2. Building model (MobileNetV2 Transfer Learning)...")
    # MobileNetV2 expects pixel values in [-1, 1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Aggressive Data Augmentation to simulate real-world "in-the-wild" conditions
    # This helps the AI learn to ignore the background and focus on the leaf itself.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    # Base model (Pre-trained on ImageNet)
    base_model = applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze the base model weights

    # Construct the final model
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    print("\n3. Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    print("\n4. Saving the model...")
    # Save as standard Keras model
    model.save('plant_disease_model.keras')
    print("Model saved as 'plant_disease_model.keras'")

    # Plot training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == '__main__':
    main()
