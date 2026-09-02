import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import matplotlib.pyplot as plt
import os
import ssl

# Fix for macOS SSL CERTIFICATE_VERIFY_FAILED error
ssl._create_default_https_context = ssl._create_unverified_context

DATASET_DIR = 'dataset'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

def main():
    print("1. Loading dataset...")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

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
    
    with open('labels.txt', 'w') as f:
        f.write('\n'.join(class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n2. Building model (MobileNetV2 Transfer Learning)...")
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    base_model = applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model for phase 1
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Early stopping callback prevents overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=6, # Wait for 6 epochs of no improvement before stopping
        restore_best_weights=True,
        verbose=1
    )

    print("\n3. Phase 1: Training the top layer...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stopping]
    )

    print("\n4. Phase 2: Fine-Tuning the base model...")
    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers EXCEPT the very last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile model with a MUCH LOWER learning rate (1e-5) to slowly adjust the weights
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history_1.epoch[-1] + 1,
        callbacks=[early_stopping]
    )

    print("\n5. Saving the fine-tuned model...")
    model.save('plant_disease_model.keras')
    print("Model saved as 'plant_disease_model.keras'")

    # Plot training results
    acc = history_1.history['accuracy'] + history_2.history['accuracy']
    val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    loss = history_1.history['loss'] + history_2.history['loss']
    val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    # Mark the start of fine-tuning
    plt.plot([history_1.epoch[-1], history_1.epoch[-1]], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (Fine-Tuned)')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([history_1.epoch[-1], history_1.epoch[-1]], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss (Fine-Tuned)')
    plt.xlabel('epoch')
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == '__main__':
    main()
