# Tomato Disease Detection for Edge Devices (Raspberry Pi)

An end-to-end Machine Learning pipeline for training and deploying a lightweight, highly accurate Convolutional Neural Network (CNN) to detect 10 different classes of tomato plant diseases.

This project was specifically engineered to be deployed on a **Raspberry Pi** using **TensorFlow Lite**.

## 🚀 The "White Paper" Hardware Solution
During development, we discovered a massive **Domain Shift** issue: AI models trained on laboratory-grade data (perfect backgrounds) struggle heavily when tested on real-world data (complex backgrounds like dirt and varying sunlight). 

Instead of requiring thousands of real-world images and a massive, slow Neural Network, we engineered a brilliant **Hardware Solution**:
By placing a simple white piece of paper behind the leaf in the field, we perfectly mimic the laboratory conditions the AI was trained on. This allows us to use a highly compressed (2.4 MB) model that runs instantly on a Raspberry Pi while achieving **86.60% accuracy**!

## 📂 Project Structure

* `train_model.py`: The core training script. Uses Transfer Learning (MobileNetV2) and a two-phase fine-tuning approach with Early Stopping to train the model on the PlantVillage dataset.
* `convert_tflite.py`: Compresses the heavy 21MB `.keras` model into a tiny 2.4MB `.tflite` model using Quantization.
* `test_model.py`: Evaluates a single image against the heavy Keras model.
* `evaluate_tflite.py`: Simulates the Raspberry Pi environment by evaluating images directly against the compressed `.tflite` model.
* `labels.txt`: The class mappings required for inference.

## 🛠️ How to Use

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
Ensure you have the `dataset/` folder populated with laboratory images, then run:
```bash
python train_model.py
```

### 3. Convert for Deployment
Once the model is trained, compress it for the Raspberry Pi:
```bash
python convert_tflite.py
```

### 4. Deploy to Raspberry Pi
Copy `plant_disease_model.tflite` and `labels.txt` to your Raspberry Pi and integrate it with your edge camera module!
