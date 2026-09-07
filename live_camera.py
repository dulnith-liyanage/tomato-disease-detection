import cv2
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index (0 for built-in, 1 for external/iPhone)')
    args = parser.parse_args()

    print("Loading AI Model (this might take a few seconds)...")
    model = tf.keras.models.load_model('plant_disease_model.keras')
    with open('labels.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    print(f"Opening Webcam (Index {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Check your permissions.")
        return

    print("\n" + "="*40)
    print("Webcam opened! Press 'q' to quit.")
    print("="*40 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # STRICT green color range to ignore yellow walls and dark hair
        # Hue 35-85 is strictly green. Saturation > 40 ignores greys/blacks.
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask to remove tiny noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area, keep top 3 largest leaves
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        
        boxes_drawn = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8000: # Minimum size for a leaf
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Make the bounding box square so we don't warp the leaf when resizing
                center_x, center_y = x + w//2, y + h//2
                size = max(w, h)
                x1 = max(0, center_x - size//2)
                y1 = max(0, center_y - size//2)
                x2 = min(frame.shape[1], center_x + size//2)
                y2 = min(frame.shape[0], center_y + size//2)
                
                roi = frame[y1:y2, x1:x2]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue
                    
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (224, 224))
                img_array = tf.keras.utils.img_to_array(roi_resized)
                img_array = tf.expand_dims(img_array, 0)
                
                # Predict
                predictions = model.predict(img_array, verbose=0)
                score = predictions[0]
                max_score = np.max(score)
                
                if max_score > 0.40:
                    predicted_class = class_names[np.argmax(score)]
                    label = predicted_class.replace('___', ' - ').replace('_', ' ').title()
                    text = f"{label} ({max_score*100:.0f}%)"
                    color = (0, 255, 0) if "Healthy" in label else (0, 0, 255)
                else:
                    text = f"Uncertain ({max_score*100:.0f}%)"
                    color = (0, 255, 255)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                boxes_drawn += 1

        # If no leaves are found, put a small hint at the bottom
        if boxes_drawn == 0:
            cv2.putText(frame, "Scanning for green leaves...", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Tomato Disease Live Scanner", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
