import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import img_to_array  # Changed from keras.preprocessing.image
import numpy as np
import os

def load_classification_model():
    """
    Load or create a classification model for microscopic objects
    """
    model_path = "models/classification_model.h5"
    
    # Check if model exists
    if os.path.exists(model_path):
        # Load existing model
        model = load_model(model_path)
    else:
        # Create a simple CNN model
        model = Sequential([
            # Convolutional layers
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')  # 4 classes: bacteria, fungi, parasite, blood cell
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
    
    return model

def classify_objects(model, image, detections, target_size=(64, 64)):
    """
    Classify detected objects using the classification model
    
    Args:
        model: Classification model
        image: Original image
        detections: List of detections [x1, y1, x2, y2, confidence, class_id]
        target_size: Size to resize cropped objects for classification
        
    Returns:
        List of classification results (class index and name)
    """
    # Class names for microscopic objects
    class_names = ["Bakteri", "Jamur", "Parasit", "Sel Darah"]
    
    classifications = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        
        # Ensure coordinates are within image boundaries
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop the detected object
        if x2 > x1 and y2 > y1:  # Ensure valid crop dimensions
            cropped_obj = image[y1:y2, x1:x2]
            
            # Resize to target size
            resized_obj = cv2.resize(cropped_obj, target_size)
            
            # Preprocess for model
            preprocessed = resized_obj.astype("float") / 255.0
            preprocessed = np.expand_dims(preprocessed, axis=0)
            
            # Predict class
            predictions = model.predict(preprocessed, verbose=0)
            class_idx = np.argmax(predictions[0])
            class_name = class_names[class_idx]
            confidence = float(predictions[0][class_idx])
            
            classifications.append((class_idx, class_name, confidence))
        else:
            # If crop dimensions are invalid, assign a default class
            classifications.append((0, "Unknown", 0.0))
    
    return classifications