import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import img_to_array, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes raised")

# Modifikasi fungsi train_classification_model untuk menerima custom callbacks dan model_path
def train_classification_model(dataset_path, epochs=20, batch_size=32, img_size=64, validation_split=0.2, custom_callbacks=None, model_path="models/classification_model.h5"):
    """Train a classification model using the provided dataset
    
    Args:
        dataset_path: Path to the classification dataset (folder with class subfolders)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for model input
        validation_split: Fraction of data to use for validation
        custom_callbacks: List of additional callbacks to use during training
        model_path: Path where to save the model
        
    Returns:
        Trained model and training history
    """
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Combine callbacks
    callbacks = [checkpoint, early_stopping]
    if custom_callbacks:
        callbacks.extend(custom_callbacks)
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save class indices mapping
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    
    # Save class names to file
    with open("models/class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    return model, history, class_names

def prepare_yolo_dataset(dataset_path, output_path="yolo_dataset"):
    """
    Prepare a dataset in YOLO format from XML annotations
    
    Args:
        dataset_path: Path to the dataset with Images and XML Files folders
        output_path: Path to save the prepared dataset
        
    Returns:
        Path to the prepared dataset
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)
    
    # Find images and XML files
    images_dir = os.path.join(dataset_path, "Images")
    xml_dir = os.path.join(dataset_path, "XML Files")
    
    if not os.path.exists(images_dir) or not os.path.exists(xml_dir):
        raise ValueError("Dataset must contain 'Images' and 'XML Files' folders")
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    
    # Create class mapping
    class_mapping = {}
    class_count = {}
    
    # Process each image and its annotation
    for i, img_file in enumerate(image_files):
        # Get corresponding XML file
        xml_file = os.path.splitext(img_file)[0] + ".xml"
        xml_path = os.path.join(xml_dir, xml_file)
        
        if not os.path.exists(xml_path):
            continue
        
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # Create YOLO format label file
        label_content = []
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            
            # Add class to mapping if not exists
            if class_name not in class_mapping:
                class_mapping[class_name] = len(class_mapping)
                class_count[class_name] = 0
            
            class_count[class_name] += 1
            class_id = class_mapping[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            # Add to label content
            label_content.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
        
        # Determine if this sample goes to train or validation set (80/20 split)
        is_train = i < int(len(image_files) * 0.8)
        subset = "train" if is_train else "val"
        
        # Copy image to dataset
        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_path, "images", subset, img_file)
        shutil.copy(src_img_path, dst_img_path)
        
        # Write label file
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(output_path, "labels", subset, label_file)
        
        with open(label_path, "w") as f:
            f.write("\n".join(label_content))
    
    # Create dataset.yaml file
    yaml_content = f"path: {os.path.abspath(output_path)}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += f"nc: {len(class_mapping)}\n"
    yaml_content += f"names: {list(class_mapping.keys())}\n"
    
    with open(os.path.join(output_path, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    
    # Print dataset statistics
    print(f"Dataset prepared with {len(image_files)} images")
    print(f"Classes: {class_mapping}")
    print(f"Class counts: {class_count}")
    
    return output_path, class_mapping

# Modifikasi fungsi train_yolo_model untuk menerima model_name dan progress_callback
def train_yolo_model(dataset_path, epochs=50, img_size=640, batch_size=16, model_name="yolov8n.pt", progress_callback=None):
    """Train a YOLO model using the prepared dataset
    
    Args:
        dataset_path: Path to the prepared YOLO dataset
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
        model_name: Name to save the model as
        progress_callback: Callback function to update training progress
        
    Returns:
        Path to the trained model
    """
    # Load a pre-trained YOLO model
    model = YOLO('yolov8n.pt')
    
    # Get path to dataset.yaml
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    
    if not os.path.exists(yaml_path):
        raise ValueError("dataset.yaml not found in the dataset path")
    
    # Setup callback for progress updates if provided
    if progress_callback:
        class ProgressCallback():
            def __init__(self):
                self.epoch = 0
                
            def __call__(self, trainer):
                # Update progress every iteration
                self.epoch = trainer.epoch
                progress = trainer.epoch / epochs
                metrics = {
                    'loss': float(trainer.loss.cpu().numpy() if isinstance(trainer.loss, torch.Tensor) else trainer.loss),
                    'precision': float(trainer.metrics.get('precision', 0)),
                    'recall': float(trainer.metrics.get('recall', 0))
                }
                progress_callback(progress, self.epoch, metrics)
        
        progress_monitor = ProgressCallback()
    else:
        progress_monitor = None
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='yolo_microscopic_objects',
        callbacks=[progress_monitor] if progress_monitor else None
    )
    
    # Copy the best model to models directory with custom name
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("runs", "detect", "yolo_microscopic_objects", "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        model_save_path = os.path.join("models", model_name)
        shutil.copy(best_model_path, model_save_path)
    
    return os.path.join("models", model_name)

def plot_training_history(history):
    """
    Plot training history for classification model
    
    Args:
        history: Training history from model.fit()
        
    Returns:
        Matplotlib figure with plots
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    return fig