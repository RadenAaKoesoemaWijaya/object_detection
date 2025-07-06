import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

def load_yolo_model():
    """
    Load or download the YOLO model for object detection
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if model exists, otherwise download it
    model_path = "models/yolov8n.pt"  # Using YOLOv8 nano model
    if not os.path.exists(model_path):
        # Use pre-trained YOLOv8 model and save it directly
        import shutil
        # Download model to temporary location and copy to models directory
        temp_model = YOLO("yolov8n.pt")
        # Copy the downloaded model file to our models directory
        source_path = "yolov8n.pt"  # Default location when downloading
        if os.path.exists(source_path):
            shutil.copy(source_path, model_path)
    
    # Load model
    model = YOLO(model_path)
    
    return model

def load_yolo_with_torch_load(model_path):
    from ultralytics.nn.tasks import DetectionModel
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DetectionModel(ckpt["model"].yaml)
    model.load_state_dict(ckpt["model"].state_dict())
    return model

def detect_objects(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Detect objects in the image using YOLO model
    
    Args:
        model: YOLO model
        image: Input image (numpy array)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of detections [x1, y1, x2, y2, confidence, class_id]
    """
    # Run inference
    results = model(image, conf=conf_threshold, iou=iou_threshold)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates (convert to int for pixel coordinates)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            detections.append([x1, y1, x2, y2, confidence, class_id])
    
    return detections