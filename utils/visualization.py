import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def visualize_results(image, detections, classifications):
    """
    Visualize detection and classification results on the image
    
    Args:
        image: Original image
        detections: List of detections [x1, y1, x2, y2, confidence, class_id]
        classifications: List of classification results (class index, name, confidence)
        
    Returns:
        Annotated image and counts of each class
    """
    # Make a copy of the image
    result_img = image.copy()
    
    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),    # Green for bacteria
        (0, 165, 255),   # Orange for fungi
        (255, 0, 0),     # Blue for parasite
        (0, 0, 255)      # Red for blood cell
    ]
    
    # Count objects by class
    counts = Counter()
    
    # Draw bounding boxes and labels
    for i, (detection, classification) in enumerate(zip(detections, classifications)):
        x1, y1, x2, y2, conf, _ = detection
        class_idx, class_name, class_conf = classification
        
        # Get color based on class
        color = colors[class_idx] if class_idx < len(colors) else (255, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {class_conf:.2f}"
        
        # Draw label background
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update counts
        counts[class_name] += 1
    
    return result_img, counts

def plot_statistics(data):
    """
    Create statistical visualizations from detection results
    
    Args:
        data: DataFrame with detection results
        
    Returns:
        Matplotlib figure with plots
    """
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    sns.barplot(x="Jenis", y="Jumlah", data=data, ax=ax1)
    ax1.set_title("Jumlah Objek per Kelas")
    ax1.set_ylabel("Jumlah")
    ax1.set_xlabel("Jenis Objek")
    
    # Pie chart
    ax2.pie(data["Jumlah"], labels=data["Jenis"], autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title("Distribusi Persentase Objek")
    
    plt.tight_layout()
    
    return fig