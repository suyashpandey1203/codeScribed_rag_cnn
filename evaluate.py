import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# IMPORT: We need the data loader that resizes images to 224x224 (RGB)
try:
    from finetune_data import get_finetune_dataloaders
except ImportError:
    print("Error: Could not import 'get_finetune_dataloaders'. Ensure 'finetune_data.py' is in the folder.")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels mapping (Same as before)
EMOTION_DICT = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 
    3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

def load_resnet_model(num_classes=7):
    """
    Recreates the exact model architecture used in finetune_train.py
    """
    # Load ResNet structure (weights=None because we will load our own)
    model = models.resnet18(weights=None) 
    
    # Replace the last layer to match our 7 emotions
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def evaluate_model():
    print(f"Evaluation running on device: {DEVICE}")

    # 1. Load Data
    # Note: We must use the same data loader as training because ResNet needs 224x224 RGB images
    print("Loading Test Data (Resizing to 224x224)...")
    _, _, test_loader = get_finetune_dataloaders('fer2013.csv')
    
    if test_loader is None:
        print("Failed to load data.")
        return

    # 2. Load Model Structure
    model = load_resnet_model(num_classes=7).to(DEVICE)
    
    # 3. Load Trained Weights
    weights_path = "resnet_finetuned.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"Loaded weights from '{weights_path}'")
    except FileNotFoundError:
        print(f"Error: '{weights_path}' not found. Run 'finetune_train.py' first!")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 4. Inference
    print("Running Inference on Test Set (this may take a moment due to image size)...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Metrics
    # Map numeric labels to string names
    target_names = [EMOTION_DICT[i] for i in range(7)]
    
    # Classification Report (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - ResNet18 (Fine-Tuned)")
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved to 'confusion_matrix.png'")
    
    # 6. Save Predictions to CSV
    # We will create a CSV mapping Image Index -> Predicted Emotion
    results_df = pd.DataFrame({
        'Image_Index': range(len(all_preds)),
        'True_Label': [EMOTION_DICT[i] for i in all_labels],
        'Predicted_Label': [EMOTION_DICT[i] for i in all_preds],
        'Correct': [p == l for p, l in zip(all_preds, all_labels)]
    })
    
    results_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")

if __name__ == "__main__":
    evaluate_model()