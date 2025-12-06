import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # Progress bar

# IMPORT YOUR LOCAL MODULES
# Assuming you named them data_setup.py and model.py based on your terminal output
try:
    from data_setup import get_dataloaders, EMOTION_DICT
    from model import FaceEmotionNet
except ImportError:
    # Fallback if you used the original filenames I gave
    from data_setup import get_dataloaders, EMOTION_DICT
    from model import FaceEmotionNet

# HYPERPARAMETERS
LEARNING_RATE = 0.001
EPOCHS = 25 # FER-2013 is hard, needs time. 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On Mac M1/M2, you might use "mps", but "cpu" is safe for now if not sure.

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        loop.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    print(f"Training on device: {DEVICE}")
    
    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders('fer2013.csv')
    
    # 2. Initialize Model
    model = FaceEmotionNet(num_classes=7).to(DEVICE)
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Scheduler reduces LR if validation accuracy plateaus
    # REMOVED verbose=True to fix TypeError in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # 4. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update Scheduler
        scheduler.step(val_acc)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> New Best Model Saved!")

    # 5. Plotting Results
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.savefig('training_curves.png')
    print("\nTraining Complete. Best Validation Accuracy: {:.2f}%".format(best_val_acc))
    print("Model saved to 'best_model.pth'. Training curves saved to 'training_curves.png'.")

if __name__ == "__main__":
    main()