import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the new data loader
from finetune_data import get_finetune_dataloaders

# CONFIG
LEARNING_RATE = 0.0001 # Slower learning rate for fine-tuning
EPOCHS = 15 # Fewer epochs needed because the model is already smart
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_resnet_model(num_classes=7):
    print("Downloading ResNet18 (Pre-trained)...")
    # Load pre-trained ResNet
    model = models.resnet18(weights='DEFAULT')
    
    # Freeze the early layers (Optional: unfreeze for better accuracy if you have time)
    # For this assignment, let's keep them active but use a low learning rate
    
    # Replace the last fully connected layer
    # ResNet18's last layer is named 'fc' and has 512 input features
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / len(loader), 100 * correct / total

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
            
    return running_loss / len(loader), 100 * correct / total

def main():
    print(f"Fine-Tuning on device: {DEVICE}")
    
    # 1. Load Data (Upscaled to 224x224)
    train_loader, val_loader, test_loader = get_finetune_dataloaders('fer2013.csv')
    
    # 2. Build Model
    model = build_resnet_model(num_classes=7).to(DEVICE)
    
    # 3. Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1)
    
    # 4. Loop
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}
    
    print("Starting Fine-Tuning...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet_finetuned.pth")
            print(">>> Best Model Saved!")
            
    print(f"Done! Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()