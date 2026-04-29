import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import ParquetMRIDataset, get_transforms
from models.cnn import AlzheimerCNN
from models.neurokan import NeuroKAN
from models.random_forest_model import train_rf, extract_features

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Tri-Model Training Pipeline")
    parser.add_argument('--model_type', type=str, required=True, choices=['kan', 'cnn', 'rf'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print(f"--- Training {args.model_type.upper()} ---")

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'train.parquet')
    
    # Random Forest extracts features sequentially; CNN/KAN train in batches
    batch_size = 32 if args.model_type in ['cnn', 'kan'] else 8
    
    dataset = ParquetMRIDataset(train_path, transform=get_transforms(is_train=True))
    
    # Train on full dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if args.model_type == 'rf':
        print("Extracting features using EfficientNet backbone...")
        X_train, y_train = extract_features(train_loader, DEVICE)
        rf_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.joblib')
        train_rf(X_train, y_train, save_path=rf_save_path)
        return

    # Deep Learning Models
    if args.model_type == 'cnn':
        model = AlzheimerCNN(num_classes=4).to(DEVICE)
        save_path = '../models/cnn_final.pth'
    elif args.model_type == 'kan':
        model = NeuroKAN(num_classes=4).to(DEVICE)
        save_path = '../models/neurokan_final.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, os.path.basename(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    main()
