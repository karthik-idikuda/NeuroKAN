import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


class FeatureExtractor(nn.Module):
    """
    Uses a pretrained EfficientNetV2-S to convert
    (B, 3, 224, 224) MRI images → (B, 1280) feature vectors.
    These vectors are then fed into the Random Forest classifier.
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        backbone = models.efficientnet_v2_s(weights='DEFAULT')
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        # Freeze all weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)


def extract_features(data_loader, device):
    """
    Runs the DataLoader through EfficientNet and returns
    numpy arrays of features and labels for Scikit-Learn.
    """
    extractor = FeatureExtractor().to(device)
    extractor.eval()

    all_features = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        feats = extractor(images).cpu().numpy()
        all_features.append(feats)
        all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def train_rf(X_train, y_train, save_path='../models/rf_model.joblib'):
    """
    Train a Random Forest classifier on the extracted feature vectors.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    print("Training Random Forest...")
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Random Forest model saved to {save_path}")
    return model


def load_rf(model_path='../models/rf_model.joblib'):
    """Load a previously saved Random Forest model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_single_rf(image_tensor, rf_model, device):
    """
    Run a single image tensor through feature extraction + Random Forest.
    Returns predicted class index and probability array.
    """
    extractor = FeatureExtractor().to(device)
    extractor.eval()

    with torch.no_grad():
        features = extractor(image_tensor.to(device)).cpu().numpy()

    probs = rf_model.predict_proba(features)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    return pred_idx, probs[0].tolist()
