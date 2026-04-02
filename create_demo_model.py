import torch
import torch.nn as nn
from torchvision import models
import os

class BreastCancerResNet(nn.Module):
    def __init__(self):
        super(BreastCancerResNet, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)

def create_demo_model():
    """Crée un modèle de démonstration pour l'application Streamlit"""
    
    # Créer le modèle
    model = BreastCancerResNet()
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    # Sauvegarder uniquement le state_dict
    torch.save(model.state_dict(), 'models/best_model_resnet_50.pth')
    
    print("✅ Modèle de démonstration créé avec succès!")
    print("📁 Fichier sauvegardé: models/best_model_resnet_50.pth")
    print("🎯 Le modèle est prêt pour l'application Streamlit")

if __name__ == "__main__":
    create_demo_model()
