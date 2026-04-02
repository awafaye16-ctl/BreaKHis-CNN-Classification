import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Configuration du matériel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Chargement du modèle ResNet-50 (Cohérent avec ton CV)
@st.cache_resource
def load_model():
    # Architecture exacte du projet BreaKHis
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2) # 2 classes : Bénin, Malin
    )
    # Chargement des poids haute précision
    model.load_state_dict(torch.load("models/best_model_resnet_50.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# 3. Fonction Grad-CAM pour l'interprétabilité
def generate_gradcam(model, input_tensor, target_layer):
    # Hook pour récupérer les gradients et les activations
    gradients = []
    def save_gradient(grad): gradients.append(grad)
    
    # On cible la dernière couche de convolution du ResNet
    handle = target_layer.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    target_layer.register_backward_hook(lambda m, gi, go: save_gradient(go[0]))
    
    # Forward pass
    output = model(input_tensor)
    idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, idx].backward()
    
    # Calcul de la Heatmap
    grads = gradients[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(grads.shape[1:], dtype=np.float32)
    
    # On récupère les activations de la couche cible
    target_layer_output = target_layer.forward(input_tensor).cpu().data.numpy()[0]
    for i, w in enumerate(weights):
        cam += w * target_layer_output[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, idx

# 4. Interface Streamlit
st.set_page_config(page_title="Breast Cancer AI", page_icon="🩺")
st.title("🩺 Breast Cancer Analysis (ResNet-50 + Grad-CAM)")

uploaded_file = st.file_uploader("Charger une biopsie (BreaKHis)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    model = load_model()
    img = Image.open(uploaded_file).convert("RGB")
    
    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Prédiction et Grad-CAM
    cam, pred_idx = generate_gradcam(model, input_tensor, model.layer4[-1])
    
    # Affichage des résultats
    classes = ['Bénin', 'Malin']
    color = "red" if pred_idx == 1 else "green"
    
    st.subheader(f"Résultat : :{color}[{classes[pred_idx].upper()}]")
    
    # Superposition de la Heatmap (Visualisation)
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Image Originale", use_container_width=True)
    with col2:
        st.image(overlay, caption="Interprétabilité (Grad-CAM)", use_container_width=True)