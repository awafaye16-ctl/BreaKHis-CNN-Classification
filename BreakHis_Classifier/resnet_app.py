import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.set_page_config(page_title="Breast Cancer Classifier (ResNet-50)", layout="wide")
st.title("Breast Cancer Classification (Bénin vs Malin)")

class BreastCancerResNet(nn.Module):
    def __init__(self):
        super(BreastCancerResNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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

@st.cache_resource
def load_model():
    model = BreastCancerResNet()
    model_path = os.path.join("..", "models", "best_model_resnet_50.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "best_model_resnet_50.pth")

    if not os.path.exists(model_path):
        st.error(
            "Aucun modèle trouvé dans 'models/best_model_resnet_50.pth'. "
            "Exécutez cnn_resnet.py et placez le checkpoint dans le dossier models/"
        )
        st.stop()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=ResNet50_Weights.IMAGENET1K_V1.transforms().mean,
                         std=ResNet50_Weights.IMAGENET1K_V1.transforms().std)
])

st.sidebar.header("Model Overview")
st.sidebar.markdown("**Backbone:** ResNet-50 (ImageNet pretrained)")
st.sidebar.markdown("**Head:** 256 + Dropout(0.4) + 2 classes")
st.sidebar.markdown("**Seuil confiance:** 70%")

uploaded_file = st.file_uploader("Upload une image histopathologique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Vérification du type de fichier
        if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            st.error("Format non supporté. Utilisez JPG, JPEG ou PNG.")
            st.stop()
        
        # Chargement robuste de l'image
        img = Image.open(uploaded_file)
        
        # Conversion en RGB avec gestion d'erreurs
        if img.mode != 'RGB':
            try:
                img = img.convert('RGB')
            except Exception as e:
                st.error(f"Erreur de conversion d'image: {str(e)}")
                st.stop()
        
        # Validation de l'image
        img.verify()  # Vérifie l'intégrité de l'image
        uploaded_file.seek(0)  # Reset le curseur après verify
        img = Image.open(uploaded_file).convert('RGB')
        
        st.image(img, caption="Image d'entrée", use_column_width=True)

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        classes = ["Bénin", "Malin"]
        label = classes[pred_idx]
        confidence = float(probs[pred_idx] * 100)

        if confidence < 70.0:
            st.warning(f"Confiance faible : {confidence:.2f}% (<70%). Vérifier manuellement.")
        else:
            st.success(f"Pred: {label} ({confidence:.2f}% confiance)")

        # Grad-CAM
        target_layer = model.backbone.layer4[-1]
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        model.zero_grad()
        output = model(img_tensor)
        pred_score = output[0, pred_idx]
        pred_score.backward(retain_graph=False)

        handle_f.remove()
        handle_b.remove()

        grad = gradients[0].detach().cpu().numpy()[0]
        act = activations[0].detach().cpu().numpy()[0]
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        cam = cv2.resize(cam, (img.size[0], img.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        st.subheader("Grad-CAM")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_column_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

        st.markdown(f"**Classe** : {label}  \
**Confiance** : {confidence:.2f}%")

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image: {str(e)}")
        st.info("Veuillez vérifier que le fichier est une image valide (JPG, JPEG, PNG) et réessayer.")
else:
    st.info("Téléversez une image pour démarrer la prédiction.")
