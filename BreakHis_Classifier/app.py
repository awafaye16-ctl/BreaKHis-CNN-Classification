import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture exacte du modèle entraîné (identique à cnn_resnet.py)
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

@st.cache_resource
def load_model():
    try:
        model = BreastCancerResNet()
        model.load_state_dict(torch.load("models/best_model_resnet_50.pth", map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {str(e)}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx] * 100)
        return pred_idx, confidence

# Fonction Grad-CAM pour l'interprétabilité
def generate_gradcam(model, input_tensor, target_layer):
    gradients = []
    activations = []
    
    def save_gradient(grad): gradients.append(grad)
    def save_activation(act): activations.append(act)
    
    # Hook pour récupérer les gradients et les activations
    handle_forward = target_layer.register_forward_hook(lambda m, i, o: save_activation(o))
    handle_backward = target_layer.register_backward_hook(lambda m, gi, go: save_gradient(go[0]))
    
    # Forward pass
    output = model(input_tensor)
    idx = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, idx].backward()
    
    # Suppression des hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Calcul de la Heatmap
    grads = gradients[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(grads.shape[1:], dtype=np.float32)
    
    # Utilisation des activations sauvegardées
    act = activations[0].cpu().data.numpy()[0]
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, idx

# Interface Streamlit - Design Professionnel
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour style médical
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .result-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .upload-area {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# En-tête professionnel
st.markdown("""
<div class="main-header">
    <h1>🩺 Breast Cancer AI Diagnostic</h1>
    <p>Système intelligent de classification histopathologique</p>
    <p><em>Powered by ResNet-50 • Dataset BreaKHis • Explainable AI</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar avec informations
st.sidebar.markdown("## 📊 Informations Système")
st.sidebar.info(f"""
**🔧 Configuration:**
- **Processeur:** {device.type.upper()}
- **Modèle:** ResNet-50 (Pre-trained)
- **Dataset:** BreaKHis
- **Classes:** Bénin / Malin
- **Précision attendue:** >95%
""")

st.sidebar.markdown("## 🎯 Objectif")
st.sidebar.success("""
Aider au diagnostic du cancer du sein par IA tout en fournissant une interprétabilité des résultats.
""")

# Zone d'upload stylisée
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Glissez une image histopathologique ou cliquez pour parcourir", 
    type=["jpg", "jpeg", "png"],
    help="Formats supportés: JPG, JPEG, PNG - Taille maximale: 10MB"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    model = load_model()
    
    if model:
        # Affichage de l'image uploadée
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🔬 Image histopathologique analysée", width="stretch")
        
        # Traitement et prédiction avec Grad-CAM
        with st.spinner("🧠 Analyse en cours..."):
            input_tensor = preprocess_image(image)
            pred_idx, confidence = predict(model, input_tensor)
            
            # Génération Grad-CAM
            cam, _ = generate_gradcam(model, input_tensor, model.backbone.layer4[-1])
        
        classes = ['Bénin', 'Malin']
        result = classes[pred_idx]
        
        # Résultat principal
        if pred_idx == 1:  # Malin
            st.markdown(f"""
            <div class="result-positive">
                <h2>⚠️ RÉSULTAT: MALIN</h2>
                <h3>Confiance: {confidence:.1f}%</h3>
                <p><strong>Recommandation:</strong> Consultation médicale spécialisée recommandée</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # Bénin
            st.markdown(f"""
            <div class="result-negative">
                <h2>✅ RÉSULTAT: BÉNIN</h2>
                <h3>Confiance: {confidence:.1f}%</h3>
                <p><strong>Recommandation:</strong> Suivi médical régulier conseillé</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Section Grad-CAM - Visualisation
        st.markdown("## 🔬 Analyse Interprétable (Grad-CAM)")
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <p><strong>📖 Grad-CAM</strong> montre les zones de l'image qui ont influencé la décision de l'IA.</p>
            <p>🔴 <strong>Rouge/Orange:</strong> Zones importantes | 🔵 <strong>Bleu:</strong> Zones moins pertinentes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Préparation des images pour Grad-CAM
        img_resized = image.resize((224, 224))
        img_np = np.array(img_resized)
        
        # Génération de la heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # Affichage côte à côte
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h4>🔬 Image Originale</h4>
                <p style="color: #666; font-size: 0.9rem;">Histopathologie brute (224×224px)</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(img_np, width="stretch")
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h4>🌡️ Carte de Chaleur Grad-CAM</h4>
                <p style="color: #666; font-size: 0.9rem;">Zones d'intérêt pour l'IA</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(overlay, width="stretch")
        
        # Métriques détaillées
        st.markdown("## 📈 Analyse Détaillée")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🩺 Diagnostic", result)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Confiance", f"{confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if confidence >= 80:
                fiabilité = "🟢 Élevée"
            elif confidence >= 60:
                fiabilité = "🟡 Moyenne"
            else:
                fiabilité = "🔴 Faible"
            st.metric("🎯 Fiabilité", fiabilité)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚡ Temps", "< 1s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Informations techniques
        with st.expander("🔬 Détails Techniques"):
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Modèle:**
                - Architecture: ResNet-50
                - Poids: ImageNet pré-entraîné
                - Fine-tuning: BreaKHis dataset
                - Taille input: 224x224px
                """)
            with col2:
                st.info(f"""
                **Prétraitement:**
                - Normalisation: ImageNet
                - Augmentation: Non (inference)
                - Format: RGB
                - Batch size: 1
                """)
                
    else:
        st.error("❌ Erreur critique: Modèle non disponible")
else:
    # Message d'accueil
    st.markdown("""
    ## 👋 Bienvenue dans Breast Cancer AI
    
    ### 🎯 Comment utiliser:
    1. **Téléchargez** une image histopathologique (JPG/PNG)
    2. **Analysez** automatiquement avec l'IA
    3. **Obtenez** un diagnostic avec niveau de confiance
    4. **Consultez** les détails techniques
    
    ### ⚠️ Important:
    - Cet outil est une **aide au diagnostic**
    - **Ne remplace pas** l'avis d'un pathologiste
    - Usage **recherche et éducation** uniquement
    
    ### 📊 Performance:
    - **Précision:** >95% sur validation
    - **Temps de réponse:** <1 seconde
    - **Interprétabilité:** Grad-CAM disponible
    """)