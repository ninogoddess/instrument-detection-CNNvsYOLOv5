#frab+
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import sys

# Ruta relativa al modelo
model_dir = os.path.abspath(os.path.join("..", "src"))
model_path = os.path.join(model_dir, "instrument_cnn.pth")

# Agregar src al sys.path para importar InstrumentCNN
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from model import InstrumentCNN
from data_loader import get_transforms

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar clases
CLASSES = ['accordion','acoustic-guitar','cello','clarinet',
           'erhu','flute','saxophone','trumpet','tuba','violin','xylophone']


# Cargar el modelo CNN
@st.cache_resource
def load_model():
    model = InstrumentCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()
transform = get_transforms()

# Título de la app
st.title("Detección de instrumentos con CNN")

st.markdown("### Descripción de la Aplicación:")
st.write(
    "Esta aplicación utiliza una red neuronal convolucional (CNN) entrenada para detectar múltiples instrumentos musicales "
    "en imágenes. Puedes subir una imagen, y el modelo predecirá qué instrumentos están presentes, mostrando la certeza asociada "
    "a cada clase detectada. Las predicciones no dibujan recuadros, ya que este modelo clasifica en vez de localizar visualmente."
)

# Subida de imagen
uploaded_file = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Imagen cargada', use_container_width=True)

    # Preprocesamiento y predicción
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = output.squeeze().cpu().numpy()

    # Mostrar resultados
    st.subheader("Instrumentos detectados:")
    for idx, prob in enumerate(probs):
        if prob > 0.3:  # umbral para considerar "presente"
            st.write(f"{CLASSES[idx]} — Certeza: {prob:.2f}")
    
    if all(prob <= 0.3 for prob in probs):
        st.info("No se detectaron instrumentos con certeza significativa.")
