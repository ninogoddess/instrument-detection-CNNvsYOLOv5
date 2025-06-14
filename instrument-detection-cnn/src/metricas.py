# frab+

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from model import InstrumentCNN
from data_loader import InstrumentDataset, get_transforms

# === CONFIGURAR RUTAS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
valid_dir = os.path.abspath(os.path.join(script_dir, "../dataset/valid"))
valid_csv = os.path.join(valid_dir, "_classes.csv")

# === TRANSFORMACIONES Y DATASET ===
transform = get_transforms()
val_dataset = InstrumentDataset(valid_dir, valid_csv, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === CARGAR MODELO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(script_dir, "instrument_cnn.pth")
model = InstrumentCNN(num_classes=len(val_dataset.classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === EVALUACIÓN ===
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        # Convertir etiquetas one-hot a índice si es necesario
        if labels.ndim > 1 and labels.size(1) > 1:
            labels = labels.argmax(dim=1)

        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === MÉTRICAS ===
print("== Classification Report ==")
print(classification_report(y_true, y_pred, target_names=val_dataset.classes))

# === PRECISION, RECALL Y F1 MACRO ===
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Obtener métricas por clase (sin promediar)
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

clases = val_dataset.classes
indices = np.arange(len(clases))

# Función para graficar cada métrica
def plot_metric(values, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(indices, values, marker='o', linestyle='-', color='b')
    plt.xticks(indices, clases, rotation=45, ha='right')
    plt.xlabel("Clases")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Graficar Precision por clase
plot_metric(precision_per_class, "Precisión por Clase", "Precisión", "precision_por_clase.png")

# Graficar Recall por clase
plot_metric(recall_per_class, "Recall por Clase", "Recall", "recall_por_clase.png")

# Graficar F1-score por clase
plot_metric(f1_per_class, "F1-score por Clase", "F1-score", "f1_por_clase.png")

# # === MATRIZ DE CONFUSIÓN ===
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d",
#             xticklabels=val_dataset.classes,
#             yticklabels=val_dataset.classes,
#             cmap="Blues")
# plt.xlabel("Predicción")
# plt.ylabel("Etiqueta real")
# plt.title("Matriz de Confusión CNN")
# plt.tight_layout()
# plt.savefig("cnn_confusion_matrix.png")
# plt.show()
