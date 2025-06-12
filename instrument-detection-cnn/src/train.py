#frab+
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_data
from model import InstrumentCNN
from tqdm import tqdm
import os

# Configuración general
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 11  # acorde al CSV

def train_model():
    # Cargar datos
    train_loader, valid_loader = load_data(batch_size=BATCH_SIZE)

    # Inicializar modelo
    model = InstrumentCNN(num_classes=NUM_CLASSES).to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.BCELoss()  # Binary Cross-Entropy para multietiqueta
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        # Barra de progreso con tqdm para el entrenamiento
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (progress_bar.n if progress_bar.n > 0 else 1)
            progress_bar.set_postfix(loss=avg_loss)

        avg_train_loss = running_loss / len(train_loader)
        print(f"Época [{epoch+1}/{NUM_EPOCHS}] finalizada, Pérdida entrenamiento: {avg_train_loss:.4f}")

        # Validación
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"           Pérdida validación: {avg_valid_loss:.4f}\n")

    # Guardar el modelo
    model_path = os.path.join(os.path.dirname(__file__), "instrument_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Entrenamiento completado. Modelo guardado como: {model_path}")

if __name__ == "__main__":
    train_model()
