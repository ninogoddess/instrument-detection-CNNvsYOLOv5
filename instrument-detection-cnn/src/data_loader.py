#frab+
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


class InstrumentDataset(Dataset):
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = images_dir
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform

        # Extraemos nombres de columnas/clases
        self.classes = list(self.annotations.columns[1:])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = row.iloc[0]
        img_path = os.path.join(self.images_dir, img_name)

        # Carga la imagen
        image = Image.open(img_path).convert("RGB")

        # Vector de etiquetas multiclase
        labels = torch.tensor(row[1:].values.astype("float32"))

        # Aplicamos transformaciones
        if self.transform:
            image = self.transform(image)

        return image, labels


# Transformaciones para entrenamiento y validación
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),                  # Redimensionar imágenes
        transforms.ToTensor(),                          # Convertir a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5],       # Normalizar canales RGB
                             std=[0.5, 0.5, 0.5])        # Rango final: [-1, 1]
    ])


# Función auxiliar para cargar dataset
def load_data(batch_size=32):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")

    train_csv = os.path.join(train_dir, "_classes.csv")
    valid_csv = os.path.join(valid_dir, "_classes.csv")

    transform = get_transforms()

    train_dataset = InstrumentDataset(train_dir, train_csv, transform=transform)
    valid_dataset = InstrumentDataset(valid_dir, valid_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


# para hacer smoke test al scrpt
# if __name__ == "__main__":
#      base_dir = "./dataset"  # Ajusta si tu estructura cambia
#      train_dataset, valid_dataset = load_datasets(base_dir)

#      print(f"Cantidad de imágenes de entrenamiento: {len(train_dataset)}")
#      print(f"Cantidad de imágenes de validación: {len(valid_dataset)}")

#      # Ver primer ítem
#      image, labels = train_dataset[0]
#      print(f"Forma de la imagen: {image.shape}")         # Ej: torch.Size([3, 224, 224])
#      print(f"Etiquetas: {labels}")                       # Ej: tensor([0., 1., 0., ...])
