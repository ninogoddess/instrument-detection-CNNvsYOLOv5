import subprocess
import os

print("Ejecutando entrenamiento YOLOv5...")

# Ruta base del proyecto
base_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas
yolov5_path = os.path.join(base_dir, "../yolov5/train.py")
data_yaml = os.path.join(base_dir, "../dataset/data.yaml")
hyp_yaml = os.path.join(base_dir, "../yolov5/data/hyps/hyp.scratch-med.yaml")

# Comando de entrenamiento
command = [
    "python", yolov5_path,
    "--img", "640",
    "--batch", "8",                     # Aumenta si tienes VRAM disponible
    "--epochs", "50",
    "--data", data_yaml,
    "--weights", "yolov5s.pt",
    "--name", "instruments_yolo_rtx2060",
    "--hyp", hyp_yaml,
    "--device", "0",                    # Forzar uso de GPU CUDA
    "--workers", "2"                    # Seguro para HDD
]

# Ejecutar el comando
subprocess.run(command, check=True)
