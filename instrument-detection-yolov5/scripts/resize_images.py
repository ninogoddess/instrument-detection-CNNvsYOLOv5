import os
from PIL import Image
from tqdm import tqdm

# Ruta a la carpeta con las imágenes
# Si este script está en scripts/, esto apunta a ../dataset/train/images
IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'valid', 'images')
TARGET_SIZE = (640, 640)

def resize_images(image_dir, size):
    supported_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_exts)]

    print(f"Redimensionando {len(images)} imágenes en: {image_dir}")
    for image_name in tqdm(images, desc="Redimensionando imágenes"):
        img_path = os.path.join(image_dir, image_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Asegura que no haya problemas de modo
                resized_img = img.resize(size, Image.Resampling.LANCZOS)
                resized_img.save(img_path)  # Sobrescribe la imagen original
        except Exception as e:
            print(f"Error procesando {image_name}: {e}")

if __name__ == '__main__':
    resize_images(IMAGE_DIR, TARGET_SIZE)
