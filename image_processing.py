import os
from PIL import Image
import numpy as np

# Definicje ścieżek
folders = ['No_findings', 'Pneumonia']
base_dir = '/home/koczka/Documents/ANN/'
processed_dir = os.path.join(base_dir, 'processed_images')

# Tworzenie nowego folderu dla przetworzonych obrazów
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Przetwarzanie obrazów z obu folderów
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    processed_folder = os.path.join(processed_dir, folder)

    # Tworzenie folderów dla przetworzonych obrazów
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    # Iteracja przez obrazy i ich przetwarzanie
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(processed_folder, filename)

            # Wczytanie obrazu i przetworzenie go
            img = Image.open(input_path)
            img = img.resize((128, 128))  # Zmiana rozmiaru na 128x128 pikseli
            img_array = np.array(img) / 255.0  # Normalizacja do zakresu [0, 1]

            # Zapisywanie przetworzonego obrazu
            if img_array.shape[-1] == 4:  # Sprawdza, czy obraz jest w trybie RGBA
                # Konwersja do RGB, jeśli obraz ma 4 kanały (RGBA)
                img_array = img_array[..., :3]
            processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
            processed_img.save(output_path)
print("Przetwarzanie zakończone.")
