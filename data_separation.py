import os
import shutil
from sklearn.model_selection import train_test_split

# Definiujemy ścieżki do folderów
base_dir = '/home/koczka/Documents/ANN/processed_images'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Tworzenie folderów dla zestawów danych
for dir in [train_dir, val_dir, test_dir]:
    for category in ['No_findings', 'Pneumonia']:
        os.makedirs(os.path.join(dir, category), exist_ok=True)


# Definicja funkcji do podziału i zapisu obrazów
def split_and_save_images(category, train_size, val_size, test_size):
    source_dir = os.path.join(base_dir, category)
    images = [os.path.join(source_dir, img) for img in os.listdir(source_dir)]

    # Losowe podzielenie obrazów
    train_images, test_images = train_test_split(images, train_size=train_size + val_size)
    val_images, test_images = train_test_split(test_images, test_size=test_size / (val_size + test_size))

    # Funkcja pomocnicza do kopiowania obrazów
    def copy_images(images, destination):
        for img in images:
            shutil.copy(img, destination)

    # Kopiowanie obrazów do odpowiednich folderów
    copy_images(train_images, os.path.join(train_dir, category))
    copy_images(val_images, os.path.join(val_dir, category))
    copy_images(test_images, os.path.join(test_dir, category))


# Przetwarzanie i zapisywanie obrazów dla każdej kategorii
split_and_save_images('No_findings', 300, 100, 100)
split_and_save_images('Pneumonia', 300, 100, 100)

print("Podział danych zakończony.")
