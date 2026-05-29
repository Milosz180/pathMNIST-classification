import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# globalny random state
GLOBAL_SEED = 42

def set_seed(seed=GLOBAL_SEED):
    # stałe ziarno losowości dla wszystkich bibliotek
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# klasa datasetu
class PathMNIST64Dataset(Dataset):
    # klasa do obługi obrazów z PATHMnist
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        # numer klasy jako lioczba całkowita
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return img, label


# sprawdzenie balansu klas
def print_class_balance(labels, dataset_part_name):
    # rozkład klas
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    print(f"\n Rozkład klas dla podzbioru: {dataset_part_name} (Suma: {total})")
    print("-" * 50)
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"Klasa {cls}: {count:6d} obrazów ({percentage:.2f}%)")

def load_and_prepare_data():
    # wczytanie danych, wyświetlanie rozkładu danych
    # ścieżka do pliku z danymi
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "pathmnist_64.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku danych pod ścieżką: {data_path}. "
            f"Upewnij się, że plik ma nazwę 'pathmnist_64.npz' i znajduje się w folderze data/."
        )
        
    # wczytanie pliku z danymi
    print(f"Wczytywanie danych z pliku: {data_path}...")
    data = np.load(data_path)
    
    # wyciąganie tablicy numpy
    train_images, train_labels = data['train_images'], data['train_labels']
    val_images, val_labels = data['val_images'], data['val_labels']
    test_images, test_labels = data['test_images'], data['test_labels']
    
    # wyświetlanie balansu klas
    print_class_balance(train_labels, "Treningowy (Train)")
    print_class_balance(val_labels, "Walidacyjny (Val)")
    print_class_balance(test_labels, "Testowy (Test)")
    
    # 3. transformacja
    # odpowiednik Z-Score dla obrazów medycznych: (piksel - mean) / std
    # wykorzystanie standardowych dla obrazów kolorowych wartości średnich i odchyleń
    base_transform = transforms.Compose([
        transforms.ToTensor(),  # konwersja 0-255 do przedziału 0.0-1.0 (Odpowiednik Min-Max)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # tworzenie obiektów dataset
    train_dataset = PathMNIST64Dataset(train_images, train_labels, transform=base_transform)
    val_dataset = PathMNIST64Dataset(val_images, val_labels, transform=base_transform)
    test_dataset = PathMNIST64Dataset(test_images, test_labels, transform=base_transform)
    
    return train_dataset, val_dataset, test_dataset


# uruchomienie pliku i sprawdzenie klas
if __name__ == "__main__":
    print("=== Uruchomienie testowe potoku danych (Część 1) ===")
    try:
        train_ds, val_ds, test_ds = load_and_prepare_data()
        print("\n[SUKCES] Potok danych zainicjalizowany poprawnie.")
        print(f"Liczba próbek w Datasetach: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    except Exception as e:
        print(f"\n[BŁĄD] Coś poszło nie tak podczas wczytywania danych:\n{e}")