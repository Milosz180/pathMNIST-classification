import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# import ładowania danych
from src.data_loader import load_and_prepare_data

def save_samples():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_script_dir, "..", "samples"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== EKSTRAKCJA 9 UNIKALNYCH PRÓBEK TESTOWYCH DO GUI ===")
    print(f"Katalog docelowy: {output_dir}\n")
    
    # ładowanie danych
    _, _, test_dataset = load_and_prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    target_classes = {
        0: "Klasa_0_Tkanka_tluszczowa",
        1: "Klasa_1_Tlo_preparatu",
        2: "Klasa_2_Geste_podscielisko",
        3: "Klasa_3_Nacieki_limfocytarne",
        4: "Klasa_4_Gruczoły_jelitowe_prawidlowe",
        5: "Klasa_5_Rak_gruczolowy",
        6: "Klasa_6_Tkanka_limfoidalna",
        7: "Klasa_7_Miesniowka_gladka",
        8: "Klasa_8_Prawidlowa_sluzowka"
    }
    
    saved_classes = set()
    current_timestamp = int(time.time())
    
    for images, labels in test_loader:
        if len(labels.shape) > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
            
        true_label = int(labels.item())
        
        if true_label in target_classes and true_label not in saved_classes:
            class_prefix = target_classes[true_label]
            file_name = f"{class_prefix}_{current_timestamp}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # odwrócenie normalizacji do zapisu czystego pliku graficznego H&E
            img_to_save = images.squeeze(0).clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_to_save = img_to_save * std + mean
            img_to_save = torch.clamp(img_to_save, 0, 1)
            
            save_image(img_to_save, file_path)
            print(f"[ZAPISANO KLASĘ {true_label}] -> {file_name}")
            saved_classes.add(true_label)
            
        if len(saved_classes) == len(target_classes):
            break
            
    print(f"\n=== PROCES ZAKOŃCZONY. Obrazki znajdują się w: {output_dir} ===")

if __name__ == "__main__":
    save_samples()