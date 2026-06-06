import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import shap

# import global seed i funkcji z innych modułów
from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# klasa podstawowa dla grad-cam, wyciąganie gradientów
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # jeśli nie podana klasa, bierzemy klasę z najwyższym wynikiem
        if target_class is None:
            target_class = torch.argmax(output)
            
        loss = output[0, target_class]
        loss.backward()

        # Global Average Pooling dla gradientów
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # liniowa kombinacja map aktywacji i wag gradientów
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # funkcja ReLU
        cam = np.maximum(cam.cpu().numpy(), 0)
        
        # normalizacja do przedziału [0, 1]
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # zmiana rozmiaru mapy do wymiarów wejściowego obrazu
        spatial_size = (input_tensor.shape[2], input_tensor.shape[3])
        heatmap = cv2.resize(cam, spatial_size)
        return heatmap

# funkcje pomocnicze do wizualizacji i selekcji przypadków
def overlay_heatmap(img_tensor, heatmap):
    """ Nakłada mapę ciepła na oryginalny obraz histopatologiczny """
    img = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    # odwrócenie normalizacji ImageNet na potrzeby poprawnego wyświetlania RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # konwersja heatmapy na kolory JET
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = np.float32(heatmap_color) / 255.0
    
    # połączenie obrazu z heatmapą, przezroczystość 0,5
    overlay = 0.5 * img + 0.5 * heatmap_color
    return np.clip(overlay, 0, 1), img

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "final_model", "mobilenetv3_final_optimized.pth")
    thresholds_path = os.path.join(base_dir, "models", "final_model", "medical_thresholds.json")
    explain_dir = os.path.join(base_dir, "results", "final_model", "explain")
    os.makedirs(explain_dir, exist_ok=True)
    
    print("=== START: FINAL_EXPLAIN (WARSTWA WYJAŚNIALNOŚCI MODELU XAI) ===")
    
    if not os.path.exists(model_path) or not os.path.exists(thresholds_path):
        raise FileNotFoundError("Brak pliku modelu (.pth) lub medycznych progów (.json) w katalogu models/!")

    # wczytanie progów Youdena
    with open(thresholds_path, "r") as f:
        thresholds = {int(k): v for k, v in json.load(f).items()}
        
    # ładowanie danych testowych i modelu
    _, _, test_dataset = load_and_prepare_data()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # batch_size=1 dla precyzyjnej analizy przypadków
    
    model = model_factory('MobileNetV3', num_classes=9, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    
    # namierzenie ostatniej warstwy splotowej w MobileNetV3
    target_layer = model.model.features[-1]
    cam_extractor = GradCAM(model, target_layer)
    
    # kontenery na 3 unikalne przypadki badawcze
    case_studies = {"true": None, "threshold": None, "false": None}
    class_names = [f"Klasa {i}" for i in range(9)]
    
    # przeszukiwanie zbioru
    print("[XAI] Przeszukiwanie zbioru testowego pod kątem ciekawych przypadków klinicznych...")
    model.eval()
    
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if len(labels.shape) > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
            
        true_label = labels.item()
        
        with torch.no_grad():
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            pred_argmax = np.argmax(outputs.cpu().numpy(), axis=1)[0]
            
            # decyzja zoptymalizowana medycznie
            scaled_probs = [probs[i] / thresholds[i] for i in range(9)]
            pred_youden = np.argmax(scaled_probs)
            
        # przypadek 1 - poprawna klasyfikacja w klasach trudnych
        if true_label == pred_youden and true_label == pred_argmax and probs[true_label] > 0.95 and case_studies["true"] is None:
            if true_label in [5, 7]:
                case_studies["true"] = (images.clone(), true_label, pred_youden, probs.copy(), "Pewny (Prawdziwie Pozytywny)")
                
        # przypadek 2 - poprawna klasyfikacja po optymalizacji z Youdenem w klasach trudnych
        if true_label == pred_youden and true_label != pred_argmax and case_studies["threshold"] is None:
            case_studies["threshold"] = (images.clone(), true_label, pred_youden, probs.copy(), "Uratowany przez próg Youdena")
            
        # przypadek 3 - błędna klasyfikacja w klasach trudnych
        if true_label != pred_youden and case_studies["false"] is None:
            if true_label in [5, 7]:
                case_studies["false"] = (images.clone(), true_label, pred_youden, probs.copy(), f"Błędna klasyfikacja (Faktyczna: {true_label}, Predykcja: {pred_youden})")
                
        if all(v is not None for v in case_studies.values()):
            break

    # 4. generowanie i zapis wyników
    print("\n[XAI] Generowanie wizualizacji Grad-CAM oraz analizy SHAP...")
    
    # inicjalizacja SHAP Background dla ImageExplainer
    def model_predict_for_shap(img_np):
        # funkcja adapter pod bibliotekę SHAP
        img_torch = torch.tensor(img_np).permute(0, 3, 1, 2).float().to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(img_torch)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    shap_explainer = shap.Explainer(model_predict_for_shap, shap.maskers.Image("inpaint_telea", (64, 64, 3)))
    
    for case_key, case_data in case_studies.items():
        if case_data is None:
            print(f"[WARN] Nie udało się automatycznie wyodrębnić przypadku dla kategorii: {case_key}")
            continue
            
        img_tensor, true_cls, pred_cls, probs_vector, description = case_data
        print(f"\n-> Przetwarzanie przypadku: {description}")
        
        # wywołanie Grad-CAM
        model.eval()
        heatmap = cam_extractor.generate_heatmap(img_tensor, target_class=pred_cls)
        gradcam_result, original_img = overlay_heatmap(img_tensor, heatmap)
        
        # wywołanie SHAP dla tego samego obrazu
        img_np_shap = original_img.copy()
        # wyznaczanie wartości Shapleya dla wygranej klasy predykcyjnej
        shap_values = shap_explainer(np.expand_dims(img_np_shap, axis=0), max_evals=300, batch_size=50, outputs=[pred_cls])
        
        # zbiorczy wykres
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        
        # podwykres 1 - oryginalny obraz histopatologiczny
        axes[0].imshow(original_img)
        axes[0].set_title(f"Oryginał (Faktyczna: Klasa {true_cls})\nPrawd. surowe: {probs_vector[pred_cls]*100:.2f}%", fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # podwykres 2 - apa ciepła Grad-CAM
        axes[1].imshow(gradcam_result)
        axes[1].set_title(f"Grad-CAM (Wyjaśnienie wizualne)\nCel: Klasa {pred_cls}", fontsize=10, fontweight='bold')
        axes[1].axis('off')
        
        # podwykres 3 - lokalne wyjaśnienie SHAP
        val_shap = shap_values.values[0, ..., 0] 
        abs_max = max(np.abs(val_shap.min()), np.abs(val_shap.max()))
        if abs_max == 0: abs_max = 1
        
        axes[2].imshow(original_img, alpha=0.6)
        shap_im = axes[2].imshow(val_shap, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, alpha=0.7)
        fig.colorbar(shap_im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].set_title(f"SHAP Local Plot (Wpływ cech)\nCzerwony (+) | Niebieski (-)", fontsize=10, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f"Analiza XAI - Przypadek: {description.upper()}", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # zapis końcowy grafiki panelowej
        save_path = os.path.join(explain_dir, f"xai_analysis_{case_key}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[SUKCES] Kompletny panel XAI zapisany w: {save_path}")
        
    print("\n=== SYSTEM WYJAŚNIALNOŚCI MODELU ZAKOŃCZYŁ PRACĘ Z SUKCESEM ===")

if __name__ == "__main__":
    main()