import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# importy z modułów
from src.config import GLOBAL_SEED, set_seed
from src.data_loader import load_and_prepare_data
from src.models import model_factory

def train_one_epoch(model, loader, criterion, optimizer, device):
    # trening przez 1 epokę
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="   Uczenie", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    # walidacja przez 1 epokę
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_screening_phase(epochs=5, batch_size=128, base_lr=0.001):
    # trenowanie 6 modeli na ujednoliconych hiperparametrach
    # global seed i losowość
    set_seed(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== URUCHAMIANIE FAZY 1: SCREENING (Benchmark 6 modeli) ===")
    print(f"Urządzenie obliczeniowe: {device}\n")
    
    # foldery na wyniki
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_out_dir = os.path.join(base_dir, "models")
    results_out_dir = os.path.join(base_dir, "results", "models")
    os.makedirs(models_out_dir, exist_ok=True)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # załadowanie potoku danych
    train_dataset, val_dataset, _ = load_and_prepare_data()
    
    # funkcja dla powtarzalności
    def seed_worker(worker_id):
        set_seed(GLOBAL_SEED)
        
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # lista 6 modeli do przetestowania 
    models_to_test = [
        'LogisticRegression',
        'SimpleCNN',
        'ResNet18',
        'ResNet50',
        'MobileNetV3',
        'DenseNet121'
    ]
    
    # główna pętla testu modeli
    for model_name in models_to_test:
        print(f"\n>>> Rozpoczęcie eksperymentu dla architektury: {model_name}")
        
        # pobranie modeli
        model = model_factory(model_name, num_classes=9, pretrained=True).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        
        # historia uczenia
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            
            # zapis do historii
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"   Epoka {epoch+1:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # zapis najlepszej iteracji na podstawie zbioru walidacyjnego
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_path = os.path.join(models_out_dir, f"{model_name}_screening_best.pth")
                torch.save(model.state_dict(), weight_path)
                
        print(f"[ZAKOŃCZONO] {model_name} -> Najlepszy Val Accuracy: {best_val_acc:.4f}")
        
        # zapis historii uczenia do pliku JSON
        history_path = os.path.join(results_out_dir, f"{model_name}_screening_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"[ZAPISANO] Historia uczenia trafia do: {history_path}")
        
        # czyszczenie pamięci podręcznej GPU przed kolejną architekturą
        del model, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n=== FAZA 1 ZAKOŃCZONA SUKCESEM ===")
    print("Wszystkie 6 modeli zostało przetestowanych. Wagi oraz historie uczenia zostały zapisane.")

if __name__ == '__main__':
    # uruchomienie na 5 epokach
    run_screening_phase(epochs=5, batch_size=128, base_lr=0.001)