import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import global seeda i funkcji ustawiającej
from src.config import GLOBAL_SEED, set_seed
from src.models import model_factory
from src.data_loader import load_and_prepare_data

# inicjalizacja środowiska
set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 15
PATIENCE = 3  # ilość epok bez poprawy stopująca i cofająca proces trenowania

def load_best_hyperparameters():
    # wczytanie parametrów wybranych przez Optunę
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "results", "models", "hyperparameters", "mobilenetv3_best_hyperparameters.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Brak pliku konfiguracji z Fazy 2 pod ścieżką: {config_path}. Uruchom najpierw tuning!")
        
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # mapowanie katalogów
    output_dir = os.path.join(base_dir, "models")
    final_model_dir = os.path.join(base_dir, "results", "final_model")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    
    print("=== START FAZY 3: TRENING FINALNY (OBLICZENIA I ZAPIS) ===")
    
    # wczytanie parametrów i danych
    hyperparams = load_best_hyperparameters()
    print(f"[CONFIG MASTER]: {hyperparams}")
    
    train_dataset, val_dataset, _ = load_and_prepare_data()
    
    batch_size = hyperparams["batch_size"]
    use_cuda = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda)
    
    # budowa nowego, czystego modelu i konfiguracja optymalizatora
    model = model_factory('MobileNetV3', num_classes=9, pretrained=True)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    lr = hyperparams["lr"]
    
    if hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif hyperparams["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    # historia do wykresów
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    # zmienne dla Early Stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "mobilenetv3_final_optimized.pth")
    
    # pętla treningowa z Early Stopping
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if len(labels.shape) > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # faza Walidacji
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if len(labels.shape) > 1 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # zapis do historii
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoka {epoch+1:02d}/{MAX_EPOCHS} | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2%} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2%}")
        
        # logika Wczesnego Zatrzymania (Early Stopping)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   [ZAPIS] Osiągnięto nową najwyższą celność walidacyjną. Zapisano checkpoint.")
        else:
            patience_counter += 1
            print(f"   [EARLY STOPPING] Brak poprawy. Licznik cierpliwości: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n[URUCHOMIONO ANULOWANIE] Aktywowano Early Stopping w epoce {epoch+1}. Przerywanie uczenia.")
                break
                
    # zapis historii do pliku JSON
    history_path = os.path.join(final_model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[SUKCES] Historia treningu została pomyślnie odłożona w: {history_path}")

if __name__ == "__main__":
    main()