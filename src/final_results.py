import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Ustawienie ścieżek
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=== GENEROWANIE KOŃCOWYCH WYKRESÓW TRAJEKTORII UCZENIA ===")
    
    # 1. RĘCZNE WPROWADZENIE WYNIKÓW Z TERMINALU (Oszczędność czasu - bez uczenia od nowa!)
    # Przepisujemy dokładnie to, co wypluł Wasz genialny trening dla epok 1-6
    epochs = [1, 2, 3, 4, 5, 6]
    
    train_loss = [0.2442, 0.0757, 0.0473, 0.0341, 0.0256, 0.0214]
    val_loss   = [0.0917, 0.2406, 0.1234, 0.0385, 0.0540, 0.0508]
    
    train_acc  = [0.9211, 0.9752, 0.9840, 0.9885, 0.9919, 0.9930]
    val_acc    = [0.9706, 0.9738, 0.9838, 0.9884, 0.9865, 0.9887]
    
    # Ustawienie estetycznego stylu wykresów
    sns.set_theme(style="whitegrid")
    
    # --- WYKRES 1: FUNKCJA STRATY (LOSS CURVE) ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, train_loss, label="Strata treningowa (Train Loss)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
    plt.plot(epochs, val_loss, label="Strata walidacyjna (Val Loss)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
    
    plt.title("Krzywa funkcji straty - Zoptymalizowany MobileNetV3 (Faza 2)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoki", fontsize=12)
    plt.ylabel("Wartość Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    
    loss_path = os.path.join(plots_dir, "final_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"[OK] Wykres funkcji straty zapisany w: {loss_path}")
    
    # --- WYKRES 2: CELNOŚĆ (ACCURACY CURVE) ---
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, train_acc, label="Celność treningowa (Train Acc)", color="#A0C4DF", linestyle="--", marker="o", linewidth=2)
    plt.plot(epochs, val_acc, label="Celność walidacyjna (Val Acc)", color="#0B3C5D", linestyle="-", marker="s", linewidth=2.5)
    
    # Formatowanie osi Y jako procenty
    ax = plt.gca()
    ax.set_yticklabels([f"{x*100:.0f}%" for x in ax.get_yticks()])
    
    plt.title("Krzywa celności - Zoptymalizowany MobileNetV3 (Faza 2)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoki", fontsize=12)
    plt.ylabel("Celność (Accuracy)", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    
    acc_path = os.path.join(plots_dir, "final_accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"[OK] Wykres celności zapisany w: {acc_path}")
    
    print("\n=== PROCES GENEROWANIA WYKRESÓW ZAKOŃCZONY SUKCESEM ===")

if __name__ == "__main__":
    main()