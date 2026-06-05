import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# gloablny seed
from src.config import GLOBAL_SEED, set_seed
set_seed()

# stała paleta barw
MODEL_COLORS = {
    'LogisticRegression': '#D95F02',  # pomarańczowy
    'SimpleCNN':          '#7570B3',  # fioletowy
    'ResNet18':           "#9E1122",  # czerwony
    'ResNet50':           '#66A61E',  # zielony
    'MobileNetV3':        "#2892C0",  # jasnoniebieski
    'DenseNet121':        "#0F147F"   # niebieski
}

def get_paths():
    # definicja ścieżek wejścia i wyjścia
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # folder wejściowy z plikami json
    input_dir = os.path.join(base_dir, "results", "models")
    # folder wyjściowy na wykresy png
    output_dir = os.path.join(base_dir, "results", "model_charts")
    
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir

def load_all_histories(input_dir, model_names):
    # wczytywanie plików
    histories = {}
    for name in model_names:
        file_path = os.path.join(input_dir, f"{name}_screening_history.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                histories[name] = json.load(f)
        else:
            print(f"[OSTRZEŻENIE] Brak pliku historii dla: {name} pod ścieżką {file_path}")
    return histories

def plot_individual_curves(histories, output_dir):
    # wykresy dla modelu z użyciem odcieni danego koloru modelu
    print("Generowanie indywidualnych wykresów krzywych uczenia...")
    sns.set_style("whitegrid")
    
    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        base_color = MODEL_COLORS.get(model_name, '#333333')
        
        # generowanie odcieni: treningowy (jaśniejszy/pastelowy), walidacyjny (główny/nasycony)
        train_color = sns.algos.colors.alter_color(base_color, l=0.7) if hasattr(sns, 'algos') else base_color
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # wykres funkcji - loss curve
        # treningowy
        ax1.plot(epochs, history['train_loss'], color=base_color, linestyle='--', marker='o', 
                 markerfacecolor='white', alpha=0.6, label='Strata treningowa (Train Loss)', linewidth=2)
        # walidacyjny
        ax1.plot(epochs, history['val_loss'], color=base_color, linestyle='-', marker='s', 
                 label='Strata walidacyjna (Val Loss)', linewidth=2.5)
        
        ax1.set_title(f'Krzywa funkcji straty - {model_name}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoki', fontsize=10)
        ax1.set_ylabel('Wartość Loss', fontsize=10)
        ax1.set_xticks(epochs)
        ax1.legend(fontsize=10)
        
        # wykres celności - accuracy curve
        # treningowy
        ax2.plot(epochs, history['train_acc'], color=base_color, linestyle='--', marker='o', 
                 markerfacecolor='white', alpha=0.6, label='Celność treningowa (Train Acc)', linewidth=2)
        # walidacyjny
        ax2.plot(epochs, history['val_acc'], color=base_color, linestyle='-', marker='s', 
                 label='Celność walidacyjna (Val Acc)', linewidth=2.5)
        
        ax2.set_title(f'Krzywa celności - {model_name}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoki', fontsize=10)
        ax2.set_ylabel('Celność (Accuracy)', fontsize=10)
        ax2.set_xticks(epochs)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"curves_individual_{model_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [ZAPISANO] Wykres dla {model_name} -> {out_path}")

def plot_collective_comparison(histories, output_dir):
    # wykres zbiorczy
    print("Generowanie zbiorczego wykresu porównawczego Val Accuracy...")
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_acc']) + 1)
        model_color = MODEL_COLORS.get(model_name, '#333333')
        
        plt.plot(epochs, history['val_acc'], marker='o', linewidth=2.5, 
                 label=model_name, color=model_color)
        
    plt.title('Porównanie trajektorii celności walidacyjnej (Faza Screeningu)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Epoki', fontsize=11)
    plt.ylabel('Celność walidacyjna (Val Accuracy)', fontsize=11)
    plt.xticks(range(1, 6))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Architektury", title_fontsize=11, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_path = os.path.join(output_dir, "curves_collective_trajectory.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [ZAPISANO] Zbiorczy wykres trajektorii -> {out_path}")

def plot_collective_ranking(histories, output_dir):
    # wykres słypkowy rankingu końcowego
    print("Generowanie słupkowego rankingu końcowego...")
    
    models = []
    best_accs = []
    
    for model_name, history in histories.items():
        models.append(model_name)
        best_accs.append(max(history['val_acc']))
        
    # sortowanie od najlepszego
    indices = np.argsort(best_accs)[::-1]
    models = [models[i] for i in indices]
    best_accs = [best_accs[i] for i in indices]
    
    # lista kolorów posortowana
    ranking_colors = [MODEL_COLORS.get(name, '#333333') for name in models]
    
    plt.figure(figsize=(10, 5))
    sns.set_style("white")
    
    # rysowanie słupków
    barplot = sns.barplot(x=best_accs, y=models, palette=ranking_colors, hue=models, legend=False)
    
    # wartości na słupkach
    for i, val in enumerate(best_accs):
        barplot.text(val + 0.01, i, f'{val:.2%}', va='center', fontweight='bold', fontsize=10)
        
    plt.title('Końcowy ranking celności walidacyjnej (Faza 1 - Screening)', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Maksymalna uzyskana celność walidacyjna', fontsize=11)
    plt.ylabel('Architektura modelu', fontsize=11)
    plt.xlim(0, 1.1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    sns.despine(left=True, bottom=True)
    
    out_path = os.path.join(output_dir, "curves_collective_ranking.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [ZAPISANO] Słupkowy ranking końcowy -> {out_path}")

def main():
    print("=== URUCHAMIANIE GENERATORA WYKRESÓW EWALUACYJNYCH ===")
    input_dir, output_dir = get_paths()
    
    # lista 6 modeli
    model_names = [
        'LogisticRegression',
        'SimpleCNN',
        'ResNet18',
        'ResNet50',
        'MobileNetV3',
        'DenseNet121'
    ]
    
    histories = load_all_histories(input_dir, model_names)
    
    if not histories:
        print("[BŁĄD] Nie znaleziono żadnych plików historii .json w folderze results/models/. Upewnij się, że ścieżka jest poprawna!")
        return
        
    # wywołanie funkcji
    plot_individual_curves(histories, output_dir)
    plot_collective_comparison(histories, output_dir)
    plot_collective_ranking(histories, output_dir)
    print("\n=== PROCES GENEROWANIA WYKRESÓW ZAKOŃCZONY SUKCESEM ===")

if __name__ == "__main__":
    main()