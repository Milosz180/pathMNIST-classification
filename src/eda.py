import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# słownik z pełnymi nazwami
CLASS_NAMES = {
    0: "Adipose (ADI)",
    1: "Background (BACK)",
    2: "Debris (DEB)",
    3: "Lymphocytes (LYM)",
    4: "Mucus (MUC)",
    5: "Smooth Muscle (MUS)",
    6: "Normal Colon Mucosa (NORM)",
    7: "Cancer-Associated Stroma (STR)",
    8: "Colorectal Adenocarcinoma Epithelium (TUM)"
}

def plot_sample_gallery(images, labels, save_dir):
    # 3 próbki na każdą klasę  
    print("Generowanie galerii próbek morfologicznych tkanek...")
    num_classes = 9
    samples_per_class = 3
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 20))
    
    # stały seed
    np.random.seed(42)
    
    for cls in range(num_classes):
        # indeksy dla danej kalsy
        cls_indices = np.where(labels == cls)[0]
        # losowanie 3 próbek
        chosen_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
        
        for idx, img_idx in enumerate(chosen_indices):
            ax = axes[cls, idx]
            ax.imshow(images[img_idx])
            ax.axis('off')
            
            # dodanie nazwy klasy
            if idx == 0:
                ax.text(-10, 32, CLASS_NAMES[cls], 
                        fontsize=12, fontweight='bold', ha='right', va='center')
                
    plt.tight_layout()
    output_path = os.path.join(save_dir, "eda_sample_gallery.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[ZAPISANO] Galeria próbek: {output_path}")


def plot_rgb_intensity_distribution(images, save_dir):
    # analiza dlobalnego rozkładu intensywności pikseli dla kanałów R, G, B
    print("Analiza rozkładu intensywności pikseli w kanałach RGB...")
    
    # pobieranie losowej próbki 5000 obrazów
    np.random.seed(42)
    sample_indices = np.random.choice(len(images), min(5000, len(images)), replace=False)
    sampled_images = images[sample_indices]
    
    # wyciąganie średnich intensywności dla kanałów
    r_channel = sampled_images[:, :, :, 0].flatten()
    g_channel = sampled_images[:, :, :, 1].flatten()
    b_channel = sampled_images[:, :, :, 2].flatten()
    
    plt.figure(figsize=(12, 5))
    sns.set_style("whitegrid")
    
    # linie gęstości (KDE) dla każdego koloru
    sns.kdeplot(r_channel, color='red', label='Kanał Czerwony (R)', linewidth=2)
    sns.kdeplot(g_channel, color='green', label='Kanał Zielony (G)', linewidth=2)
    sns.kdeplot(b_channel, color='blue', label='Kanał Niebieski (B)', linewidth=2)
    
    plt.title("Rozkład intensywności pikseli w kanałach RGB (Barwienie H&E)", fontsize=14, pad=15)
    plt.xlabel("Wartość piksela (0 - 255)", fontsize=12)
    plt.ylabel("Gęstość rozkładu (Density)", fontsize=12)
    plt.xlim(0, 255)
    plt.legend(fontsize=11)
    
    output_path = os.path.join(save_dir, "eda_rgb_distribution.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[ZAPISANO] Wykres rozkładu RGB: {output_path}")

def plot_brightness_boxplot(images, labels, save_dir):
    # obliczanie średniej jasności dla każdego obrazu, box-plot dla 9 klas
    print("Generowanie wykresu pudełkowego jasności dla klas...")
    
    # konwersja obrazów RGB do skali szarości za pomocą standardowych wag luminancji
    # Y = 0.299*R + 0.587*G + 0.114*B
    grayscale_images = (
        0.299 * images[:, :, :, 0] + 
        0.587 * images[:, :, :, 1] + 
        0.114 * images[:, :, :, 2]
    )
    
    # średnia jasność dla pojedynczego zdjęcia
    mean_brightness = grayscale_images.mean(axis=(1, 2))
    
    # przygotowanie danych dla box-plotu
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # lista etykiet dla klas
    formatted_labels = [CLASS_NAMES[lbl] for lbl in labels]

    # odwrócenie kolejności wyświetlania dla zachowania spójności
    explicit_order = [CLASS_NAMES[i] for i in range(9)]
    
    # rysowanie boxplotu
    sns.boxplot(x=mean_brightness, y=formatted_labels, order=explicit_order, palette="Spectral")
    
    plt.title("Analiza statystyczna jasności próbek w podziale na klasy tkankowe", fontsize=14, pad=15)
    plt.xlabel("Średnia jasność pikseli (0 - Czarny, 255 - Biały)", fontsize=12)
    plt.ylabel("Klasa tkankowa", fontsize=12)
    
    output_path = os.path.join(save_dir, "eda_brightness_boxplot.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[ZAPISANO] Wykres pudełkowy jasności: {output_path}")

def plot_pca_separation(images, labels, save_dir):
    # spłaszczenie obrazów i redukcja ich wymiarowości za pomocą PCA, sprawdzenie stopnia liniowej separowalności 9 klas
    print("Generowanie analizy redukcji wymiarowości PCA...")
    
    # losowa próbka 3000 próbek dla czytelności
    np.random.seed(42)
    sample_indices = np.random.choice(len(images), min(3000, len(images)), replace=False)
    sampled_images = images[sample_indices]
    sampled_labels = labels[sample_indices]
    
    # spłaszczenie obrazów
    num_samples = len(sampled_images)
    flattened_images = sampled_images.reshape(num_samples, -1)
    
    # PCA do 2 głównych składowych
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(flattened_images)
    
    # rysowanie wykresu
    plt.figure(figsize=(12, 8))
    sns.set_style("white")
    
    # paleta 9 kolorów jak w box-plocie
    custom_spectral_colors = sns.color_palette("Spectral", n_colors=9)
    
    for cls in range(9):
        cls_mask = (sampled_labels == cls)
        plt.scatter(
            pca_results[cls_mask, 0], 
            pca_results[cls_mask, 1], 
            label=CLASS_NAMES[cls],
            alpha=0.6,
            edgecolors='none',
            s=25,
            color=custom_spectral_colors[cls]
        )
        
    plt.title("Wizualizacja przestrzeni cech PathMNIST za pomocą PCA (2D)", fontsize=14, pad=15)
    plt.xlabel(f"Główna składowa 1 ({pca.explained_variance_ratio_[0]*100:.2f}% wariancji)", fontsize=12)
    plt.ylabel(f"Główna składowa 2 ({pca.explained_variance_ratio_[1]*100:.2f}% wariancji)", fontsize=12)
    
    # legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = os.path.join(save_dir, "eda_pca_separation.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[ZAPISANO] Wykres rzutowania PCA: {output_path}")

def run_eda():
    # ścieżki wyjściowe
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "pathmnist_64.npz")
    results_dir = os.path.join(base_dir, "results", "eda")
    os.makedirs(results_dir, exist_ok=True)
    
    # załadowanie surowych danych
    data = np.load(data_path)
    train_images = data['train_images']
    train_labels = data['train_labels'].squeeze()
    
    # analizy wizualne
    #plot_sample_gallery(train_images, train_labels, results_dir) # wywołanie próbek graficznych dla klas
    #plot_rgb_intensity_distribution(train_images, results_dir) # wywołanie dystrybucjki RGB
    plot_brightness_boxplot(train_images, train_labels, results_dir) # wywołanie boxplotu jasności
    plot_pca_separation(train_images, train_labels, results_dir) # seraracja PCA
    print("\n[SUKCES] Wszystkie wizualizacje EDA zostały wygenerowane i zapisane w folderze 'reports/eda/'.")

if __name__ == "__main__":
    run_eda()