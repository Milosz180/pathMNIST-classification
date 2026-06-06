# PathMNIST CADx — Computer-Aided Pathomorphological Diagnosis System

A desktop CADx (Computer-Aided Diagnosis) application designed for the automated classification of colorectal cancer histopathological tissue slides based on the **PathMNIST** medical dataset. The core of the system is an optimized **MobileNetV3** convolutional neural network developed in **PyTorch**, enhanced with asymmetric decision threshold calibration and an advanced Explainable AI (**XAI**) interpretability layer.

---

## 📌 Project Overview & Pipeline Architecture

The project implements a complete data engineering and deep learning pipeline, divided into logical, sequential phases:

1. **Exploratory Data Analysis (EDA):** Verifying data integrity and performing PCA dimensionality reduction to analyze tissue class separability.
2. **Architecture Screening (Benchmark):** Comparative testing of 6 models (from Logistic Regression to DenseNet121). **MobileNetV3** was selected as the optimal performance-accuracy compromise.
3. **Bayesian Optimization (Optuna):** Automated hyperparameter tuning (Learning Rate, batch size, optimizer) utilizing a *MedianPruner* for early stopping of underperforming trials.
4. **Production Training:** Full-scale training integrated with an *Early Stopping* mechanism to prevent overfitting.
5. **Medical Calibration (Youden Rescue):** Rejecting the standard `argmax` decision function in favor of optimal cutoff thresholds derived from **Youden's $J$ statistic** (ROC curves), successfully minimizing critical *False Negative* errors.
6. **Clinical Explainability (XAI):** A hybrid interpretability pipeline combining the visual **Grad-CAM** algorithm (mapping cellular structure activations) with the game-theoretic **SHAP** method.

---

## 📦 Required Binary Assets (Pre-trained Models)

To run the CADx graphical user interface immediately without going through the time-consuming model training process, download the following two files from the *Releases* section and place them inside the `models/final_model/` directory:

* **Network Weights File** **title:** `mobilenetv3_final_optimized.pth`  
  **description:** [GitHub Releases / Model pth](https://github.com/milosz-gronowski/pathMNIST/releases/download/v1.0.0/mobilenetv3_final_optimized.pth)
* **Medical Thresholds File** **title:** `medical_thresholds.json`  
  **description:** [GitHub Releases / Youden Thresholds](https://github.com/milosz-gronowski/pathMNIST/releases/download/v1.0.0/medical_thresholds.json)

---

## 🛠️ Environment Prerequisites and Installation

Before launching the interface, ensure you have **Python >= 3.12** installed on your system.

1. Clone the project repository:
   ```bash
   git clone [https://github.com/milosz-gronowski/pathMNIST.git](https://github.com/milosz-gronowski/pathMNIST.git)
   cd pathMNIST

2. Install the required computational and graphical packages using the pip package manager:
   pip install -r requirements.txt

## 🚀 Application Launch Instructions

1. Method 1 - The simplest method, requiring no command-line interaction. Navigate to the main project directory and double-click the following file:
   run_app.bat

2. Method 2 - Designed for maintenance, debugging, and monitoring network logs in real-time. Open your terminal (CMD/PowerShell) in the root directory of the project and execute:
   python -m src.app




# PathMNIST CADx — System Wspomagania Diagnostyki Patomorfologicznej

Aplikacja desktopowa typu CADx (Computer-Aided Diagnosis) służąca do zautomatyzowanej klasyfikacji wycinków histopatologicznych raka jelita grubego na podstawie medycznego zbioru danych **PathMNIST**. Sercem systemu jest zoptymalizowana sieć splotowa **MobileNetV3**, zaprogramowana w środowisku **PyTorch**, wzbogacona o asymetryczną kalibrację progów decyzyjnych oraz zaawansowaną warstwę interpretowalności sztucznej inteligencji (**XAI**).

---

## 📌 Podsumowanie projektu & Architektura potoku

Projekt realizuje pełen potok inżynierii danych oraz uczenia głębokiego (Deep Learning), podzielony na logiczne, sekwencyjne fazy:

1. **Analiza eksploracyjna (EDA):** Weryfikacja integralności danych i redukcja wymiarowości metodą PCA w celu analizy separowalności klas tkankowych.
2. **Benchmark architektur (Screening):** Testy porównawcze 6 modeli (od regresji logistycznej po DenseNet121). Jako optymalny kompromis wydajnościowy wyłoniono sieć **MobileNetV3**.
3. **Optymalizacja bajesowska (Optuna):** Automatyczny tuning hiperparametrów (tempo uczenia, rozmiar paczki, optymalizator) z wykorzystaniem mechanizmu *MedianPruner* do wczesnego odrzucania słabych prób.
4. **Trening produkcyjny:** Pełnowymiarowe uczenie zintegrowane z mechanizmem *Early Stopping* chroniącym przed przeuczeniem (overfittingiem).
5. **Kalibracja medyczna (Youden Rescue):** Odrzucenie klasycznej funkcji `argmax` w warstwie decyzyjnej na rzecz optymalnych progów odcięcia wyznaczonych ze statystyki **$J$ Youdena** (krzywe ROC), co pozwoliło skutecznie zminimalizować krytyczne błędy typu *False Negative*.
6. **Wyjaśnialność kliniczna (XAI):** Hybrydowy potok interpretowalności łączący wizualny algorytm **Grad-CAM** (mapowanie aktywacji struktur komórkowych) z teoriogrową metodą **SHAP**.

---

## 📦 Wymagane artefakty binarne (Pre-trained Models)

Aby uruchomić aplikację graficzną CADx natychmiast, z pominięciem czasochłonnego procesu trenowania modeli, należy pobrać z sekcji *Releases* i umieścić w katalogu `models/final_model/` dwa kluczowe pliki:

* **Plik wag sieci** **title:** `mobilenetv3_final_optimized.pth`  
  **description:** [GitHub Releases / Model pth](https://github.com/milosz-gronowski/pathMNIST/releases/download/v1.0.0/mobilenetv3_final_optimized.pth)
* **Plik progów medycznych** **title:** `medical_thresholds.json`  
  **description:** [GitHub Releases / Progi Youdena](https://github.com/milosz-gronowski/pathMNIST/releases/download/v1.0.0/medical_thresholds.json)

---

## 🛠️ Wymagania środowiskowe i instalacja

Przed uruchomieniem interfejsu należy upewnić się, że w systemie zainstalowany jest interpretator **Python w wersji >= 3.12**.

1. Sklonuj repozytorium projektu:
```bash
   git clone [https://github.com/milosz-gronowski/pathMNIST.git](https://github.com/milosz-gronowski/pathMNIST.git)
   cd pathMNIST
   
2. Zainstaluj wymagane pakiety obliczeniowe i graficzne za pomocą menedżera pip:
   pip install -r requirements.txt

## 🚀 Instrukcja uruchomienia aplikacji

1. Metoda 1 - Najprostsza metoda, niewymagająca interakcji z wierszem poleceń. Przejdź do głównego katalogu projektu i kliknij dwukrotnie lewym przyciskiem myszy na plik:
   run_app.bat

2. Metoda 2 - Przeznaczona do prac konserwacyjnych, debugowania oraz monitorowania logów sieci w czasie rzeczywistym. Otwórz terminal (CMD/PowerShell) w głównym katalogu projektu i wykonaj:
   python -m src.app