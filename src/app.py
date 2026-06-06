import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# import biblioteki PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QLabel, QFileDialog, QSlider, QVBoxLayout, 
                             QHBoxLayout, QFrame, QSplitter, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QCursor, QColor, QIcon
from PyQt6.QtCore import Qt
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# import fabryki modeli
from src.models import model_factory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# słownik mapowania klas - każda klasa posiada własny kolor
CLASS_MAPPING = {
    0: {"name": "Tkanka tłuszczowa (Adipose)", "short": "Tkanka tłuszczowa", "status": "Prawidłowa / Bez zmian", "color": "#27ae60", "is_patho": False}, # Żywa zieleń
    1: {"name": "Tło preparatu (Background)", "short": "Tło preparatu", "status": "Neutralny", "color": "#7f8c8d", "is_patho": False},             # Neutralny szary
    2: {"name": "Gęste podścielisko nowotworowe (Stroma)", "short": "Podścielisko nowotw.", "status": "PATOLOGIA / ALARM", "color": "#e67e22", "is_patho": True}, # Pomarańczowy
    3: {"name": "Nacieki limfocytarne (Lymphocytes)", "short": "Nacieki limfocyt.", "status": "PATOLOGIA / MONITORUJ", "color": "#9b59b6", "is_patho": True},   # Wyrazisty fiolet
    4: {"name": "Gruczoły jelitowe prawidłowe (Mucosa)", "short": "Gruczoły prawidłowe", "status": "Prawidłowa / Bez zmian", "color": "#1abc9c", "is_patho": False}, # Turkusowy
    5: {"name": "Gruczolak / Rak gruczołowy (Adenocarcinoma)", "short": "Rak gruczołowy", "status": "KRYTYCZNY / NOWOTWÓR", "color": "#c0392b", "is_patho": True}, # Intensywna czerwień
    6: {"name": "Tkanka limfoidalna (Lymphoid tissue)", "short": "Tkanka limfoidalna", "status": "Prawidłowa / Bez zmian", "color": "#006699", "is_patho": False}, # Głęboki błękit
    7: {"name": "Mięśniówka gładka (Smooth muscle)", "short": "Mięśniówka gładka", "status": "Prawidłowa / Bez zmian", "color": "#f1c40f", "is_patho": False},    # Jaskrawy żółty
    8: {"name": "Prawidłowa śluzówka jelita (Normal mucosa)", "short": "Śluzówka prawidłowa", "status": "Prawidłowa / Bez zmian", "color": "#d81b60", "is_patho": False} # Karminowy/Róż
}

# funckja do automatycznego obliczania odcieni tła i czcionki
def get_auto_colors(hex_color):
    color = QColor(hex_color)
    h, s, v, a = color.getHsv()
    bg_color = QColor.fromHsv(h, int(s * 0.12), int(255 * 0.97)) # pastel jasny
    text_color = QColor.fromHsv(h, int(s * 1.0), int(255 * 0.35)) # ciemny kontrast
    return bg_color.name(), text_color.name()

# klasa silnika Grad-CAM
class GradCAMEngine:
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

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = np.maximum(cam.cpu().numpy(), 0)
        
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, output

# główne okno aplikacji desktopowej
class MEdicalCADxApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasyfikacja PathMNIST") # nazwa okna
        self.resize(1250, 800)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "final_model", "mobilenetv3_final_optimized.pth")
        self.thresholds_path = os.path.join(base_dir, "models", "final_model", "medical_thresholds.json")
        self.logo_path = os.path.join(base_dir, "logo.png") # logo okna aplikacji

        # wymuszenie wyświetlania własnej ikony na pasku zadań
        import sys
        if sys.platform == "win32":
            import ctypes
            myappid = "milosz.pathmnist.cadx.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        # ikona dla głównego okna aplikacji
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))

        self.orig_cv_img = None
        self.heatmap_cv_img = None
        self.input_tensor = None
        
        self.load_medical_backend()
        self.init_ui()

    def load_medical_backend(self):
        with open(self.thresholds_path, "r") as f:
            self.thresholds = {int(k): v for k, v in json.load(f).items()}
            
        self.model = model_factory('MobileNetV3', num_classes=9, pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        target_layer = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)][-1]
        self.cam_engine = GradCAMEngine(self.model, target_layer)
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # panel lewy - przycisk, wczytywany obraz, nazwa pliku i suwak
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_load = QPushButton("Wczytaj obraz histopatologiczny")
        self.btn_load.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_load.setFixedHeight(45)
        self.btn_load.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) # kursor rączki
        self.btn_load.clicked.connect(self.open_file_dialog)
        left_layout.addWidget(self.btn_load)
        
        self.view_label = QLabel("Proszę wczytać zdjęcie tkanki bioptatu...")
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.view_label.setMinimumSize(450, 450)
        left_layout.addWidget(self.view_label)
        
        # nazwa wczytywanego pliku
        self.lbl_file_name = QLabel("")
        self.lbl_file_name.setFont(QFont("Arial", 10, QFont.Weight.Medium))
        self.lbl_file_name.setStyleSheet("color: #555555; margin-top: 2px; margin-bottom: 5px;")
        self.lbl_file_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_file_name)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Tkanka (0%)"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(0) 
        self.alpha_slider.setEnabled(False)
        self.alpha_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) # kursor rączki
        self.alpha_slider.valueChanged.connect(self.update_blend_view)
        slider_layout.addWidget(self.alpha_slider)
        slider_layout.addWidget(QLabel("Grad-CAM (100%)"))
        left_layout.addLayout(slider_layout)
        
        splitter.addWidget(left_panel)
        
        # panel prawy - wynik, diagnoza i wykres
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        header_layout = QHBoxLayout()
        self.result_frame = QFrame()
        self.result_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.result_frame.setStyleSheet("background-color: #f1f2f6; border: 1px solid #ced6e0; border-radius: 8px;")
        rf_layout = QVBoxLayout(self.result_frame)
        
        self.lbl_class_title = QLabel("DIAGNOZA: Oczekiwanie na badanie...")
        self.lbl_class_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_class_title.setLineWidth(0)
        self.lbl_class_title.setMidLineWidth(0)
        self.lbl_class_title.setStyleSheet("color: #2f3542; background: transparent; border: 0px solid transparent; border-image: none;")
        self.lbl_class_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rf_layout.addWidget(self.lbl_class_title)
        
        self.lbl_status = QLabel("Status kliniczny: -")
        self.lbl_status.setFont(QFont("Arial", 11, QFont.Weight.Medium))
        self.lbl_status.setLineWidth(0)
        self.lbl_status.setMidLineWidth(0)
        self.lbl_status.setStyleSheet("color: #57606f; background: transparent; border: 0px solid transparent; border-image: none;") 
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rf_layout.addWidget(self.lbl_status)
        
        self.lbl_class_title.setWordWrap(True)
        self.lbl_status.setWordWrap(True)

        header_layout.addWidget(self.result_frame, stretch=5)
        
        # przycisk INFO
        self.btn_info = QPushButton("i")
        self.btn_info.setFont(QFont("Georgia", 16, QFont.Weight.Bold))
        self.btn_info.setFixedSize(45, 60)
        self.btn_info.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_info.setStyleSheet("""
            QPushButton { background-color: #2980b9; color: white; border-radius: 8px; border: 1px solid #2471a3; }
            QPushButton:hover { background-color: #3498db; }
        """)
        self.btn_info.clicked.connect(self.show_info_modal)
        header_layout.addWidget(self.btn_info, stretch=0)
        
        right_layout.addLayout(header_layout)

        self.fig, self.ax = plt.subplots(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])

    def open_file_dialog(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        samples_dir = os.path.join(base_dir, "samples")
        
        # jeśli folder /samples nie istnieje, zaczynamy w głównym katalogu projektu
        if not os.path.exists(samples_dir):
            samples_dir = base_dir
            
        # otwarcie okna dialogowego w wyznaczonym folderze startowym
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Wybierz zdjęcie tkanki", 
            samples_dir,
            "Images (*.png *.jpg *.jpeg *.tif)"
        )
        
        if file_path:
            try:
                pure_name = os.path.basename(file_path)
                self.lbl_file_name.setText(f"Plik: {pure_name}")
                
                pil_raw = Image.open(file_path).convert('RGB')
                self.orig_cv_img = np.array(pil_raw)
                self.orig_cv_img = cv2.resize(self.orig_cv_img, (450, 450))
                
                self.input_tensor = self.transform(pil_raw).unsqueeze(0).to(DEVICE)
                self.run_cadx_diagnosis()
            except Exception as e:
                print(f"Błąd podczas ładowania pliku: {e}")

    def run_cadx_diagnosis(self):
        with torch.set_grad_enabled(True):
            cam_map, outputs = self.cam_engine.generate(self.input_tensor, target_class=0)
            probs = F.softmax(outputs, dim=1).squeeze(0).detach().cpu().numpy()
        
        scaled_probs = [probs[i] / self.thresholds[i] for i in range(9)]
        final_class = int(np.argmax(scaled_probs))
        
        with torch.set_grad_enabled(True):
            cam_map, _ = self.cam_engine.generate(self.input_tensor, target_class=final_class)
            
        heatmap_resized = cv2.resize(cam_map, (450, 450))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        self.heatmap_cv_img = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        self.alpha_slider.setEnabled(True)
        self.alpha_slider.setValue(0) 
        
        self.update_ui_cards(final_class)
        self.render_interactive_chart(probs, final_class)
        self.update_blend_view()

    def update_ui_cards(self, med_class):
        # aktualizacja tła
        class_info = CLASS_MAPPING[med_class]
        
        self.lbl_class_title.setText(f"DIAGNOZA: {class_info['name'].upper()} [KLASA {med_class}]")
        self.lbl_status.setText(f"Status kliniczny: {class_info['status']}")
        
        bg_hex, text_hex = get_auto_colors(class_info['color'])
        
        self.result_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_hex}; 
                border: 2px solid {class_info['color']}; 
                border-radius: 8px;
            }}
            QLabel {{
                border: none;
                background: transparent;
                border-image: none;
            }}
        """)
        
        self.lbl_class_title.setStyleSheet(f"color: {text_hex}; font-weight: bold; background: transparent; border: none;")
        self.lbl_status.setStyleSheet(f"color: {text_hex}; font-weight: bold; background: transparent; border: none;")

    def update_blend_view(self):
        if self.orig_cv_img is None or self.heatmap_cv_img is None:
            return
            
        alpha = self.alpha_slider.value() / 100.0
        beta = 1.0 - alpha
        
        blended = cv2.addWeighted(self.heatmap_cv_img, alpha, self.orig_cv_img, beta, 0)
        
        h, w, ch = blended.shape
        bytes_per_line = ch * w
        q_img = QImage(blended.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.view_label.setPixmap(QPixmap.fromImage(q_img))

    def render_interactive_chart(self, probabilities, selected_class):
        self.ax.clear()
        classes_labels = [f"[Klasa {i}] {CLASS_MAPPING[i]['short']}" for i in range(9)]
        percents = probabilities * 100
        
        # unikalne kolory słupków
        colors = [CLASS_MAPPING[i]['color'] for i in range(9)]
        
        # pełne nasycenie wygranej klasy
        bar_alphas = [1.0 if i == selected_class else 0.4 for i in range(9)]
        
        bars = self.ax.barh(classes_labels, percents, color=colors, edgecolor='black', height=0.6)        
        
        # zastosowanie przezroczystości dla wyróżnienia wygranej klasy
        for idx, bar in enumerate(bars):
            bar.set_alpha(bar_alphas[idx])
            
        self.ax.set_xlim(0, max(percents) + 15)
        self.ax.set_xlabel("Prawdopodobieństwo surowe (%)", fontweight='bold')
        self.ax.set_title("Rozkład prawdopodobieństw w sieci neuronowej", fontweight='bold', fontsize=11)
        self.ax.tick_params(axis='y', labelsize=8.5)
        self.ax.invert_yaxis()
        
        for bar in bars:
            width = bar.get_width()
            self.ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                         va='center', ha='left', fontsize=9, fontweight='bold')
            
        self.fig.tight_layout()
        self.canvas.draw()

    # funkcja modalnego INFO
    def show_info_modal(self):
        info_box = QMessageBox(self)
        info_box.setWindowTitle("Informacje o aplikacji")

        # ikona aplikacji dla modalnego okna
        if hasattr(self, 'logo_path') and os.path.exists(self.logo_path):
            info_box.setWindowIcon(QIcon(self.logo_path))
        
        html_content = """
        <h3>System Wspomagania Diagnostyki Patomorfologicznej</h3>
        <p><b>Autor aplikacji:</b> Miłosz Gronowski</p>
        <p>Aplikacja służy do klasyfikacji wycinków histopatologicznych raka jelita grubego na podstawie zbioru <b>PathMNIST</b> przy użyciu zoptymalizowanej sieci splotowej <b>MobileNetV3</b> z asymetryczną kalibracją progów decyzyjnych indeksu Youdena.</p>
        
        <hr>
        <h4><b>KRÓTKA INSTRUKCJA OBSŁUGI:</b></h4>
        <ol>
            <li><b>Wczytanie obrazu:</b> Kliknij przycisk <i>"Wczytaj obraz histopatologiczny"</i> i wskaż plik biopsyjny (np. z katalogu <code>/samples/</code>). Pod obrazem wyświetli się nazwa analizowanego pliku.</li>
            <li><b>Interpretacja diagnozy:</b> Górna karta natychmiast wyświetli ostateczny werdykt medyczny wraz z technicznym numerem klasy oraz statusem bezpieczeństwa.</li>
            <li><b>Czytanie wykresu:</b> Stały wykres po prawej stronie prezentuje surowe prawdopodobieństwa sieci dla wszystkich 9 klas. Słupek klasy zakwalifikowanej jako diagnoza główna jest podświetlony pełnym nasyceniem koloru, ułatwiając szybką ocenę różnicową.</li>
            <li><b>Suwak przezroczystości (Alpha Blending):</b> Przesuwaj suwak pod zdjęciem tkanki w prawo, aby płynnie nałożyć mapę aktywacji sieci <b>Grad-CAM</b>. Pozwala to dokładnie zweryfikować, jakie konkretne struktury komórkowe wywołały alarm modelu.</li>
        </ol>
        
        <hr>
        <h4><b>SŁOWNIK I LEGENDA KLAS HISTOPATOLOGICZNYCH:</b></h4>
        <table border="0" cellpadding="2">
        """
        # Dynamiczne generowanie legendy klas
        for k, v in CLASS_MAPPING.items():
            patho_tag = "<b style='color:#c0392b;'>[PATOLOGIA]</b>" if v['is_patho'] else "<b style='color:#27ae60;'>[NORMA]</b>"
            html_content += f"""
            <tr>
                <td><span style='color:{v['color']}; font-size:14px;'>■</span> <b>Klasa {k}:</b></td>
                <td>{v['name']}</td>
                <td>{patho_tag}</td>
            </tr>
            """
        html_content += """
        </table>
        """
        
        info_box.setText(html_content)
        info_box.setTextFormat(Qt.TextFormat.RichText)
        info_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        ok_button = info_box.button(QMessageBox.StandardButton.Ok)
        ok_button.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        ok_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        ok_button.setMinimumSize(90, 35)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 6px;
                border: 1px solid #2471a3;
                padding-left: 15px;
                padding-right: 15px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        info_box.setStyleSheet("""
            QMessageBox {
                dialogbuttonbox-buttons-have-icons: 0;
            }
            QDialogButtonBox {
                button-layout: 0; /* Wymuszenie symetrycznego układu środkowego */
                alignment: center;
            }
        """)
        info_box.exec()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MEdicalCADxApp()
    gui.show()
    
    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Zamykanie pętli aplikacji GUI.")