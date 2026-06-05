import torch
import torch.nn as nn
import torchvision.models as models

class LogisticRegressionBaseline(nn.Module):
    # regresja logistyczna
    def __init__(self, input_dim=64*64*3, num_classes=9):
        super(LogisticRegressionBaseline, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Spłaszczenie obrazu (Batch_size, 3, 64, 64) -> (Batch_size, 12288)
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)


class SimpleCNN(nn.Module):
    # prosty CNN
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # blok splotowy 1: wejście 64x64x3 -> wyjście 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # blok splotowy 2: wejście 32x32x32 -> wyjście 16x16x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # blok splotowy 3: wejście 16x16x64 -> wyjście 8x8x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5), # ochrona przed overfittingiem
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet18Transfer(nn.Module):
    # transfer learning bazowy REsNet18
    def __init__(self, num_classes=9, pretrained=True):
        super(ResNet18Transfer, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # podmiana warstwy w pełni połączonej pod 9 klas PathMNIST
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50Transfer(nn.Module):
    # transfer learning głęboki ResNet50
    def __init__(self, num_classes=9, pretrained=True):
        super(ResNet50Transfer, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV3Transfer(nn.Module):
    # efekwyny transfer learning z rozdzielnymi splotami
    def __init__(self, num_classes=9, pretrained=True):
        super(MobileNetV3Transfer, self).__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_large(weights=weights)
        
        # podmiana końcowego klasyfikatora
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class DenseNetTransfer(nn.Module):
    # metoda Advanced Sota
    def __init__(self, num_classes=9, pretrained=True):
        super(DenseNetTransfer, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)
        
        # podmiara klasyfikatora końcowego pod 9 klas
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def model_factory(model_name, num_classes=9, pretrained=True):
    # funkcja fabryki do pobierania modeli
    models_map = {
        'LogisticRegression': lambda: LogisticRegressionBaseline(num_classes=num_classes),
        'SimpleCNN':          lambda: SimpleCNN(num_classes=num_classes),
        'ResNet18':           lambda: ResNet18Transfer(num_classes=num_classes, pretrained=pretrained),
        'ResNet50':           lambda: ResNet50Transfer(num_classes=num_classes, pretrained=pretrained),
        'MobileNetV3':        lambda: MobileNetV3Transfer(num_classes=num_classes, pretrained=pretrained),
        'DenseNet121':        lambda: DenseNetTransfer(num_classes=num_classes, pretrained=pretrained)
    }
    
    if model_name in models_map:
        return models_map[model_name]()
    else:
        raise ValueError(f"Nieprawidłowa nazwa modelu: {model_name}. Wybierz z: {list(models_map.keys())}")


if __name__ == "__main__":
    print("=== Test poprawności alokacji wymiarów modeli ===")
    mock_input = torch.randn(2, 3, 64, 64) # 2 testowe obrazów 64x64
    names = ['LogisticRegression', 'SimpleCNN', 'ResNet18', 'ResNet50', 'MobileNetV3', 'DenseNet121']
    
    for name in names:
        try:
            model = model_factory(name, pretrained=False)
            output = model(mock_input)
            print(f"[OK] Model: {name:20s} | Shape wyjściowy: {list(output.shape)}")
        except Exception as e:
            print(f"[BŁĄD] Model: {name:20s} | Szczegóły błędu: {e}")