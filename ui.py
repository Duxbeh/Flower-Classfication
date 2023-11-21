import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle('Flower Classifier')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.btn_open = QPushButton('Open Image', self)
        self.btn_open.clicked.connect(self.open_image)
        self.layout.addWidget(self.btn_open)

        self.prediction_label = QLabel('', self)
        self.layout.addWidget(self.prediction_label)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def load_model(self):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5)  # Assuming 5 classes
        self.model.load_state_dict(torch.load('best_model.pth', map_location=torch.device(device)))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def open_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            if filenames:
                image_path = filenames[0]
                self.predict_flower(image_path)

    def predict_flower(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        prediction = class_names[predicted.item()]

        self.display_image(image_path)
        self.prediction_label.setText(f'Predicted flower: {prediction}')

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(300, 200)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FlowerClassifierApp()
    window.show()
    sys.exit(app.exec_())
