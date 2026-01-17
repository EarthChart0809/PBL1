# ==========================================
# YOLOv8-face + FER3ResNet 表情分類 GUI版
# ==========================================
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image, ImageTk
import cv2
import tkinter as tk

# ==========================================
# 1️⃣ FER3ResNet 定義とモデルロード
# ==========================================
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels)
    )
    self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.conv(x) + self.shortcut(x))

class FER3ResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
        ResidualBlock(1, 32),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        ResidualBlock(32, 64),
        nn.MaxPool2d(2),
        nn.Dropout(0.3),
        ResidualBlock(64, 128),
        nn.MaxPool2d(2),
        nn.Dropout(0.4),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3)
    )

  def forward(self, x):
    return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fer_model = FER3ResNet().to(device)
fer_model.load_state_dict(torch.load(
    "fer3class_resnet_ft_jaffe.pth", map_location=device))
fer_model.eval()

labels = ["positive", "negative", "neutral"]

# ==========================================
# 2️⃣ YOLOv8-face モデルロード
# ==========================================
yolo_model = YOLO("yolov8n-face.pt")  # 学習済みモデル

preprocess = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# 3️⃣ GUIセットアップ
# ==========================================
class FaceGUI:
  def __init__(self, window):
    self.window = window
    self.window.title("顔検出 + 表情分類")
    self.window.protocol('WM_DELETE_WINDOW', self.on_close)

    # Canvasにカメラ映像を描画
    self.label = tk.Label(window)
    self.label.pack()

    # Webカメラ
    self.cap = cv2.VideoCapture(0)

    self.update_frame()

  def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
      img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = yolo_model(img_rgb)[0]

      for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        face = img_rgb[y1:y2, x1:x2]

        if face.size == 0:
          continue

        face_pil = Image.fromarray(face)
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
          output = fer_model(face_tensor)
          pred = torch.argmax(output, dim=1).item()
          label = labels[pred]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img)
      imgtk = ImageTk.PhotoImage(image=img)
      self.label.imgtk = imgtk
      self.label.configure(image=imgtk)

    self.window.after(100, self.update_frame)

  def on_close(self):
    self.cap.release()
    self.window.destroy()

# ==========================================
# 4️⃣ 実行
# ==========================================
root = tk.Tk()
app = FaceGUI(root)
root.mainloop()
