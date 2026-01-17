import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datetime import datetime
import torch.nn.functional as F

# --- FER3ResNet 定義 ---
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

# --- モデルロードと前処理 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def try_load_model(path="fer3class_resnet_ft_jaffe.pth"):
  try:
    model = FER3ResNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
  except Exception as e:
    print(f"FER model load error: {e}")
    return None

fer_model = try_load_model()

preprocess = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

labels = ["positive", "negative", "neutral"]

def classify_pil(face_pil):
  """PIL Image -> ラベル (str). ターミナルに結果を print する。"""
  if fer_model is None:
    return None
  try:
    x = preprocess(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
      out = fer_model(x)
      probs = F.softmax(out, dim=1).cpu().squeeze().numpy()
      pred_idx = int(probs.argmax())
      label = labels[pred_idx]
      score = float(probs[pred_idx])
    # ターミナル出力（タイムスタンプ付き）
    print(f"[{datetime.now().isoformat(timespec='seconds')}] 表情: {label} (score={score:.3f})")
    return label
  except Exception as e:
    print(f"classify_pil error: {e}")
    return None
