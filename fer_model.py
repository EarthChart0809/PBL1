import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
  """PIL Image -> ラベル (str)"""
  if fer_model is None:
    return None
  x = preprocess(face_pil).unsqueeze(0).to(device)
  with torch.no_grad():
    out = fer_model(x)
    pred = torch.argmax(out, dim=1).item()
  return labels[pred]
