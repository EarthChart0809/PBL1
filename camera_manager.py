import tkinter
from tkinter import Button
import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import threading
import struct
import time

# YOLO と 表情分類モジュールをインポート
# from ultralytics import YOLO
# from PIL import Image
from PIL import Image
from fer_model import fer_model, classify_pil
from detector import detect_faces, yolo_available

# # YOLO モデル読み込み（存在しなければ None）
# try:
#   yolo_model = YOLO("yolov8n-face.pt")
# except Exception as e:
#   print(f"YOLO model load error: {e}")
#   yolo_model = None

class CameraManager:
  def __init__(self, server_ip, server_port, canvas, window):
    self.server_ip = server_ip
    self.server_port = server_port
    self.canvas = canvas
    self.window = window  # **ここで Tkinter のウィンドウを管理**
    self.photo_var = [None]
    self.client = None
    self.last_draw_time = 0  # 描画のためのタイムスタンプ

    # 検出/分類のスロットリング設定
    self.last_detection_time = 0.0
    self.detection_interval = 5  # 秒単位（必要に応じて変更）
    self.cached_boxes = []
    self.cached_labels = []

  # 各カメラのフレームを表示する関数
  def update_image(self, data, canvas, photo_var):
    if not canvas.winfo_ismapped():
      return

    img = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    if img is None:
      print("Failed to decode image data.")
      return

    # 顔検出と表情分類を間引き実行（重い処理を dtection_interval 秒ごとに実行）
    try:
      now = time.time()
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if (yolo_available and fer_model is not None and
              (now - self.last_detection_time) >= self.detection_interval):
        boxes = detect_faces(img_rgb)
        labels = []
        for (x1, y1, x2, y2) in boxes:
          x1 = max(0, x1); y1 = max(0, y1)
          x2 = min(img_rgb.shape[1], x2); y2 = min(img_rgb.shape[0], y2)
          face = img_rgb[y1:y2, x1:x2]
          if face.size == 0:
            labels.append(None)
            continue
          face_pil = Image.fromarray(face)
          label = classify_pil(face_pil)
          labels.append(label)
        # キャッシュ更新
        self.cached_boxes = boxes
        self.cached_labels = labels
        self.last_detection_time = now
      # 描画はキャッシュを使う（直近の検出結果をオーバーレイ）
      if self.cached_boxes:
        for idx, (x1, y1, x2, y2) in enumerate(self.cached_boxes):
          lbl = self.cached_labels[idx] if idx < len(
              self.cached_labels) else None
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
          if lbl:
            cv2.putText(img, lbl, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      # 表示用にRGBへ変換（PhotoImageのため）
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
      print(f"Error in detection/classify: {e}")
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))

    def update_canvas():
      canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
      photo_var[0] = photo

    self.window.after(0, update_canvas)

  # 各カメラの更新ループを実行する関数
  def update_loop(self, client, canvas, photo_var):
    data = b""
    print("カメラの受信ループ開始")
    while True:
      try:
        while len(data) < 4:
          packet = client.recv(4096)
          if not packet:
            return
          data += packet
        data_size = struct.unpack(">L", data[:4])[0]
        data = data[4:]

        # print(f"受信データサイズ: {data_size} バイト")

        while len(data) < data_size:
          packet = client.recv(4096)
          if not packet:
            return
          data += packet

        img_data = data[:data_size]
        data = data[data_size:]

        # **メインスレッドで画像を更新**
        self.window.after(0, self.update_image, img_data, self.canvas,
                          self.photo_var)

      except Exception as e:
        print(f"Error in update_loop: {e}")
        break
