import socket
import tkinter
import cv2
from PIL import Image, ImageTk
from camera_manager import CameraManager
from concurrent.futures import ThreadPoolExecutor

# 表情モデルと分類ユーティリティ
from fer_model import fer_model, classify_pil

# YOLO はローカルウェブカメラ表示用に必要ならロード
# from ultralytics import YOLO
# try:
#   yolo_local = YOLO("yolov8n-face.pt")
# except Exception as e:
#   print(f"YOLO model load error (local): {e}")
#   yolo_local = None
from detector import detect_faces, yolo_available  # 追加

def on_key_press(event, current_camera_var, canvas_list, window):
  """キー入力を処理し、カメラの表示切り替えやズームを行う"""
  if event.keysym == 'q':
    window.quit()

# app.py の FaceGUI をローカル用に統合（名前を LocalWebcam に変更）
class LocalWebcam:
  def __init__(self, parent, width=320, height=240):
    self.parent = parent
    self.label = tkinter.Label(parent)
    self.label.pack(side="right")
    self.cap = cv2.VideoCapture(0)
    self.width = width
    self.height = height
    if self.cap.isOpened():
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    self.update_frame()

  def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
      img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # 顔検出 + 表情分類（モデルがあれば）
      try:
        if yolo_available and fer_model is not None:
          boxes = detect_faces(img_rgb)
          for (x1, y1, x2, y2) in boxes:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_rgb.shape[1], x2); y2 = min(img_rgb.shape[0], y2)
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
              continue
            face_pil = Image.fromarray(face)
            label = classify_pil(face_pil)
            if label is None:
              continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      except Exception:
        pass

      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = Image.fromarray(img)
      imgtk = ImageTk.PhotoImage(image=img)
      self.label.imgtk = imgtk
      self.label.configure(image=imgtk)
    self.parent.after(100, self.update_frame)

  def release(self):
    if self.cap.isOpened():
      self.cap.release()

def main():
  # メインウィンドウの作成
  window = tkinter.Tk()
  window.title("カメラ映像表示 (Raspi + Local)")

  # ラズパイ映像用キャンバス（左）
  canvas1 = tkinter.Canvas(window, width=640, height=480)
  canvas1.pack(side="left")

  # 画像参照を保持する変数（ラズパイ用）
  photo_var1 = [None]

  # キャンバスリスト（将来拡張用）
  canvas_list = [canvas1]

  # 現在表示しているカメラを保持する変数
  current_camera_var = [0]

  SERVER_IP = "10.133.4.202"  # ラズパイのIPアドレスを指定してください
  SERVER_PORT = 36131        # ラズパイのサーバーポートを指定してください

  # ソケット接続の確立
  client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client1.connect((SERVER_IP, SERVER_PORT))

  # qキーでプログラムを終了するためのイベントバインド
  window.bind('<KeyPress>', lambda event: on_key_press(
      event, current_camera_var, canvas_list, window))

  # CameraManager（ラズパイ受信）を作成
  camera1 = CameraManager(SERVER_IP, SERVER_PORT, canvas1, window)

  # ローカルウェブカメラ表示を作成（右）
  local_cam = LocalWebcam(window)

  # スレッドプールでラズパイ受信ループを実行
  with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(camera1.update_loop, client1, canvas1, photo_var1)
    try:
      window.mainloop()
    finally:
      # 終了時クリーンアップ
      local_cam.release()
      try:
        client1.close()
      except:
        pass

if __name__ == "__main__":
  main()
