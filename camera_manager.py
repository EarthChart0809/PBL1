import tkinter
from tkinter import Button
import numpy as np
import PIL.Image
import PIL.ImageTk
import cv2
import threading
import struct

class CameraManager:
  def __init__(self, server_ip, server_port, canvas,window):
        self.server_ip = server_ip
        self.server_port = server_port
        self.canvas = canvas
        self.window = window  # **ここで Tkinter のウィンドウを管理**
        self.photo_var = [None]
        self.zoom_factor = [1]  # ズーム倍率をリストで管理
        self.zoom_lock = threading.Lock()
        self.client = None
        self.last_qr_time = 0 # QRコード取得のためのタイムスタンプ
        self.last_draw_time = 0  # 描画のためのタイムスタンプ


  # 各カメラのフレームを表示する関数
  def update_image(self,data, canvas, photo_var):
    # now = time.time()
    # if now - self.last_draw_time < 0.033:  # 約30fps
    #     return
    # self.last_draw_time = now
    
    if not canvas.winfo_ismapped():  # キャンバスが表示されていない場合は処理しない
        return
    
    img = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    if img is None:
        print("Failed to decode image data.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))

    def update_canvas():
        canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
        photo_var[0] = photo  # メモリ上で画像を保持するために参照を保存

    self.window.after(0, update_canvas)  # **`window` ではなく `canvas` に `after` を適用**

  # 各カメラの更新ループを実行する関数
  def update_loop(self,client, canvas, photo_var):
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
                              self.photo_var, self.zoom_factor, self.zoom_lock)

        except Exception as e:
            print(f"Error in update_loop: {e}")
            break


  