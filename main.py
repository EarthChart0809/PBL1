import socket
import tkinter
from camera_manager import CameraManager
from concurrent.futures import ThreadPoolExecutor

def on_key_press(event, current_camera_var, canvas_list, window):
  """キー入力を処理し、カメラの表示切り替えやズームを行う"""
  if event.keysym == 'q':
    window.quit()

def main():
  # メインウィンドウの作成
  window = tkinter.Tk()
  window.title("カメラ映像表示 (Raspi)")

  # ラズパイ映像用キャンバス（左）
  canvas1 = tkinter.Canvas(window, width=640, height=480)
  canvas1.pack(side="left")

  # 画像参照を保持する変数（ラズパイ用）
  photo_var1 = [None]

  # キャンバスリスト（将来拡張用）
  canvas_list = [canvas1]

  # 現在表示しているカメラを保持する変数
  current_camera_var = [0]

  SERVER_IP = "172.20.10.2"  # ラズパイのIPアドレスを指定してください
  SERVER_PORT = 36131        # ラズパイのサーバーポートを指定してください

  # ソケット接続の確立
  client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client1.connect((SERVER_IP, SERVER_PORT))

  # qキーでプログラムを終了するためのイベントバインド
  window.bind('<KeyPress>', lambda event: on_key_press(
      event, current_camera_var, canvas_list, window))

  # CameraManager（ラズパイ受信）を作成
  camera1 = CameraManager(SERVER_IP, SERVER_PORT, canvas1, window)

  # スレッドプールでラズパイ受信ループを実行
  with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(camera1.update_loop, client1, canvas1, photo_var1)
    try:
      window.mainloop()
    finally:
      try:
        client1.close()
      except:
        pass

if __name__ == "__main__":
  main()
