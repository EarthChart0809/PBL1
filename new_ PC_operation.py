import socket
import tkinter
from tkinter import Button
import threading
import copy
from camera_manager import CameraManager
from concurrent.futures import ThreadPoolExecutor
  

def on_key_press(event, current_camera_var, canvas_list, window):
  """キー入力を処理し、カメラの表示切り替えやズームを行う"""
  if event.keysym == 'q':
    window.quit()


def main():
  # メインウィンドウの作成
  window = tkinter.Tk()
  window.title("カメラ映像表示")

  # 2つのキャンバスを作成（それぞれのカメラ用）
  canvas1 = tkinter.Canvas(window, width=640, height=480)
  canvas1.pack(side="left")

  # 画像参照を保持する変数を作成（それぞれのカメラ用）
  photo_var1 = [None]
  photo_var2 = [None]

  # キャンバスリストを作成
  canvas_list = [canvas1]

  # 現在表示しているカメラを保持する変数
  current_camera_var = [0]  # 初期状態ではカメラ1

  SERVER_IP = "10.133.4.202"  # ★ラズパイのIPアドレスを指定してください
  SERVER_PORT = 36131        # ★ラズパイのサーバーポートを指定してください

  # ソケット接続の確立
  client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client1.connect((SERVER_IP, SERVER_PORT))

  # qキーでプログラムを終了するためのイベントバインド
  window.bind('<KeyPress>', lambda event: on_key_press(
      event,current_camera_var, canvas_list, window))

  camera1 = CameraManager(SERVER_IP, SERVER_PORT, canvas1, window)

  # **スレッドプールの作成**
  with ThreadPoolExecutor(max_workers=7) as executor:
      # カメラデータ受信スレッド
    executor.submit(camera1.update_loop, client1, canvas1,photo_var1)

    # **Tkinterのメインループを実行**
    window.mainloop()

# スクリプトとして実行された場合に main() を呼び出す
if __name__ == "__main__":
  main()
