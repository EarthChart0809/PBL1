import queue
import socket
import cv2
import struct
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future

SERVER_IP = '192.168.3.9'
SERVER_PORT = 36131
SERVER_PORT_CONTROLLER = 36132
SERVER_PORT_SERIAL = 36133
BUFSIZE = 4096


# キューを作成（エンコード待ちの画像フレームを格納）
frame_queue = queue.Queue(maxsize=5)  # キューのサイズを適切に設定

def encode_and_send(client_socket, frame_queue):
  """画像をエンコードして送信（別スレッドで処理）"""
  while True:
    try:
      frame = frame_queue.get()  # キューからフレームを取得
      if frame is None:
        break  # Noneを受け取ったらスレッド終了

      # **JPEG にエンコード（圧縮率を調整）**
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 画質 50 に設定（40 から微調整）
      _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
      data = img_encoded.tobytes()

      # **フレームサイズを送信**
      data_size = struct.pack(">L", len(data))
      client_socket.sendall(data_size)
      client_socket.sendall(data)

    except Exception as e:
      print(f"Error in encode_and_send: {e}")
      break

def capture_camera(camera_index, frame_queue, client_socket):
  """カメラキャプチャ（メインスレッドで処理）"""
  cap = cv2.VideoCapture(camera_index)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 解像度を下げて軽量化
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファを小さくして遅延を減らす
  cap.set(cv2.CAP_PROP_FPS, 15)  # FPSを下げてCPU負荷を軽減

  # # **TCP通信の最適化**
  # client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 即時送信
  # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)  # 送信バッファサイズを小さく
  # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # 受信バッファサイズを小さく
  # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # KeepAliveを有効化

  if not cap.isOpened():
    print(f"Error: Could not open camera {camera_index}.")
    return

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        print(
            f"Error: Failed to capture image from camera {camera_index}.")
        time.sleep(0.05)
        continue

      # **フレームをキューに追加（エンコードスレッドが処理）**
      if not frame_queue.full():  # キューが満杯ならスキップして最新フレームを優先
        frame_queue.put(frame)

  except Exception as e:
    print(f"Error in capture_camera {camera_index}: {e}")
  finally:
    cap.release()
    frame_queue.put(None)  # エンコードスレッドを終了させる


def main():

  # メイン処理
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.bind((SERVER_IP, SERVER_PORT))
  server.listen(2)  # 2つのクライアントを待機

  print("Waiting for connection...")

  sv = socket.socket(socket.AF_INET)
  sv.bind((SERVER_IP, SERVER_PORT_SERIAL))
  sv.listen()
  sv.settimeout(1.0)  # **タイムアウトを 1 秒に**

  client_socket1, client_address1 = server.accept()
  print(f"Connection from: {client_address1} for Camera 1")

  # **カメラごとのフレームキューを作成**
  frame_queue1 = queue.Queue(maxsize=5)

  # **エンコード専用スレッドを開始**
  encode_thread1 = threading.Thread(target=encode_and_send, args=(client_socket1, frame_queue1), daemon=True)

  encode_thread1.start()

  # カメラ処理を別スレッドで実行
  thread1 = threading.Thread(target=capture_camera, args=(0, frame_queue1, client_socket1))
  thread1.start()


if __name__ == '__main__':
  main()
