# Camera & FER (顔表情認識) Project

## 概要
Raspberry Pi で JPEG 圧縮して送信したカメラ映像を PC 側で受信・表示し、YOLOv8-face による顔検出と FER3ResNet による表情分類を行うサンプルプロジェクト。GUI は Tkinter、顔検出モデルは detector.py で一度だけロードして共有します。

## カメラ映像取得機能概要
Raspberry Pi から JPEG 圧縮したカメラ映像を TCP で送信し、PC 側で受信・デコードして Tkinter の Canvas に描画するサンプルプロジェクト。表情認識（FER）、YOLO 関連の機能はこの README から除外しています。

## 仕組み（簡潔）
- 送信（Raspberry Pi 側）
  - カメラを OpenCV でキャプチャ。
  - JPEG にエンコード（例: cv2.imencode('.jpg', frame, [IMWRITE_JPEG_QUALITY, 50])）。
  - 送信前に先頭 4 バイトでデータ長を big-endian unsigned long（">L"）で付加。
  - TCP ソケットでサーバ（PC）へ送信。
  - スレッド／キュー構成：キャプチャスレッドはフレームをキューに入れ、別スレッドでエンコード＆送信して負荷を分離。

- 受信（PC 側）
  - TCP クライアントで接続後、受信ループで先頭 4 バイトを読みデータ長を取得。
  - 指定バイト数を完全受信してからバッファを切り出し、cv2.imdecode でデコード。
  - Tkinter メインスレッドで Canvas に描画（別スレッドで受信し、window.after(0, ...) でメインスレッドに描画委譲）。
  - ローカルウェブカメラは別ラベル（右側）で同時表示可能。

## 主要ファイル（表情認識除外）
- new_sent_Raspi.py — ラズパイ側のキャプチャ／エンコード／送信サンプル（キューとスレッドで実装）
- main.py — PC 側メイン（Tkinter GUI、受信ループ、ローカルウェブカメラ表示）
- camera_manager.py — 受信データのデコードと Canvas 更新、受信ループ実装

## 依存ライブラリ（最低限）
python -m pip install opencv-python pillow numpy

※ 上の他にネットワーク周りは標準ライブラリ（socket, struct, threading, concurrent.futures）を使用。

## 実行手順（簡潔）
1. Raspberry Pi 側で IP/PORT を設定して送信を開始:
   python new_sent_Raspi.py
2. PC 側で main.py の SERVER_IP / SERVER_PORT を送信側に合わせて実行:
   python main.py

## トラブルシューティング
- 受信が途中で切れる・フレームが壊れる場合：受信ループで完全受信（先頭4バイト→サイズ→サイズ分のデータ）を確保しているか確認。
- 接続エラー：SERVER_IP / SERVER_PORT を両側で一致させ、ファイアウォールを確認。
- パフォーマンス改善：送信側の解像度（例 320x240）、FPS、JPEG 品質を下げる。送受信バッファサイズや TCP_NODELAY を調整。
- キューが満杯なら最新フレーム優先の設計になっています（古いフレームを破棄）。

## 注意
本プロジェクトは学習・実験目的のサンプルです。公開・配布時は利用ライブラリのライセンスに注意してください。

## 必要ファイル（プロジェクト直下に配置）
- yolov8n-face.pt
- fer3class_resnet_ft_jaffe.pth

## 主要ファイル
- `main.py` — PC 側メイン（Tkinter GUI、Raspi 受信、ローカル webcam 表示）
- `camera_manager.py` — ラズパイから受信した画像のデコード・描画（Canvas 更新）
- `detector.py` — YOLO モデルを一度だけ読み込み、detect_faces を提供
- `fer_model.py` — FER3ResNet 定義、モデルロード、`classify_pil()` を提供（出力は "positive"/"negative"/"neutral" の文字列）
- `new_sent_Raspi.py` — ラズパイ側送信サンプル（JPEG エンコードして TCP 送信）
- （補助）その他ソース・ドキュメント

## 出力形式
- 表情分類: `classify_pil()` は文字列ラベルを返します — `"positive"`, `"negative"`, `"neutral"`
- 顔検出: `detect_faces(img_rgb)` は (x1,y1,x2,y2) のタプルリストを返します

## 依存ライブラリ（例）
```bash
python -m pip install ultralytics torch torchvision pillow opencv-python numpy pyzbar
# PyTorch は環境に合わせて公式サイトの指示でインストールしてください（CUDA 有無でコマンドが変わります）
```

## 実行手順
1. ラズパイ（送信側）でモデル不要。IP/PORT を `new_sent_Raspi.py` の設定に合わせて実行:
   ```bash
   python new_sent_Raspi.py
   ```
2. PC 側でモデルファイルを配置し、IP/PORT を `main.py` に合わせて実行:
   ```bash
   python main.py
   ```

※ main.py 内の `SERVER_IP` / `SERVER_PORT` をラズパイ側と一致させてください。

## 注意点・トラブルシューティング
- YOLO は `detector.py` で一度だけロードしてメモリ／起動時間を節約しています。複数ファイルから利用する場合は detector を import してください。
- モデルファイルが無い、またはロード失敗時は該当機能が無効化されます（エラーメッセージを確認）。
- 受信データは先頭4バイトにデータ長（big-endian unsigned long）を付加して送っています。受信側の `update_loop` が正しく完全受信できているか確認してください。
- パフォーマンス調整: ラズパイ側の JPEG 品質やフレームレート（例: encode_quality / CAP_PROP_FPS）を下げると安定します。

## 開発メモ
- GUI（Tkinter）はメインスレッドで動作。受信は別スレッドで行い、`window.after(0, ...)` を使ってメインスレッドに描画を委譲しています。
- 表情分類は入力を PIL.Image（顔領域）に変換して `classify_pil()` を呼ぶ設計です。

## ライセンス
このプロジェクトは学習／実験用途を想定しています。外部公開の際は依存ライブラリのライセンスに注意してください。
