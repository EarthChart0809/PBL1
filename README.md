# Camera & Controller

簡潔な説明
- PC側で2台分のカメラ映像を受信・表示し、ジョイスティック（コントローラ）入力をラズパイ（送信側）に送るプロジェクト。

主要ファイル
- new_PC_operation.py — PC側メインアプリ（Tkinter GUI、カメラ受信、コントローラ送信）
- camera_manager.py — 受信画像のデコード／描画、QR検出、ズーム処理
- new_sent_Raspi.py / neo_sent_Raspi.py / sent_cameradate.py — ラズパイ側（カメラキャプチャと送信）の例
- controller_get.py — ジョイスティック（pygame）入力取得および送信ヘルパ
- socketmanager.py — 非同期で応答を受け取る補助
- README.md — 本ファイル

必要な依存（PC / Raspberry Pi 共通）
- Python 3.8+
- pip install opencv-python numpy pillow pyzbar
- PC側でジョイスティックを使うなら: pip install pygame
- 必要に応じて system のパッケージ（pyzbarのために libzbar 等）

推奨インストール例
```bash
python -m pip install opencv-python numpy pillow pyzbar pygame
# raspbian 等では zbar ライブラリが必要:
# sudo apt install libzbar0
```

設定（主にIP／ポート）
- カメラ送信ポート（デフォルト）: 36131
- コントローラ応答ポート（デフォルト）: 36132 / 36133（コード参照）
- 必要に応じて new_PC_operation.py / new_sent_Raspi.py 内の SERVER_IP / SERVER_PORT を編集

実行例
- ラズパイ（送信側）:
  - python sent_cameradate.py  または  python neo_sent_Raspi.py
- PC（受信側）:
  - python new_PC_operation.py

トラブルシューティング（片方のカメラが白画面になる等）
- 接続は受理されているが白画面:
  - camera_manager.py のログ（`Failed to decode image data.`）を確認。デコード失敗なら送信側のJPEG作成か送信途中で破損している可能性。
  - `update_loop` にある受信サイズ（先頭4バイト）と受信ループで完全受信できているか確認。途中で切れると画像が壊れる。
  - PhotoImageがガベージコレクトされないように `photo_var[0] = photo` が必要（既に実装済み）。
  - 表示対象のCanvasが非表示（pack_forget）か確認。非表示でも受信自体は行われるが見えないだけ。
  - カメラインデックス（例: 0 と 2）が正しいか確認。別カメラの映像が送られている可能性。
  - ネットワーク／ファイアウォール設定を確認。
- accept のタイムアウト例外:
  - socket のタイムアウトは捕捉して再試行する設計（socketmanager.py を参考）。必要ならタイムアウト値を調整。

パフォーマンス
- カメラ送信側で fps を 15 に設定（time.sleep(0.05) 等）しており、PC側での描画制御は camera_manager.py の `last_draw_time` によって制限可能。
- QR検出は負荷が高いため間隔（現在は last_qr_time）を調整すること。

開発メモ
- GUI は Tkinter でシングルスレッド。受信は別スレッドで行い、Tkinter 更新は `window.after(0, ...)` でメインスレッドに委譲している。
- 画像送信フォーマット: 先頭4バイトにデータ長（big-endian unsigned long）、続けてJPEGバイト列。

追加情報／改善案
- 受信デバッグ用に `update_loop` / `update_image` にログ（受信バイト数・デコード可否・処理時間）を追加すると原因切り分けが速いです。
- ネットワークの安定化（TCP_NODELAY, SO_SNDBUF などのソケットオプション調整）やフレームキューの調整で安定性が向上します。

ライセンス・注意
- 個人プロジェクト向け。外部に公開する場合は依存ライブラリのライセンスに注意すること。
