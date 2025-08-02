# Text2Field v0.1

サッカーフィールドの要素を検出・セグメンテーションするコンピュータビジョンプロジェクト

## 概要

Text2Fieldは、サッカーフィールドの様々な要素を画像から検出・セグメンテーションするプロジェクトです。従来の画像処理手法と深層学習の両方のアプローチを実装しています。

## なぜ深層学習が必要か？

`cv_ellipse_detection.py`で実装されている従来の画像処理手法（HSVカラースペースとエッジ検出）では、以下の理由により精度が低いことが判明しました：

- 照明条件の変化に弱い
- フィールドの色のばらつきに対応できない
- ノイズや影の影響を受けやすい
- 部分的に隠れた円の検出が困難

そのため、より堅牢な深層学習アプローチ（YOLOv8）を採用しています。

## Roboflowでサッカー関連データセットを探す

[Roboflow Universe](https://universe.roboflow.com/)では、サッカー関連の様々なデータセットやモデルが公開されています：

- サッカーフィールドのセグメンテーション
- プレイヤー検出
- ボール追跡
- ゴール検出

検索キーワード例：`soccer field`, `football pitch`, `soccer segmentation`

## プロジェクトの使い方

### 1. 推論を実行する（DL_ellipse_detection.ipynb）

学習済みモデルを使って推論を実行：

```python
# DL_ellipse_detection.ipynbを開いて実行
from ultralytics import YOLO

# モデルを自動検出してロード
results = inference_jupyter("sample1.png")
```

### 2. 独自のモデルを構築する（DL_learning_model.ipynb）

自分のデータセットでモデルを学習する場合：

```python
# DL_learning_model.ipynbを開いて実行
trainer = YOLOv8MPSTrainerAuto(dataset_path="./field-6")

# システムに応じて自動的に最適化された設定で学習
model_path, results = trainer.train_adaptive(
    model_size='auto',  # メモリに基づいて自動選択
    epochs=100,
    imgsz=640
)
```

## プロジェクト構成

```
Text2Field_v0.1/
├── cv_ellipse_detection.py     # 従来手法（精度が低い例）
├── DL_ellipse_detection.ipynb  # 推論用ノートブック
├── DL_learning_model.ipynb     # モデル学習用ノートブック
├── field-6/                    # データセット
│   ├── train/                  # 訓練データ
│   ├── valid/                  # 検証データ
│   └── test/                   # テストデータ
├── sample1.png                 # サンプル画像
├── sample2.png
└── runs/                       # 学習済みモデル
```

## 検出可能なクラス

- 18ヤードボックス
- 18ヤード円弧
- 5ヤードボックス
- センターサークル（前半）
- フィールド（前半）
- センターサークル（後半）
- フィールド（後半）

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/RNMUDS/Text2Field_v0.1.git
cd Text2Field_v0.1

# 依存関係をインストール
pip install -r requirements.txt
```

## 主な特徴

- 🎯 7クラスのマルチクラスセグメンテーション
- 🚀 ハードウェアに応じた自動最適化（Apple Silicon/CUDA/CPU）
- 💾 メモリ容量に基づく自動設定調整
- 📊 Jupyter環境でのインタラクティブな推論
- 🔧 学習済みモデル付属

## システム要件

最小要件：
- 8GB RAM
- Python 3.8以上

推奨要件：
- 16GB以上のRAM
- GPU（CUDA対応またはApple Silicon）

## ライセンス

MIT License

## 謝辞

- データセット提供：[Roboflow](https://roboflow.com)
- 使用フレームワーク：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)