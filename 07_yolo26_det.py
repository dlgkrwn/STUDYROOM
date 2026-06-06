"""
07_yolo26_det.py
YOLO26m detection training — 병변 bbox 탐지 (Jan 2026 최신)
  - YOLO12m 대비 주요 개선:
    · NMS-Free End-to-End 추론 (후처리 제거)
    · DFL 제거 → ProgLoss + STAL (소형 객체 정확도↑)
    · MuSGD 옵티마이저 (SGD + Muon 결합)
    · CPU 추론 43% 빠름
  - Output: data/results/yolo/yolo26m_det/weights/best.pt
"""
import os, shutil, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATA_DIR     = Path("/home/lhj/CV/data")
RESULT_DIR   = DATA_DIR / "results/yolo"
WEIGHTS_DIR  = DATA_DIR / "weights"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_YAML = DATA_DIR / "yolo_dataset/dataset.yaml"

MODEL_NAME = "yolo26m.pt"
MODEL_PATH = WEIGHTS_DIR / MODEL_NAME

if not MODEL_PATH.exists():
    print(f"Downloading {MODEL_NAME}...")
    model_tmp = YOLO("yolo26m.pt")   # ultralytics auto-download
    shutil.move("yolo26m.pt", str(MODEL_PATH))
    print(f"Saved: {MODEL_PATH}")
else:
    print(f"Using cached: {MODEL_PATH} ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")

EPOCHS   = 50
IMGSZ    = 640
BATCH    = 16
PATIENCE = 20
RUN_NAME = "yolo26m_det"

print(f"\nTraining {MODEL_NAME}  epochs={EPOCHS}  batch={BATCH}  imgsz={IMGSZ}")
print("YOLO26 개선사항: NMS-Free / ProgLoss+STAL / MuSGD optimizer")

model = YOLO(str(MODEL_PATH))
results = model.train(
    data=str(DATASET_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    device=0,
    project=str(RESULT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    save=True,
    save_period=0,
    plots=True,
    verbose=True,
    # Augmentation (YOLO12m 동일 설정 유지)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    flipud=0.3,  fliplr=0.5,
    degrees=15,  scale=0.3,
)

# ── 학습 곡선 시각화 ──
results_csv = RESULT_DIR / RUN_NAME / "results.csv"
if results_csv.exists():
    import pandas as pd
    r = pd.read_csv(results_csv)
    r.columns = r.columns.str.strip()
    # YOLO26은 DFL Loss 제거됨 → 6개 지표에서 조정
    metrics = [
        ('train/box_loss',       'Train Box Loss',   '#2196F3'),
        ('train/cls_loss',       'Train Cls Loss',   '#FF9800'),
        ('metrics/mAP50(B)',     'mAP50',            '#4CAF50'),
        ('metrics/mAP50-95(B)',  'mAP50-95',         '#F44336'),
        ('metrics/precision(B)', 'Precision',        '#00BCD4'),
        ('metrics/recall(B)',    'Recall',            '#9C27B0'),
    ]
    available = [(c, t, col) for c, t, col in metrics if c in r.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('YOLO26m Detection Training Curves', fontsize=14, fontweight='bold')
    for ax, (col, title, color) in zip(axes.flatten(), available):
        ax.plot(r['epoch'], r[col], color=color, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    for ax in axes.flatten()[len(available):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'yolo26_training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: yolo26_training_curves.png")

# ── YOLO12m vs YOLO26m 비교 ──
yolo12_best = RESULT_DIR / "yolo12m_det/weights/best.pt"
yolo26_best = RESULT_DIR / f"{RUN_NAME}/weights/best.pt"

print("\n" + "="*50)
print("YOLO26m training complete!")
if yolo12_best.exists() and yolo26_best.exists():
    print("\n기존 YOLO12m 가중치: data/results/yolo/yolo12m_det/weights/best.pt")
    print("신규 YOLO26m 가중치: data/results/yolo/yolo26m_det/weights/best.pt")
    print("→ 06_app.py의 YOLO_W 경로를 yolo26m_det로 변경하면 앱에 적용됩니다.")
print("="*50)
