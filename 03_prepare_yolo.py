"""
03_prepare_yolo.py
Extract ~21K images + detection labels to disk for YOLO12m training.
- 3,000 images per disease class (A1-A6, 유증상) → 18K
- 3,000 무증상 (empty label = background) → 3K
- 500 per class for val → 3.5K
Label format: YOLO detection  →  0 cx cy w h  (normalized, class=0 skin_lesion)
Output: /home/lhj/CV/data/yolo_dataset/
"""
import pandas as pd
import zipfile
from pathlib import Path

DATA_DIR = Path("/home/lhj/CV/data")
CSV_PATH = DATA_DIR / "full_index.csv"
YOLO_DIR = DATA_DIR / "yolo_dataset"

N_PER_CLASS = 1500
N_NORMAL    = 1500
N_VAL_CLASS = 300

CNAMES = ['Papule','Scale','Lichen','Pustule','Erosion','Nodule','Normal']

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
df = df[(df['bbox_w'] > 0) & (df['img_w'] > 0)]

for split in ['train','val']:
    (YOLO_DIR/'images'/split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR/'labels'/split).mkdir(parents=True, exist_ok=True)

zip_cache = {}
def get_zip(path):
    if path not in zip_cache:
        zip_cache[path] = zipfile.ZipFile(path, 'r')
    return zip_cache[path]

def bbox_to_yolo(row):
    """Convert absolute bbox to normalized cx cy w h."""
    iw, ih = float(row['img_w']), float(row['img_h'])
    cx = (row['bbox_x'] + row['bbox_w'] / 2) / iw
    cy = (row['bbox_y'] + row['bbox_h'] / 2) / ih
    bw = row['bbox_w'] / iw
    bh = row['bbox_h'] / ih
    return min(max(cx,0),1), min(max(cy,0),1), min(max(bw,0),1), min(max(bh,0),1)

def extract(row, out_split):
    img_name = Path(row['img_inner']).name
    img_path = YOLO_DIR / 'images' / out_split / img_name
    lbl_path = YOLO_DIR / 'labels' / out_split / (Path(img_name).stem + '.txt')
    try:
        data = get_zip(row['img_zip']).read(row['img_inner'])
        with open(img_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        return False, str(e)
    if row['class_id'] == 6:
        lbl_path.write_text('')  # 무증상 = no lesion, empty label
    else:
        cx, cy, bw, bh = bbox_to_yolo(row)
        lbl_path.write_text(f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
    return True, ''

# --- Sample train ---
print("\nSampling train set...")
train_df = df[df['split'] == 'train']
selected = []
for cid in range(6):
    sub = train_df[train_df['class_id'] == cid]
    n   = min(N_PER_CLASS, len(sub))
    selected.append(sub.sample(n, random_state=42))
    print(f"  {CNAMES[cid]}: {n:,}")
normal  = train_df[train_df['class_id'] == 6]
n_norm  = min(N_NORMAL, len(normal))
selected.append(normal.sample(n_norm, random_state=42))
print(f"  Normal:  {n_norm:,}")
train_sel = pd.concat(selected).sample(frac=1, random_state=42)
print(f"Total train: {len(train_sel):,}")

# --- Sample val ---
print("\nSampling val set...")
val_df  = df[df['split'] == 'val']
val_parts = []
for cid in range(6):
    sub = val_df[val_df['class_id'] == cid]
    n   = min(N_VAL_CLASS, len(sub))
    val_parts.append(sub.sample(n, random_state=42))
vn = val_df[val_df['class_id'] == 6]
val_parts.append(vn.sample(min(N_VAL_CLASS, len(vn)), random_state=42))
val_sel = pd.concat(val_parts)
print(f"Total val: {len(val_sel):,}")

# --- Extract ---
for name, sel, split in [('train', train_sel, 'train'), ('val', val_sel, 'val')]:
    print(f"\nExtracting {name}...")
    ok = err = 0
    for _, row in sel.iterrows():
        success, msg = extract(row, split)
        if success: ok += 1
        else:        err += 1
        if (ok + err) % 1000 == 0:
            print(f"  {ok+err:,}/{len(sel):,} ({err} err)", flush=True)
    print(f"  Done: {ok:,} ok, {err} errors")

for zf in zip_cache.values():
    zf.close()

# --- dataset.yaml ---
yaml = f"""path: {YOLO_DIR}
train: images/train
val:   images/val

nc: 1
names: ['skin_lesion']
"""
with open(YOLO_DIR / 'dataset.yaml', 'w') as f:
    f.write(yaml)

print(f"\ndataset.yaml saved: {YOLO_DIR}/dataset.yaml")
print("YOLO12m detection data preparation complete!")
