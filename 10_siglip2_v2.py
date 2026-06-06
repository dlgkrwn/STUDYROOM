"""
10_siglip2_v2.py
구진(0) + 인설/가피(1) 제거 → 5클래스 SigLIP2 개선 학습
클래스: 태선화, 농포/여드름, 미란/궤양, 결절/종괴, 정상

개선사항 vs 이전:
  - 모호한 두 클래스 제거 → 명확한 5클래스
  - 마지막 8 블록 언프리즈
  - 강화 증강: 색상변화 + 회전 + 수직반전
  - Focal Loss (gamma=2)
  - 15 에포크
"""
import os, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from pathlib import Path
import zipfile, io, random
from PIL import Image
import cv2

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATA_DIR   = Path("/home/lhj/CV/data")
INDEX_CSV  = DATA_DIR / "full_index_v3.csv"
PREV_CKPT  = DATA_DIR / "weights/siglip2_6class.pt"   # ep5 warm-start
SAVE_PATH  = DATA_DIR / "weights/siglip2_6class.pt"   # 덮어쓰기
SIGLIP2_ID = "google/siglip2-so400m-patch14-384"

NUM_CLASSES = 5
UNFREEZE_N  = 8        # 8 블록 언프리즈
BATCH       = 8        # GPU 여유 확보 후 증가
ACCUM_STEPS = 6        # effective batch = 48
EPOCHS      = 15
LR_HEAD     = 3e-4
LR_ENC      = 2e-6
WARMUP_EP   = 2
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# 구진(0) + 인설/가피(1) 제거, 나머지 재매핑
REMAP     = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
NEW_NAMES = ['태선화', '농포/여드름', '미란/궤양', '결절/종괴', '정상']


# ── Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── Dataset ──
class SkinDataset(Dataset):
    def __init__(self, df, processor, aug=False):
        self.df        = df.reset_index(drop=True)
        self.processor = processor
        self.aug       = aug
        self._zip_cache = {}

    def _load_img(self, zip_path, inner_path):
        if zip_path not in self._zip_cache:
            self._zip_cache[zip_path] = zipfile.ZipFile(zip_path, 'r')
        raw = self._zip_cache[zip_path].read(inner_path)
        return Image.open(io.BytesIO(raw)).convert('RGB')

    def _clahe(self, img):
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def _augment(self, img):
        # 수평/수직 반전
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 회전 ±15°
        if random.random() < 0.4:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))
        # 색상 변화 (PIL ImageEnhance)
        from PIL import ImageEnhance
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.4:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        return img

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = REMAP[int(row['class_id'])]
        try:
            img = self._load_img(str(row['img_zip']), str(row['img_inner']))
            img = self._clahe(img)
            if self.aug:
                img = self._augment(img)
        except Exception:
            img = Image.new('RGB', (384, 384), 128)
        inputs = self.processor(images=img, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in inputs.items()}, label


def collate(batch):
    inputs_list, labels = zip(*batch)
    keys    = inputs_list[0].keys()
    batched = {k: torch.stack([x[k] for x in inputs_list]) for k in keys}
    return batched, torch.tensor(labels, dtype=torch.long)


# ── 6-class 헤드 ──
class DiseaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Linear(1152, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def main():
    print(f"Device: {DEVICE}")
    print(f"클래스: {NEW_NAMES}")
    print(f"언프리즈: 마지막 {UNFREEZE_N} 블록 | Focal Loss | 증강 강화")

    # ── 데이터 (구진 class_id=0 제거) ──
    df       = pd.read_csv(INDEX_CSV)
    df       = df[~df['class_id'].isin([0, 1])]  # 구진 + 인설/가피 제거
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    test_df  = df[df['split'] == 'test']

    print("\n=== 클래스 분포 (train) ===")
    mapped = train_df['class_id'].map(REMAP)
    for i, n in enumerate(NEW_NAMES):
        print(f"  {i} {n}: {(mapped == i).sum()}개")

    processor = AutoProcessor.from_pretrained(SIGLIP2_ID)
    train_ds  = SkinDataset(train_df, processor, aug=True)
    val_ds    = SkinDataset(val_df,   processor, aug=False)
    test_ds   = SkinDataset(test_df,  processor, aug=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH,   shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH*2, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate)

    # ── 모델 로드 ──
    print(f"\n모델 로드: {SIGLIP2_ID}")
    full_model = AutoModel.from_pretrained(SIGLIP2_ID)
    encoder    = full_model.vision_model.to(DEVICE)
    total_blocks = len(encoder.encoder.layers)
    print(f"  전체 블록: {total_blocks}, 언프리즈: 마지막 {UNFREEZE_N}개")

    # 이전 체크포인트 encoder warm-start (헤드는 클래스 수 달라서 제외)
    if PREV_CKPT.exists():
        ckpt   = torch.load(str(PREV_CKPT), map_location=DEVICE)
        enc_sd = encoder.state_dict()
        prev_blocks = {k: v.float() for k, v in ckpt['encoder_last_blocks'].items()}
        enc_sd.update(prev_blocks)
        encoder.load_state_dict(enc_sd)
        print(f"  인코더 가중치 적용 (이전 val_acc={ckpt['val_acc']:.4f})")

    # 전체 동결 후 마지막 UNFREEZE_N 블록 + post_layernorm 해제
    for p in encoder.parameters(): p.requires_grad_(False)
    for layer in encoder.encoder.layers[-UNFREEZE_N:]:
        for p in layer.parameters(): p.requires_grad_(True)
    for p in encoder.post_layernorm.parameters(): p.requires_grad_(True)

    head = DiseaseHead().to(DEVICE)

    # ── Focal Loss ──
    mapped_train = train_df['class_id'].map(REMAP).values
    counts       = np.bincount(mapped_train, minlength=NUM_CLASSES).astype(float)
    weights      = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
    weights      = weights / weights.sum() * NUM_CLASSES
    criterion    = FocalLoss(gamma=2.0, weight=weights)
    print(f"\n클래스 가중치: {weights.cpu().numpy().round(3)}")

    # ── 옵티마이저 ──
    enc_params  = [p for p in encoder.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer   = torch.optim.AdamW([
        {'params': enc_params,  'lr': LR_ENC,  'weight_decay': 1e-4},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 1e-3},
    ])

    total_steps  = len(train_dl) * EPOCHS // ACCUM_STEPS
    warmup_steps = len(train_dl) * WARMUP_EP // ACCUM_STEPS
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler()

    # ── 학습 루프 ──
    best_acc = 0.0
    best_ep  = 0

    for ep in range(1, EPOCHS + 1):
        encoder.train(); head.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dl, 1):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                feats  = encoder(**inputs).pooler_output
                logits = head(feats)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUM_STEPS

            if step % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        # ── 검증 ──
        encoder.eval(); head.eval()
        correct = total = 0
        class_correct = np.zeros(NUM_CLASSES)
        class_total   = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    feats  = encoder(**inputs).pooler_output
                    logits = head(feats)
                preds  = logits.argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)
                for i in range(NUM_CLASSES):
                    mask = (labels == i)
                    class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                    class_total[i]   += mask.sum().item()

        val_acc  = correct / total
        avg_loss = total_loss / len(train_dl)
        print(f"\nEp {ep:02d}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")
        for i, n in enumerate(NEW_NAMES):
            if class_total[i] > 0:
                print(f"  {n}: {class_correct[i]:.0f}/{class_total[i]:.0f} = "
                      f"{class_correct[i]/class_total[i]*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc; best_ep = ep
            enc_sd  = encoder.state_dict()
            save_sd = {k: v.half() for k, v in enc_sd.items()
                       if any(f'encoder.layers.{i}.' in k
                              for i in range(total_blocks - UNFREEZE_N, total_blocks))
                       or 'post_layernorm' in k}
            torch.save({
                'head':                head.state_dict(),
                'encoder_last_blocks': save_sd,
                'unfreeze_n':          UNFREEZE_N,
                'total_blocks':        total_blocks,
                'val_acc':             val_acc,
                'epoch':               ep,
                'num_classes':         NUM_CLASSES,
                'remap':               REMAP,
                'class_names':         NEW_NAMES,
            }, SAVE_PATH)
            print(f"  *** Best: {best_acc:.4f} → siglip2_6class.pt 저장")

    # ── 테스트 ──
    print(f"\n=== 최종 테스트 (best: Ep{best_ep}, val={best_acc:.4f}) ===")
    ckpt    = torch.load(str(SAVE_PATH), map_location=DEVICE)
    enc_sd  = encoder.state_dict()
    enc_sd.update({k: v.float() for k, v in ckpt['encoder_last_blocks'].items()})
    encoder.load_state_dict(enc_sd)
    head.load_state_dict(ckpt['head'])
    encoder.eval(); head.eval()

    correct = total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total   = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            all_logits = []
            for aug_fn in [
                lambda x: x,
                lambda x: torch.flip(x, [-1]),
                lambda x: torch.flip(x, [-2]),
                lambda x: torch.rot90(x, 2, [-2, -1]),
            ]:
                aug_inp = {k: (aug_fn(v) if v.dim() == 4 else v)
                           for k, v in inputs.items()}
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    feats  = encoder(**aug_inp).pooler_output
                    logits = head(feats)
                all_logits.append(logits)
            logits = torch.stack(all_logits).mean(0)
            preds  = logits.argmax(1).cpu()
            correct += (preds == labels).sum().item()
            total   += len(labels)
            for i in range(NUM_CLASSES):
                mask = (labels == i)
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                class_total[i]   += mask.sum().item()

    test_acc = correct / total
    print(f"\n최종 테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("클래스별:")
    for i, n in enumerate(NEW_NAMES):
        if class_total[i] > 0:
            print(f"  {n}: {class_correct[i]:.0f}/{class_total[i]:.0f} = "
                  f"{class_correct[i]/class_total[i]*100:.1f}%")
    print(f"\n저장 위치: {SAVE_PATH}")
    print(f"파일 크기: {SAVE_PATH.stat().st_size/1024/1024:.1f} MB")
    print("Training complete")


if __name__ == '__main__':
    main()
