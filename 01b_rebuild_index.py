"""
01b_rebuild_index.py
데이터 전처리 재구성
  - Split: 70/15/15 (stratified by class + species)
  - bbox < 1% 이미지 면적 제거
  - 클래스당 최대 5,000개 cap (A7 Normal 포함)
  - 신체 4분류 → 8분류 세분화
  - 개/고양이(species) feature 추가
  - 결과: /home/lhj/CV/data/full_index_v2.csv
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR  = Path("/home/lhj/CV/data")
CSV_IN    = DATA_DIR / "full_index.csv"
CSV_OUT   = DATA_DIR / "full_index_v3.csv"
CLASS_CAP = 10_000  # 클래스당 최대 샘플 수
BBOX_MIN  = 0.01    # bbox 최소 면적 비율 (1%)

DISEASE_NAMES = ['A1_Papule','A2_Scale','A3_Lichen','A4_Pustule',
                 'A5_Erosion','A6_Nodule','A7_Normal']

# ── 신체 8분류 매핑 ──
# 기존 4분류(H/B/L/A) → 8분류
# region_id(4) + 세부 region_id(8) 모두 저장
REGION_MAP_KR = {
    'H': {'id4': 0, 'name4': '머리', 'name4_full': '머리 (얼굴·귀·목)'},
    'B': {'id4': 1, 'name4': '몸통', 'name4_full': '몸통 (등·옆구리)'},
    'L': {'id4': 2, 'name4': '사지', 'name4_full': '사지 (다리·발)'},
    'A': {'id4': 3, 'name4': '복부', 'name4_full': '복부 (배)'},
}

# 8분류: 이미지 내 bbox 위치로 세분화
# Head (H) → 상단: 얼굴/귀, 하단: 목
# Body (B) → 상단: 등, 하단: 옆구리/가슴
# Limb (L) → 앞발, 뒷발
# Abdomen (A) → 복부 단일 유지
# bbox의 y중심 기준으로 상/하 구분

REGION8_NAMES = [
    '얼굴·귀',    # 0: H + bbox 상단
    '목',         # 1: H + bbox 하단
    '등',         # 2: B + bbox 상단
    '옆구리·가슴', # 3: B + bbox 하단
    '앞발·앞다리', # 4: L + bbox 좌측/상단
    '뒷발·뒷다리', # 5: L + bbox 우측/하단
    '복부',        # 6: A
    '기타',        # 7: 분류 불가
]


def assign_region8(row):
    """bbox 위치 기반으로 8분류 할당"""
    region = row['region']
    # bbox 중심의 이미지 내 상대 위치
    cx_ratio = (row['bbox_x'] + row['bbox_w']/2) / max(row['img_w'], 1)
    cy_ratio = (row['bbox_y'] + row['bbox_h']/2) / max(row['img_h'], 1)

    if region == 'H':
        return 0 if cy_ratio < 0.5 else 1    # 상=얼굴·귀, 하=목
    elif region == 'B':
        return 2 if cy_ratio < 0.45 else 3   # 상=등, 하=옆구리
    elif region == 'L':
        return 4 if cx_ratio < 0.5 else 5    # 좌=앞발, 우=뒷발
    elif region == 'A':
        return 6
    else:
        return 7


print("=" * 60)
print("데이터 전처리 재구성 (v2)")
print("=" * 60)

# ── 1. 원본 로드 ──
print("\n[1] 원본 데이터 로드...")
df = pd.read_csv(CSV_IN)
print(f"  원본: {len(df):,}개")

# ── 2. bbox 면적 필터 ──
print("\n[2] bbox 면적 필터 (< 1% 제거)...")
df['bbox_ratio'] = (df['bbox_w'] * df['bbox_h']) / (df['img_w'] * df['img_h'])
before = len(df)
df = df[df['bbox_ratio'] >= BBOX_MIN].reset_index(drop=True)
print(f"  제거: {before - len(df):,}개 | 남은 것: {len(df):,}개")

# ── 3. 클래스별 분포 확인 ──
print("\n[3] 클래스별 분포 (필터 후):")
for i, n in enumerate(DISEASE_NAMES):
    cnt = len(df[df['class_id'] == i])
    print(f"  {n}: {cnt:,}")

# ── 4. 클래스당 cap 적용 (균형 샘플링) ──
print(f"\n[4] 클래스당 최대 {CLASS_CAP:,}개 cap 적용...")
dfs = []
for cid in range(7):
    cls_df = df[df['class_id'] == cid]
    if len(cls_df) > CLASS_CAP:
        # 개/고양이 비율 유지하며 샘플링 (groupby 대신 수동 처리)
        n_total = len(cls_df)
        parts = []
        for sp in cls_df['species'].unique():
            sp_df = cls_df[cls_df['species'] == sp]
            n_take = min(len(sp_df), int(CLASS_CAP * len(sp_df) / n_total))
            parts.append(sp_df.sample(n_take, random_state=42))
        sampled = pd.concat(parts)
        # 부족분 채우기
        if len(sampled) < CLASS_CAP:
            remain = cls_df[~cls_df.index.isin(sampled.index)]
            extra  = remain.sample(min(CLASS_CAP - len(sampled), len(remain)), random_state=42)
            sampled = pd.concat([sampled, extra])
        dfs.append(sampled.head(CLASS_CAP))
    else:
        dfs.append(cls_df)

df_cap = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Cap 후 총 샘플: {len(df_cap):,}")
for i, n in enumerate(DISEASE_NAMES):
    cnt = len(df_cap[df_cap['class_id'] == i])
    d_cnt = len(df_cap[(df_cap['class_id']==i) & (df_cap['species']=='D')])
    c_cnt = len(df_cap[(df_cap['class_id']==i) & (df_cap['species']=='C')])
    print(f"  {n}: {cnt:,} (개 {d_cnt:,} / 고양이 {c_cnt:,})")

# ── 5. 신체 8분류 추가 ──
print("\n[5] 신체 8분류 할당...")
df_cap['region8_id'] = df_cap.apply(assign_region8, axis=1)
df_cap['region4_name'] = df_cap['region'].map(lambda r: REGION_MAP_KR.get(r, {}).get('name4_full', r))
df_cap['region8_name'] = df_cap['region8_id'].map(lambda i: REGION8_NAMES[i])
df_cap['species_id'] = (df_cap['species'] == 'C').astype(int)  # 0=개, 1=고양이

print("  8분류 분포:")
for i, n in enumerate(REGION8_NAMES):
    cnt = len(df_cap[df_cap['region8_id'] == i])
    print(f"  [{i}] {n}: {cnt:,}")

# ── 6. 70/15/15 Split ──
print("\n[6] 70/15/15 Stratified Split...")

# stratify key: class_id + species 조합
df_cap['strat_key'] = df_cap['class_id'].astype(str) + '_' + df_cap['species']

# 먼저 train(70%) vs temp(30%) 분리
train_df, temp_df = train_test_split(
    df_cap, test_size=0.30,
    stratify=df_cap['strat_key'],
    random_state=42
)
# temp를 val(15%) vs test(15%) 분리
val_df, test_df = train_test_split(
    temp_df, test_size=0.50,
    stratify=temp_df['strat_key'],
    random_state=42
)

train_df = train_df.copy(); train_df['split'] = 'train'
val_df   = val_df.copy();   val_df['split']   = 'val'
test_df  = test_df.copy();  test_df['split']  = 'test'

df_final = pd.concat([train_df, val_df, test_df]).sample(frac=1, random_state=42).reset_index(drop=True)
df_final.drop(columns=['strat_key'], inplace=True)

print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_final)*100:.1f}%)")
print(f"  Val:   {len(val_df):,}   ({len(val_df)/len(df_final)*100:.1f}%)")
print(f"  Test:  {len(test_df):,}  ({len(test_df)/len(df_final)*100:.1f}%)")

# ── 7. Split 내 클래스 균형 확인 ──
print("\n[7] Split별 클래스 분포 확인:")
for s in ['train', 'val', 'test']:
    s_df = df_final[df_final['split'] == s]
    counts = [len(s_df[s_df['class_id']==i]) for i in range(7)]
    print(f"  {s}: {counts}")

# ── 8. 저장 ──
df_final.to_csv(CSV_OUT, index=False)
print(f"\n[완료] 저장: {CSV_OUT}")
print(f"  총 샘플: {len(df_final):,}")
print(f"  컬럼: {list(df_final.columns)}")

# ── 9. 최종 요약 ──
print("\n" + "="*60)
print("최종 요약")
print("="*60)
print(f"  원본 → {before:,}개")
print(f"  bbox 필터 후 → {len(df):,}개")
print(f"  클래스 균형 후 → {len(df_cap):,}개")
print(f"  최종 (split 포함) → {len(df_final):,}개")
print(f"\n  신체 8분류: {REGION8_NAMES}")
print(f"  새 컬럼: region8_id, region8_name, region4_name, species_id")
print(f"\n  이전 대비 학습 속도: 약 {len(df[df['split']=='train'])//len(train_df)}배 빠름")
