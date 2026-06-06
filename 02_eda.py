"""
02_eda.py - EDA visualization from full_index.csv
Output: /home/lhj/CV/data/eda_figures/
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import zipfile, cv2, random
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

DATA_DIR = Path("/home/lhj/CV/data")
CSV_PATH = DATA_DIR / "full_index.csv"
OUT_DIR  = DATA_DIR / "eda_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CNAMES  = ['A1\nPapule', 'A2\nScale', 'A3\nLichen', 'A4\nPustule', 'A5\nErosion', 'A6\nNodule', 'A7\nNormal']
CLABELS = ['A1_Papule','A2_Scale','A3_Lichen','A4_Pustule','A5_Erosion','A6_Nodule','A7_Normal']
RNAMES  = {'H':'Head(H)','B':'Body(B)','L':'Limb(L)','A':'Abdomen(A)'}
SPLITS  = ['train','val','test']
SCOLORS = {'train':'#2196F3','val':'#FF9800','test':'#4CAF50'}

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"Total: {len(df):,}")
print(df['split'].value_counts())

# Fig1: Class distribution by split
fig, axes = plt.subplots(1,3,figsize=(18,5))
fig.suptitle('Disease Class Distribution by Split', fontsize=14, fontweight='bold')
for ax, split in zip(axes, SPLITS):
    sub  = df[df['split']==split]
    cnts = [len(sub[sub['class_id']==i]) for i in range(7)]
    bars = ax.bar(range(7), cnts, color=SCOLORS[split], edgecolor='white', width=0.7)
    ax.set_xticks(range(7)); ax.set_xticklabels(CNAMES, fontsize=8)
    ax.set_title(f'{split.upper()} (n={len(sub):,})'); ax.set_ylabel('Count')
    for bar,cnt in zip(bars,cnts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(cnts)*0.01,
                f'{cnt:,}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR/'fig1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig2: Region distribution
fig, axes = plt.subplots(1,3,figsize=(14,4))
fig.suptitle('Body Region Distribution by Split', fontsize=14, fontweight='bold')
for ax, split in zip(axes, SPLITS):
    sub = df[df['split']==split]
    rc  = sub['region'].value_counts()
    ax.pie(rc.values, labels=[RNAMES.get(r,r) for r in rc.index],
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'{split.upper()} (n={len(sub):,})')
plt.tight_layout()
plt.savefig(OUT_DIR/'fig2_region_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig3: Species distribution
fig, ax = plt.subplots(figsize=(8,4))
sp_split = df.groupby(['split','species']).size().unstack(fill_value=0)
sp_split.plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4'])
ax.set_title('Species Distribution by Split', fontsize=13, fontweight='bold')
ax.set_xlabel('Split'); ax.set_ylabel('Count')
ax.legend(['Dog (D)','Cat (C)']); ax.set_xticklabels(SPLITS, rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR/'fig3_species.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig4: Class x Region heatmap (train)
trn = df[df['split']=='train']
hm  = trn.groupby(['class_id','region']).size().unstack(fill_value=0)
hm  = hm.reindex(columns=['H','B','L','A'], fill_value=0)
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(hm.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels([RNAMES[r] for r in ['H','B','L','A']])
ax.set_yticks(range(7)); ax.set_yticklabels(CLABELS, fontsize=9)
plt.colorbar(im, ax=ax, label='Count')
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j,i,f'{hm.values[i,j]:,}',ha='center',va='center',fontsize=8,
                color='white' if hm.values[i,j]>hm.values.max()*0.6 else 'black')
ax.set_title('Disease x Region Heatmap (Train)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/'fig4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4")

# Fig5: Sample images (5 per disease class)
print("Loading sample images from zips...")
zip_cache = {}
def load_img(img_zip, img_inner, size=112):
    if img_zip not in zip_cache:
        zip_cache[img_zip] = zipfile.ZipFile(img_zip,'r')
    zf = zip_cache[img_zip]
    try:
        data = zf.read(img_inner)
        arr  = np.frombuffer(data, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img,(size,size))
    except:
        return np.zeros((size,size,3),dtype=np.uint8)

N=5
fig, axes = plt.subplots(7,N,figsize=(N*2.2,7*2.2))
fig.suptitle('Sample Images per Disease Class (Train)', fontsize=13, fontweight='bold')
for cid in range(7):
    sub = trn[trn['class_id']==cid]
    smp = sub.sample(min(N,len(sub)), random_state=42)
    for j,(_, row) in enumerate(smp.iterrows()):
        axes[cid][j].imshow(load_img(row['img_zip'], row['img_inner']))
        axes[cid][j].axis('off')
        if j==0:
            axes[cid][j].set_ylabel(CLABELS[cid], fontsize=7,
                rotation=0, ha='right', va='center', labelpad=55)
plt.tight_layout()
plt.savefig(OUT_DIR/'fig5_samples.png', dpi=110, bbox_inches='tight')
plt.close()
for zf in zip_cache.values(): zf.close()
print("Saved fig5")

# Class weights
print("\n=== Class Weights (train) ===")
cc = trn['class_id'].value_counts().sort_index()
mx = cc.max()
weights = {}
for i in range(7):
    cnt = cc.get(i,1)
    w   = round(mx/cnt, 3)
    weights[i] = w
    print(f"  {CLABELS[i]}: count={cnt:,}  weight={w:.3f}")

summary_lines = [f"Total: {len(df):,}\n"]
for s in SPLITS:
    summary_lines.append(f"  {s}: {len(df[df['split']==s]):,}\n")
summary_lines.append("\nClass Weights (train):\n")
for i,w in weights.items():
    summary_lines.append(f"  class_{i} ({CLABELS[i]}): {w:.3f}\n")
with open(OUT_DIR/'eda_summary.txt','w') as f:
    f.writelines(summary_lines)

print(f"\nAll figures saved: {OUT_DIR}")
print("EDA complete!")
