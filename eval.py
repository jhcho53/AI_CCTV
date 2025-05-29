import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import SingleVideoAnomalyDataset
from model.model import GRUAnomalyDetector

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ────────────────────────────────────────────────────────────────────────────
LABEL_DIR   = '/home/jaehyeon/Desktop/졸작/BTS_graduate_project/eval'
FPS         = 30
SEQ_LEN     = 8
BATCH_SIZE  = 64
MODEL_PATH  = 'anomaly_multi_balanced_epoch2.pth'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = GRUAnomalyDetector().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
criterion = nn.BCELoss()

# JSON 파일 리스트
json_paths = glob.glob(os.path.join(LABEL_DIR, '*.json'))
if not json_paths:
    raise FileNotFoundError(f"No .json files found in {LABEL_DIR}")

# 결과 저장 폴더
RESULT_DIR = 'Results'
os.makedirs(RESULT_DIR, exist_ok=True)

# 전체 평균용 메트릭 수집 리스트
loss_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []

# ────────────────────────────────────────────────────────────────────────────
# 파일별 평가 & 저장
for ann_path in json_paths:
    basename = os.path.splitext(os.path.basename(ann_path))[0]

    eval_dataset = SingleVideoAnomalyDataset(ann_path=ann_path, seq_len=SEQ_LEN, fps=FPS)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for seqs, labels in tqdm(eval_loader, desc=f'Eval {basename}', unit='batch'):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            probs = model(seqs).squeeze()
            total_loss += criterion(probs, labels).item() * seqs.size(0)
            preds = (probs >= 0.5).float()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 메트릭 계산
    avg_loss  = float(total_loss / len(eval_dataset))
    precision = float(precision_score(all_labels, all_preds, zero_division=0))
    recall    = float(recall_score(all_labels, all_preds, zero_division=0))
    f1        = float(f1_score(all_labels, all_preds, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        roc_auc = float('nan')

    # 리스트에 저장
    loss_list.append(avg_loss)
    prec_list.append(precision)
    rec_list.append(recall)
    f1_list.append(f1)
    auc_list.append(roc_auc)

    # (기존) JSON, bar chart 저장 생략...

# ────────────────────────────────────────────────────────────────────────────
# 폴더 전체 평균 메트릭 계산
metrics_mean = {
    'Avg Loss' : np.mean(loss_list),
    'Precision': np.mean(prec_list),
    'Recall'   : np.mean(rec_list),
    'F1 Score' : np.mean(f1_list),
    'ROC AUC'  : np.mean(auc_list),
}

# ────────────────────────────────────────────────────────────────────────────
# 레이더 차트 그리기
labels = list(metrics_mean.keys())
values = list(metrics_mean.values())
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2)
ax.fill(angles, values, alpha=0.25)

# 축 레이블 바깥으로 밀기
label_pad = 30
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)
ax.tick_params(axis='x', which='major', pad=label_pad)

# 수치 표시
for angle, value in zip(angles, values):
    ax.text(angle, value + 0.05, f"{value:.2f}", ha='center', va='center', fontsize=12)

ax.set_ylim(0, 1)
ax.set_title('Overall Average Metrics (All Files)', y=1.1, fontsize=16)

# 저장
out_path = os.path.join(RESULT_DIR, 'overall_metrics.png')
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved overall radar plot → {out_path}")
