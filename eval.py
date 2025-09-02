import os
import os.path as osp
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv

from dataset import MultiVideoAnomalyDataset
from model.model import GRUAnomalyDetector

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ────────────────────────────────────────────────────────────────────────────
# 설정
LABEL_DIR   = '/home/jaehyeon/Desktop/졸작/BTS_graduate_project/eval'
FPS         = 30
SEQ_LEN     = 8
BATCH_SIZE  = 4
MODEL_PATH  = 'anomaly_multi_balanced_epoch24.pth'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ────────────────────────────────────────────────────────────────────────────
# 결과 저장 폴더
RESULT_DIR = 'Results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# JSON 존재 확인
json_paths = glob.glob(os.path.join(LABEL_DIR, '*.json'))
if not json_paths:
    raise FileNotFoundError(f"No .json files found in {LABEL_DIR}")

# ────────────────────────────────────────────────────────────────────────────
# 모델 로드
model = GRUAnomalyDetector().to(DEVICE)
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)  # PyTorch 2.4+
except TypeError:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# 모델 출력이 확률([0,1])이라고 가정 → BCELoss
criterion = nn.BCELoss()

# ────────────────────────────────────────────────────────────────────────────
# 헬퍼: frame_meta에서 "윈도우의 끝 프레임(e_frame)"만 안전 추출
def _extract_window_end(meta_item, seq_len=SEQ_LEN):
    """
    meta_item 예시:
      - [f0, f1, ..., f_{L-1}] : 마지막 원소를 end로 사용
      - {'end': e} 또는 {'e': e} : 그대로 사용
      - {'start': s} 또는 {'s': s} : s + (L-1)
      - 단일 int(센터/시작 등으로 저장했다면) : 값 + (L-1)
    실패 시 None 반환
    """
    if isinstance(meta_item, (list, tuple)) and len(meta_item) > 0:
        return int(meta_item[-1])
    if isinstance(meta_item, dict):
        if 'end' in meta_item:
            return int(meta_item['end'])
        if 'e' in meta_item:
            return int(meta_item['e'])
        if 'last' in meta_item:
            return int(meta_item['last'])
        if 'center' in meta_item:
            return int(meta_item['center']) + (seq_len // 2)
        if 'start' in meta_item or 's' in meta_item:
            s = int(meta_item.get('start', meta_item.get('s')))
            return s + seq_len - 1
    if isinstance(meta_item, (int, np.integer)):
        return int(meta_item) + seq_len - 1
    return None

# ────────────────────────────────────────────────────────────────────────────
# 윈도우 단위 타임라인 플롯 (end-aligned)
def _plot_timeline_window(vid, t_end, gt_win, prob_win, pred_win, out_dir, threshold=0.5):
    if len(t_end) == 0:
        return None

    # 시간 기준 정렬
    order = np.argsort(t_end)
    t_end   = np.asarray(t_end, dtype=float)[order]
    gt_win  = np.asarray(gt_win, dtype=float)[order]
    pred_win= np.asarray(pred_win, dtype=float)[order]
    prob_win= np.asarray(prob_win, dtype=float)[order]

    fig, ax = plt.subplots(figsize=(12, 4))
    # 윈도우 "끝 시각"에 값을 붙여서 step(post)로 그리면 겹침 없이 1:1 표현
    ax.step(t_end, gt_win,   where='post', linewidth=2, label='GT (window)')
    ax.step(t_end, pred_win, where='post', linewidth=2, linestyle='--', label='Pred (0/1)')
    ax.plot(t_end, prob_win, linewidth=1.5, alpha=0.9, label='Pred Prob')
    ax.axhline(threshold, linestyle=':', linewidth=1, label=f'Threshold={threshold}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score / Class')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Result — {vid}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    save_png = osp.join(out_dir, f'{vid}_timeline_window.png')
    fig.tight_layout()
    fig.savefig(save_png, dpi=200)
    plt.close(fig)

    # CSV 저장 (윈도우 끝 시각 기준)
    save_csv = osp.join(out_dir, f'{vid}_timeline_window.csv')
    with open(save_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_end(s)', 'gt_window', 'pred_window', 'prob'])
        for te, g, p, pr in zip(t_end, gt_win, pred_win, prob_win):
            w.writerow([f"{te:.6f}", int(g), int(p), f"{pr:.6f}"])

    return save_png

# ────────────────────────────────────────────────────────────────────────────
# 멀티 비디오 데이터셋 (디렉토리 전체 평가)
eval_dataset = MultiVideoAnomalyDataset(
    label_dir   = LABEL_DIR,
    seq_len     = SEQ_LEN,
    default_roi = (0, 0, 999999, 999999),   # 전체 프레임 사용
    transform   = None
)
if len(eval_dataset) == 0:
    raise RuntimeError(
        "MultiVideoAnomalyDataset has 0 samples. "
        "SEQ_LEN / ROI / 라벨 파일 내용을 확인해 주세요."
    )

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ────────────────────────────────────────────────────────────────────────────
# 평가 루프 (완전 '윈도우 단위')
total_loss = 0.0
total_elems = 0
all_probs, all_preds, all_labels = [], [], []

# 비디오별 윈도우-종단시각 수집
per_video_win = defaultdict(lambda: {'t_end': [], 'gt': [], 'prob': [], 'pred': []})

samples_ref = getattr(eval_dataset, 'samples', None)  # (video_path, frame_meta, label, roi) 가정
if samples_ref is None:
    raise RuntimeError("eval_dataset.samples 를 찾을 수 없습니다. MultiVideoAnomalyDataset에 samples 속성이 필요합니다.")

global_idx = 0  # DataLoader 순회시 현재 샘플의 전역 인덱스

with torch.no_grad():
    for seqs, labels in tqdm(eval_loader, desc='Eval (multi-video, window-level)', unit='batch'):
        seqs   = seqs.to(DEVICE)
        labels = labels.to(DEVICE).float().reshape(-1)     # [B]

        probs = model(seqs).float().reshape(-1)            # [B]
        if probs.shape != labels.shape:
            raise RuntimeError(f"Shape mismatch: probs {probs.shape}, labels {labels.shape}")

        loss = criterion(probs, labels)
        total_loss  += loss.item() * labels.numel()
        total_elems += labels.numel()

        preds = (probs >= 0.5).float()

        # 전체 메트릭용 누적
        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        # ── 영상별: 윈도우 "끝 시각"만 기록 (겹침 제거, 1:1 정렬)
        bs = probs.numel()
        for i in range(bs):
            if global_idx >= len(samples_ref):
                break

            sample_tuple = samples_ref[global_idx]
            video_path = sample_tuple[0]
            frame_meta = sample_tuple[1] if len(sample_tuple) > 1 else None
            vid = osp.splitext(osp.basename(video_path))[0]

            e_frame = _extract_window_end(frame_meta, SEQ_LEN)
            if e_frame is None:
                # 같은 비디오 안에서 이전 end가 있다면 +1 프레임, 없으면 첫 윈도우 가정
                rec = per_video_win[vid]
                if len(rec['t_end']) == 0:
                    e_frame = SEQ_LEN - 1
                else:
                    # 직전 t_end를 프레임으로 환산 후 +1
                    last_e_frame = int(round(rec['t_end'][-1] * FPS)) - 1
                    e_frame = last_e_frame + 1

            # 윈도우 끝 "직후" 시각(초) = (e_frame + 1) / FPS
            t_end = float(e_frame + 1) / float(FPS)

            per_video_win[vid]['t_end'].append(t_end)
            per_video_win[vid]['gt'].append(float(labels[i].item()))
            per_video_win[vid]['prob'].append(float(probs[i].item()))
            per_video_win[vid]['pred'].append(float(preds[i].item()))

            global_idx += 1

# ────────────────────────────────────────────────────────────────────────────
# 메트릭 계산 (윈도우 단위 평균)
avg_loss  = float(total_loss / total_elems) if total_elems > 0 else float('nan')
precision = float(precision_score(all_labels, all_preds, zero_division=0))
recall    = float(recall_score(all_labels, all_preds, zero_division=0))
f1        = float(f1_score(all_labels, all_preds, zero_division=0))
try:
    roc_auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else float('nan')
except ValueError:
    roc_auc = float('nan')

metrics_mean = {
    'Avg Loss' : avg_loss,
    'Precision': precision,
    'Recall'   : recall,
    'F1 Score' : f1,
    'ROC AUC'  : roc_auc,
}

# 저장
with open(os.path.join(RESULT_DIR, 'overall_metrics.json'), 'w') as f:
    json.dump(metrics_mean, f, indent=2)

# ────────────────────────────────────────────────────────────────────────────
# 레이더 차트 (0~1 스케일 지표만)
radar_keys = ['Precision', 'Recall', 'F1 Score', 'ROC AUC']
radar_vals = []
for k in radar_keys:
    v = metrics_mean[k]
    if np.isnan(v):
        v = 0.0
    radar_vals.append(float(max(0.0, min(1.0, v))))

angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
angles += angles[:1]
radar_vals += radar_vals[:1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, radar_vals, 'o-', linewidth=2)
ax.fill(angles, radar_vals, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), radar_keys, fontsize=14)
ax.set_ylim(0, 1)
ax.set_title('Overall Average Metrics (Window-level)', y=1.1, fontsize=16)

for a, v in zip(angles[:-1], radar_vals[:-1]):
    ax.text(a, min(1.0, v + 0.05), f"{v:.2f}", ha='center', va='center', fontsize=12)

out_path = os.path.join(RESULT_DIR, 'overall_metrics.png')
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved overall radar plot → {out_path}")
print("Saved overall metrics →", os.path.join(RESULT_DIR, 'overall_metrics.json'))

# ────────────────────────────────────────────────────────────────────────────
# 비디오별 '윈도우 단위(end-aligned)' 타임라인 저장
TL_DIR = osp.join(RESULT_DIR, 'timelines_window')
os.makedirs(TL_DIR, exist_ok=True)

print("\nSaving per-video window-aligned timelines...")
saved_cnt = 0
for vid, rec in per_video_win.items():
    out_png = _plot_timeline_window(
        vid,
        rec['t_end'],
        rec['gt'],
        rec['prob'],
        rec['pred'],
        TL_DIR,
        threshold=0.5
    )
    if out_png:
        print(f"  - {vid}: {out_png}")
        saved_cnt += 1

if saved_cnt == 0:
    print("  (No timelines were generated.)")
else:
    print(f"Done. {saved_cnt} timeline(s) saved under {TL_DIR}")
