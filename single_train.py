import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import SingleVideoAnomalyDataset
from model import GRUAnomalyDetector

# ────────────────────────────────────────────────────────────────────────────
class BalancedBatchSampler(Sampler):
    """
    매 배치마다 정상/이상 샘플을 반반씩 뽑는 Sampler,
    데이터 불균형 시에도 샘플 수 부족하면 복원추출(중복허용)로 보완합니다.

    Args:
        labels (list[int]): 데이터셋의 0/1 레이블 리스트
        batch_size (int): 짝수로 설정 (예: 16)
    """
    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0, "batch_size는 짝수여야 합니다"
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.labels = labels

        # 클래스별 인덱스 분리
        self.pos_indices = [i for i, l in enumerate(labels) if l == 1.0]
        self.neg_indices = [i for i, l in enumerate(labels) if l == 0.0]
        # 한 에폭 당 생성할 배치 수
        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            # positive samples
            if self.pos_indices:
                if len(self.pos_indices) >= self.half:
                    pos_batch = random.sample(self.pos_indices, self.half)
                else:
                    pos_batch = random.choices(self.pos_indices, k=self.half)
            else:
                pos_batch = random.choices(self.neg_indices, k=self.half)
            # negative samples
            if self.neg_indices:
                if len(self.neg_indices) >= self.half:
                    neg_batch = random.sample(self.neg_indices, self.half)
                else:
                    neg_batch = random.choices(self.neg_indices, k=self.half)
            else:
                neg_batch = random.choices(self.pos_indices, k=self.half)

            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

# ────────────────────────────────────────────────────────────────────────────
# 1) 하이퍼파라미터 설정
ANN_PATH = '/home/vip1/Desktop/SSDC/neubility/gd/BTS_graduate_project/Training/label/E01_001.json'
FPS      = 30
SEQ_LEN  = 8
BATCH    = 64   # 짝수
EPOCHS   = 5
LR       = 1e-3
USE_CUDA = True

device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')

# 2) Dataset 생성
dataset = SingleVideoAnomalyDataset(
    ann_path=ANN_PATH,
    seq_len=SEQ_LEN,
    fps=FPS
)
print(f"총 시퀀스: {len(dataset)}")
labels = [label for (_, _, label) in dataset.samples]
print(f"정상 샘플 수: {int(sum(1 for l in labels if l==0.0))}")
print(f"이상 샘플 수: {int(sum(1 for l in labels if l==1.0))}")

# 3) BalancedBatchSampler 생성
labels = [label for (_, _, label) in dataset.samples]
sampler = BalancedBatchSampler(labels, batch_size=BATCH)

# 4) DataLoader
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=0
)

# 5) 모델, 옵티마이저, 손실
model     = GRUAnomalyDetector().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# 6) 학습 루프
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch}/{EPOCHS}', unit='batch')
    for seqs, labels in pbar:
        seqs   = seqs.to(device)
        labels = labels.to(device)

        probs = model(seqs)
        loss  = criterion(probs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"▶️ Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    # 7) 모델 저장
    torch.save(model.state_dict(), 'anomaly_single_balanced.pth')
    print('Saved 👉 anomaly_single_balanced.pth')


