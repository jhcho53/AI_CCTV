import os
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import MultiVideoAnomalyDataset
from model.model import GRUAnomalyDetector
from dataloader.dataloader import BalancedBatchSampler

def parse_args():
    p = argparse.ArgumentParser(description="Train GRU-based Video Anomaly Detector")
    p.add_argument('--label_dir', type=str, required=True,
                   help="JSON 파일들이 모여있는 디렉토리")
    p.add_argument('--seq_len', type=int, default=8, help="시퀀스 길이")
    p.add_argument('--batch_size', type=int, default=64, help="배치 크기 (짝수)")
    p.add_argument('--epochs', type=int, default=5, help="학습 에폭 수")
    p.add_argument('--lr', type=float, default=1e-3, help="학습률")
    p.add_argument('--default_roi', type=int, nargs=4,
                   default=[1150,300,1600,700],
                   help="JSON에 ROI 없을 때 사용할 기본 ROI: x1 y1 x2 y2")
    p.add_argument('--num_workers', type=int, default=4, help="DataLoader 워커 수")
    p.add_argument('--pin_memory', action='store_true', help="pin_memory 사용 여부")
    p.add_argument('--use_cuda', action='store_true', help="CUDA 사용 여부")
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Sampler
    dataset = MultiVideoAnomalyDataset(
        label_dir   = args.label_dir,
        seq_len     = args.seq_len,
        default_roi = tuple(args.default_roi),
        transform   = None
    )
    print(f"총 시퀀스 샘플: {len(dataset)}")

    labels = [label for (_, _, label, _) in dataset.samples]
    print(f"정상 샘플: {sum(l==0.0 for l in labels)}, 이상 샘플: {sum(l==1.0 for l in labels)}")

    sampler = BalancedBatchSampler(labels, batch_size=args.batch_size)
    loader  = DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers    = args.num_workers,
        pin_memory     = args.pin_memory
    )

    # Model, Optimizer, Criterion
    model     = GRUAnomalyDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for seqs, labs in pbar:
            seqs = seqs.to(device)
            labs = labs.to(device)

            probs = model(seqs)
            loss  = criterion(probs, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

        ckpt = f"anomaly_multi_balanced_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Saved {ckpt}")

if __name__ == '__main__':
    main()
