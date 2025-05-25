import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoAnomalyIterableDataset
from model import GRUAnomalyDetector

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--label_dir', type=str, required=True,
                   help='…/Training/label/ 폴더 경로')
    p.add_argument('--fps',       type=int, default=8)
    p.add_argument('--seq',       type=int, default=8)
    p.add_argument('--bs',        type=int, default=16)
    p.add_argument('--epochs',    type=int, default=10)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--cuda',      action='store_true')
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    dataset = VideoAnomalyIterableDataset(
        label_dir=args.label_dir,
        seq_len=args.seq,
        fps=args.fps
    )
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )

    # Model, Optimizer, Loss
    model     = GRUAnomalyDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Training loop with tqdm
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for seqs, labels in pbar:
            seqs   = seqs.to(device)
            labels = labels.to(device)
            probs  = model(seqs)
            loss   = criterion(probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), 'anomaly_detector.pth')

if __name__ == '__main__':
    main()