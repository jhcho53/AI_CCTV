# model.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# 시퀀스 길이, 이미지 크기
SEQ_LEN    = 8
H, W       = 256, 256   # 변경: 256×256 입력
FEAT_DIM   = 64
HIDDEN_DIM = 128

class MobileNetEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, pretrained=True):
        super().__init__()
        # 1) 분류 헤드(fc) 전까지 백본 로드
        backbone = mobilenet_v2(pretrained=pretrained)
        self.features = backbone.features   # 출력 채널 1280

        # 2) Adaptive Pool + FC로 FEAT_DIM 차원으로 투영
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(1280, feat_dim)

        # 3) 백본 동결
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, C, H, W) where H=W=256
        f = self.features(x)                   # -> (B,1280,H/32,W/32)
        f = self.pool(f).view(f.size(0), -1)   # -> (B,1280)
        return self.fc(f)                      # -> (B, FEAT_DIM)

class GRUAnomalyDetector(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM, pretrained_backbone=True):
        super().__init__()
        # MobileNetEncoder 백본 (동결된 상태)
        self.encoder = MobileNetEncoder(feat_dim=feat_dim, pretrained=pretrained_backbone)
        self.gru     = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 이상 확률 [0,1]
        )

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W), H=W=256
        B, T, C, H, W = x_seq.size()
        x = x_seq.view(B * T, C, H, W)         # (B*T, C, 256, 256)
        feats = self.encoder(x)                # (B*T, FEAT_DIM)
        feats = feats.view(B, T, -1)           # (B, T, FEAT_DIM)
        out, _ = self.gru(feats)               # (B, T, HIDDEN_DIM)
        last = out[:, -1, :]                   # (B, HIDDEN_DIM)
        prob = self.head(last).squeeze(1)      # (B,)
        return prob
