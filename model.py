import torch
import torch.nn as nn

SEQ_LEN    = 8
H, W       = 128, 128
FEAT_DIM   = 64
HIDDEN_DIM = 128

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, feat_dim=FEAT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64,128, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * (H//8) * (W//8), feat_dim)

    def forward(self, x):
        f = self.conv(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)

class GRUAnomalyDetector(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = ConvEncoder()
        self.gru     = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        x = x_seq.view(B*T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        out, _ = self.gru(feats)
        last = out[:, -1, :]
        prob = self.head(last).squeeze(1)
        return prob
