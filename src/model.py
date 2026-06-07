"""
src/model.py — CRNN arhitektura za OCR tablica
"""

import torch
import torch.nn as nn


class CNNBlok(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel),
        )

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    CRNN arhitektura koja automatski izračunava veličinu za RNN.
    """
    
    def __init__(self, num_chars: int, rnn_hidden: int = 256, rnn_layers: int = 2, 
                 img_h: int = 64, img_w: int = 256):
        super().__init__()
        
        self.img_h = img_h
        self.img_w = img_w
        self.rnn_hidden = rnn_hidden
        
        # CNN deo
        self.cnn = nn.Sequential(
            CNNBlok(1, 32, pool_kernel=(2, 2)),   # 64×256 → 32×128
            CNNBlok(32, 64, pool_kernel=(2, 2)),   # 32×128 → 16×64
            CNNBlok(64, 128, pool_kernel=(2, 1)),  # 16×64 → 8×64
            CNNBlok(128, 256, pool_kernel=(2, 1)), # 8×64 → 4×64 (DODAT SLOJ!)
        )
        
        # Provera dimenzija - izračunaj veličinu nakon CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            dummy_out = self.cnn(dummy)
            _, c, h, w = dummy_out.shape
            self.cnn_output_size = c * h
            self.time_steps = w
        
        print(f"  CNN izlaz: {c}×{h}×{w} → {self.cnn_output_size}×{self.time_steps}")
        
        # RNN deo
        self.rnn = nn.GRU(
            input_size=self.cnn_output_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if rnn_layers > 1 else 0.0,
        )
        
        # Izlazni sloj
        self.fc = nn.Linear(rnn_hidden * 2, num_chars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) — batch slika u grayscale
        returns: (B, T, num_chars) — logiti po vremenskom koraku
        """
        # CNN
        x = self.cnn(x)                              # (B, C, H, W)
        
        # Reshape za RNN: (B, W, C*H)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)                    # (B, W, C, H)
        x = x.reshape(B, W, C * H)                   # (B, W, C*H)
        
        # RNN
        x, _ = self.rnn(x)                           # (B, W, hidden*2)
        
        # FC
        x = self.fc(x)                               # (B, W, num_chars)
        return x


def broj_parametara(model: nn.Module) -> dict:
    """Vraća broj trenabilnih i ukupnih parametara modela."""
    ukupno = sum(p.numel() for p in model.parameters())
    trenabilni = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"ukupno": ukupno, "trenabilni": trenabilni}