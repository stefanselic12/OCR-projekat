"""
src/dataset.py — Dataset i rečnik karaktera

Rečnik:
  - index 0 = <BLANK> (rezervisan za CTC)
  - index 1 = <PAD> (za paddovanje labela)
  - ostali su stvarni karakteri

Labela se iz .txt fajla učitava u originalnom formatu
(npr. "IN 054-DN", "J41-E-514", "BT 81342"),
a zatim se izbacuju razmaci i crtice pre nego što se enkoduje.
Time model uči da prepoznaje samo slova i cifre.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Dict


# ── Rečnik karaktera ────────────────────────────────────────────────────────
# Dodaj ovo na vrh src/dataset.py, posle import-a

class ResizeWithPadding:
    """Resize sliku ali sačuvaj aspect ratio, dodaj padding"""
    def __init__(self, target_h=32, target_w=128):
        self.target_h = target_h
        self.target_w = target_w
    
    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = T.Resize((new_h, new_w))(img)
        
        pad_left = (self.target_w - new_w) // 2
        pad_right = self.target_w - new_w - pad_left
        pad_top = (self.target_h - new_h) // 2
        pad_bottom = self.target_h - new_h - pad_top
        
        return T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)(img)

def ispravi_krivinu(img: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """
    Automatski ispravlja nagnutost slike tablice koristeći OpenCV.
    Detektuje ugao nagiba preko ivica (Canny + minAreaRect) i rotira sliku.
    Ignoruje nagibu ispod 0.5° i iznad max_angle da ne bi preterano korigovao.
    """
    try:
        gray = np.array(img.convert("L"))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 20:
            return img

        angle = cv2.minAreaRect(coords)[2]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return img

        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img_np = np.array(img.convert("RGB"))
        rotated = cv2.warpAffine(img_np, M, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception:
        return img


def napravi_recnik(folderi: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Prolazi kroz sve .txt fajlove i gradi rečnik od karaktera koji se javljaju.
    """
    svi_karakteri = set()
    for folder in folderi:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".txt"):
                path = os.path.join(folder, f)
                with open(path, "r", encoding="utf-8") as fp:
                    tekst = fp.read().strip()
                # Uzimamo samo alfanumeričke karaktere
                for c in tekst:
                    if c.isalnum():
                        svi_karakteri.add(c.upper())

    karakteri = ["<BLANK>", "<PAD>"] + sorted(svi_karakteri)
    char_to_idx = {c: i for i, c in enumerate(karakteri)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


# ── Transformacije ──────────────────────────────────────────────────────────

def trening_transform(img_h: int = 32, img_w: int = 128) -> T.Compose:
    """
    Augmentacije za trening.
    """
    return T.Compose([
        T.Lambda(ispravi_krivinu),
        ResizeWithPadding(img_h, img_w),  # ← ZAMENI T.Resize sa ovim!
        T.Grayscale(num_output_channels=1),
        T.RandomRotation(degrees=3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def val_transform(img_h: int = 32, img_w: int = 128) -> T.Compose:
    """
    Validacija i test — bez augmentacije.
    """
    return T.Compose([
        T.Lambda(ispravi_krivinu),
        ResizeWithPadding(img_h, img_w),  # ← ZAMENI T.Resize sa ovim!
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


# ── Dataset ─────────────────────────────────────────────────────────────────

class TabliceDataset(Dataset):
    """
    Dataset za CRNN trening.

    Očekuje strukturu:
        data/train/srb1.jpg + data/train/srb1.txt ("IN 054-DN")
    
    __getitem__ vraća:
        img     : (1, 32, 128) tensor
        encoded : (max_len,) tensor — indeksi karaktera (samo slova i cifre, bez separatora)
        length  : int — stvarna dužina labele
    """

    def __init__(
        self,
        folder: str,
        char_to_idx: Dict[str, int],
        max_len: int = 12,
        augment: bool = False,
        img_h: int = 32,
        img_w: int = 128,
    ):
        self.folder = folder
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.transform = trening_transform(img_h, img_w) if augment else val_transform(img_h, img_w)

        self.uzorci = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, f)
                base = f.rsplit(".", 1)[0]
                txt_path = os.path.join(folder, base + ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as fp:
                        labela = fp.read().strip()
                    # Uzimamo samo alfanumeričke karaktere (bez razmaka i crtica)
                    labela_cista = "".join(c for c in labela if c.isalnum()).upper()
                    self.uzorci.append((img_path, labela_cista))

    def __len__(self):
        return len(self.uzorci)

    def __getitem__(self, idx: int):
        img_path, labela = self.uzorci[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Enkoduj
        unk_idx = self.char_to_idx.get("<PAD>", 1)
        encoded = [self.char_to_idx.get(c, unk_idx) for c in labela]
        length = len(encoded)

        # Paduj do max_len
        pad_idx = self.char_to_idx["<PAD>"]
        encoded += [pad_idx] * (self.max_len - length)
        encoded = encoded[:self.max_len]

        return img, torch.tensor(encoded, dtype=torch.long), length

    def labele(self) -> List[str]:
        return [lab for _, lab in self.uzorci]


# ── Collate funkcija ────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Slaže batch u tenzore za CTC loss.
    
    Returns:
        imgs: (B, 1, H, W)
        labels_1d: 1D tenzor sa svim labelama spojenim (za CTC loss)
        target_lengths: dužine svake labele
        labels_padded: (B, max_len) padovane labele (za metrike)
    """
    imgs = torch.stack([item[0] for item in batch])
    
    # Pripremi za CTC (1D)
    all_labels = []
    target_lengths = []
    max_len = 0
    
    for item in batch:
        encoded = item[1]  # (max_len,)
        length = item[2]   # stvarna dužina
        all_labels.append(encoded[:length])
        target_lengths.append(length)
        max_len = max(max_len, length)
    
    labels_1d = torch.cat(all_labels)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    # Pripremi padovane labele za metrike
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        encoded = item[1]
        length = item[2]
        labels_padded[i, :length] = encoded[:length]
    
    return imgs, labels_1d, target_lengths, labels_padded


# ═════════════════════════════════════════════════════════════════════════════
#  STARE FUNKCIJE — ZADRŽANE ZBOG KOMPATIBILNOSTI (mogu se obrisati kasnije)
# ═════════════════════════════════════════════════════════════════════════════


