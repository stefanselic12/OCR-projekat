"""
src/train.py — Kompletan trening CRNN modela za OCR tablica
"""

import os
import sys
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Dodaj parent folder u path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CRNN, broj_parametara
from src.dataset import TabliceDataset, napravi_recnik, collate_fn
from src.metrics import greedy_decode, beam_search_decode, izracunaj_metrike, dekoduj_labele


# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURACIJA
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Centralna konfiguracija sa svim parametrima"""
    
    # Putanje
    TRAIN_FOLDER = "data/train"
    VAL_FOLDER = "data/val"
    TEST_FOLDER = "data/test"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    
    # Dimenzije slike - POVEĆANE za bolji kvalitet
    IMG_H = 64   # sa 32 na 64
    IMG_W = 256  # sa 128 na 256
    
    # Arhitektura modela
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    
    # Trening parametri
    BATCH_SIZE = 8  # smanjeno zbog većih slika
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 150
    EARLY_STOP_PATIENCE = 20
    SCHEDULER_PATIENCE = 8
    SCHEDULER_FACTOR = 0.5
    GRAD_CLIP = 5.0
    WEIGHT_DECAY = 1e-5
    
    # DataLoader
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Eksperimentalne opcije
    USE_BEAM_SEARCH = True
    BEAM_WIDTH = 10
    USE_MIXED_PRECISION = False
    USE_COSINE_SCHEDULER = False
    
    # Reproducibilnost
    SEED = 42
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not callable(v)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pripremi_direktorijume():
    for folder in [Config.MODELS_DIR, Config.LOGS_DIR, 
                   Config.TRAIN_FOLDER, Config.VAL_FOLDER, Config.TEST_FOLDER]:
        os.makedirs(folder, exist_ok=True)


def izracunaj_max_duzinu_labele(folder):
    """Izračunava maksimalnu dužinu labele u folderu"""
    max_len = 0
    folder_path = Path(folder)
    if folder_path.exists():
        for txt in folder_path.glob("*.txt"):
            with open(txt, 'r', encoding='utf-8') as f:
                labela = f.read().strip()
                max_len = max(max_len, len(labela))
    return max_len


# ─────────────────────────────────────────────────────────────────────────────
#  TRENING FUNKCIJE
# ─────────────────────────────────────────────────────────────────────────────

def jedna_epoha_trening(model, loader, optimizer, criterion, device, idx_to_char):
    """Jedna epoha treninga"""
    model.train()
    ukupan_loss = 0.0
    sve_predikcije = []
    sve_tacne = []
    
    pbar = tqdm(loader, desc="Trening", leave=False)
    
    for batch in pbar:
        # Podrška za oba formata (3 ili 4 elementa)
        if len(batch) == 4:
            imgs, labels_1d, target_lengths, _ = batch
        else:
            imgs, labels_1d, target_lengths = batch
        
        imgs = imgs.to(device, non_blocking=True)
        labels_1d = labels_1d.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        output = model(imgs)
        
        log_probs = output.log_softmax(2).permute(1, 0, 2)
        T = output.size(1)
        input_lengths = torch.full((output.size(0),), T, dtype=torch.long, device=device)
        
        loss = criterion(log_probs, labels_1d, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()
        
        ukupan_loss += loss.item()
        
        # Dekodiranje za metrike
        preds = greedy_decode(output.detach(), idx_to_char)
        sve_predikcije.extend(preds)
        
        # Dekodiranje labela
        trues = dekoduj_labele(labels_1d, target_lengths, idx_to_char)
        sve_tacne.extend(trues)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrike = izracunaj_metrike(sve_predikcije, sve_tacne)
    metrike["loss"] = ukupan_loss / len(loader)
    return metrike


def jedna_epoha_validacija(model, loader, criterion, device, idx_to_char):
    """Jedna epoha validacije"""
    model.eval()
    ukupan_loss = 0.0
    sve_predikcije = []
    sve_tacne = []
    primeri = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validacija", leave=False):
            # Podrška za oba formata
            if len(batch) == 4:
                imgs, labels_1d, target_lengths, _ = batch
            else:
                imgs, labels_1d, target_lengths = batch
            
            imgs = imgs.to(device, non_blocking=True)
            labels_1d = labels_1d.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            
            output = model(imgs)
            
            log_probs = output.log_softmax(2).permute(1, 0, 2)
            T = output.size(1)
            input_lengths = torch.full((output.size(0),), T, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, labels_1d, input_lengths, target_lengths)
            ukupan_loss += loss.item()
            
            # Dekodiranje
            if Config.USE_BEAM_SEARCH:
                preds = beam_search_decode(output, idx_to_char, beam_width=Config.BEAM_WIDTH)
            else:
                preds = greedy_decode(output, idx_to_char)
            
            trues = dekoduj_labele(labels_1d, target_lengths, idx_to_char)
            sve_predikcije.extend(preds)
            sve_tacne.extend(trues)
            
            # Sačuvaj primere
            if len(primeri) < 5:
                for p, t in zip(preds, trues):
                    if (p, t) not in primeri:
                        primeri.append((p, t))
                        if len(primeri) >= 5:
                            break
    
    metrike = izracunaj_metrike(sve_predikcije, sve_tacne)
    metrike["loss"] = ukupan_loss / len(loader)
    return metrike, primeri


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class TreningLogger:
    def __init__(self, models_dir, logs_dir):
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.istorija = {
            "train_loss": [], "val_loss": [],
            "train_cer": [], "val_cer": [],
            "train_word_acc": [], "val_word_acc": [],
            "lr": [], "epoch": []
        }
        self.najbolji_val_cer = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log(self, epoch, train_met, val_met, lr):
        self.istorija["epoch"].append(epoch)
        self.istorija["train_loss"].append(train_met["loss"])
        self.istorija["val_loss"].append(val_met["loss"])
        self.istorija["train_cer"].append(train_met["cer"])
        self.istorija["val_cer"].append(val_met["cer"])
        self.istorija["train_word_acc"].append(train_met["word_acc"])
        self.istorija["val_word_acc"].append(val_met["word_acc"])
        self.istorija["lr"].append(lr)
    
    def should_save(self, val_met):
        if val_met["cer"] < self.najbolji_val_cer:
            self.najbolji_val_cer = val_met["cer"]
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False
    
    def save_model(self, model, epoch, val_met, suffix="best"):
        model_path = os.path.join(self.models_dir, f"crnn_{suffix}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_cer': val_met["cer"],
            'val_word_acc': val_met["word_acc"],
        }, model_path)
        print(f"  → Sačuvan model (CER={val_met['cer']:.4f})")

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        """Čuva kompletan checkpoint za nastavak treninga."""
        path = os.path.join(self.models_dir, "crnn_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'najbolji_val_cer': self.najbolji_val_cer,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'istorija': self.istorija,
        }, path)

    @classmethod
    def load_checkpoint(cls, models_dir, logs_dir, model, optimizer, scheduler, device):
        """Učitava checkpoint i vraća epohu od koje treba nastaviti."""
        path = os.path.join(models_dir, "crnn_last.pth")
        if not os.path.exists(path):
            print("GREŠKA: crnn_last.pth nije pronađen.")
            return None, 1
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger = cls(models_dir, logs_dir)
        logger.najbolji_val_cer = ckpt['najbolji_val_cer']
        logger.best_epoch = ckpt['best_epoch']
        logger.patience_counter = ckpt['patience_counter']
        logger.istorija = ckpt['istorija']
        start_epoch = ckpt['epoch'] + 1
        print(f"  Nastavak od epohe {start_epoch} (best CER={logger.najbolji_val_cer:.4f})")
        return logger, start_epoch
    
    def save_history(self):
        log_path = os.path.join(self.logs_dir, f"trening_istorija_{self.timestamp}.json")
        with open(log_path, "w") as f:
            json.dump(self.istorija, f, indent=2)
        return log_path
    
    def should_stop(self, patience):
        return self.patience_counter >= patience


# ─────────────────────────────────────────────────────────────────────────────
#  GLAVNA FUNKCIJA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trening CRNN modela za OCR")
    parser.add_argument("--beam", action="store_true", help="Koristi Beam Search")
    parser.add_argument("--beam-width", type=int, default=10, help="Širina beam-a")
    parser.add_argument("--epochs", type=int, default=150, help="Broj epoha")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Nastavi trening od poslednjeg checkpointa")
    args = parser.parse_args()
    
    Config.USE_BEAM_SEARCH = args.beam
    Config.BEAM_WIDTH = args.beam_width
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    pripremi_direktorijume()
    set_seed(Config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  CRNN TRENING - OCR AUTOMOBILSKIH TABLICA")
    print(f"{'='*70}")
    print(f"  Uređaj:          {device}")
    print(f"  Dimenzije slike: {Config.IMG_H}×{Config.IMG_W}")
    print(f"  Dekoder:         {'Beam Search (w=' + str(Config.BEAM_WIDTH) + ')' if Config.USE_BEAM_SEARCH else 'Greedy'}")
    print(f"{'='*70}\n")
    
    # Izračunaj max dužinu labele
    print("📏 Provera dužina labela...")
    train_max = izracunaj_max_duzinu_labele(Config.TRAIN_FOLDER)
    val_max = izracunaj_max_duzinu_labele(Config.VAL_FOLDER)
    max_label_len = max(train_max, val_max) + 5
    print(f"  Maksimalna dužina labele: {max(train_max, val_max)}")
    print(f"  Postavljen max_label_len: {max_label_len}")
    Config.MAX_LABEL_LEN = max_label_len
    
    # Kreiranje rečnika
    print("\n📖 Kreiranje rečnika karaktera...")
    char_to_idx, idx_to_char = napravi_recnik([Config.TRAIN_FOLDER, Config.VAL_FOLDER])
    num_chars = len(char_to_idx)
    print(f"  Rečnik: {num_chars} klasa")
    
    # Sačuvaj rečnik
    recnik_path = os.path.join(Config.MODELS_DIR, "recnik.json")
    with open(recnik_path, "w", encoding="utf-8") as f:
        json.dump({
            "char_to_idx": char_to_idx, 
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()}
        }, f, ensure_ascii=False, indent=2)
    
    # Učitavanje podataka
    print("\n📂 Učitavanje podataka...")
    train_ds = TabliceDataset(Config.TRAIN_FOLDER, char_to_idx, 
                               Config.MAX_LABEL_LEN, augment=True,
                               img_h=Config.IMG_H, img_w=Config.IMG_W)
    val_ds = TabliceDataset(Config.VAL_FOLDER, char_to_idx, 
                             Config.MAX_LABEL_LEN, augment=False,
                             img_h=Config.IMG_H, img_w=Config.IMG_W)
    
    print(f"  Trening:    {len(train_ds)} slika")
    print(f"  Validacija: {len(val_ds)} slika")

    if len(train_ds) == 0:
        print("\nGREŠKA: Trening skup je prazan. Pokrenite: python -m src.pripremi_podatke")
        return
    if len(val_ds) == 0:
        print("\nGREŠKA: Validacioni skup je prazan. Pokrenite: python -m src.pripremi_podatke")
        return
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=Config.NUM_WORKERS)
    
    # Kreiranje modela
    print("\n🏗️  Kreiranje modela...")
    model = CRNN(num_chars, Config.RNN_HIDDEN, Config.RNN_LAYERS).to(device)
    params = broj_parametara(model)
    print(f"  Parametara: {params['trenabilni']:,}")
    
    # Loss i optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                   weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=Config.SCHEDULER_FACTOR,
                                   patience=Config.SCHEDULER_PATIENCE)
    
    if args.resume:
        print("\n🔄 Učitavanje checkpointa...")
        logger, start_epoch = TreningLogger.load_checkpoint(
            Config.MODELS_DIR, Config.LOGS_DIR, model, optimizer, scheduler, device)
        if logger is None:
            return
    else:
        logger = TreningLogger(Config.MODELS_DIR, Config.LOGS_DIR)
        start_epoch = 1

    # Trening
    print(f"\n{'='*70}")
    print(f"  🚀 POČETAK TRENINGA ({Config.NUM_EPOCHS} epoha, od epohe {start_epoch})")
    print(f"{'='*70}\n")

    for epoha in range(start_epoch, Config.NUM_EPOCHS + 1):
        train_met = jedna_epoha_trening(model, train_loader, optimizer, criterion,
                                         device, idx_to_char)
        val_met, primeri = jedna_epoha_validacija(model, val_loader, criterion,
                                                   device, idx_to_char)
        
        trenutni_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_met["loss"])
        logger.log(epoha, train_met, val_met, trenutni_lr)
        
        print(f"Epoha {epoha:3d}/{Config.NUM_EPOCHS} | "
              f"Loss: {train_met['loss']:.4f}/{val_met['loss']:.4f} | "
              f"CER: {train_met['cer']:.3f}/{val_met['cer']:.3f} | "
              f"WordAcc: {train_met['word_acc']*100:.1f}%/{val_met['word_acc']*100:.1f}% | "
              f"LR: {trenutni_lr:.2e}")
        
        print("  📝 Primeri:")
        for pred, tacno in primeri[:3]:
            status = "✅" if pred == tacno else "❌"
            print(f"    {status} Pred: '{pred:<20}' | Tačno: '{tacno:<20}'")
        
        if logger.should_save(val_met):
            logger.save_model(model, epoha, val_met, "best")
            logger.best_epoch = epoha

        logger.save_checkpoint(model, optimizer, scheduler, epoha)

        if logger.should_stop(Config.EARLY_STOP_PATIENCE):
            print(f"\n⏹️  Early stopping na epohi {epoha}")
            break
    
    logger.save_model(model, logger.best_epoch, 
                      {"cer": logger.najbolji_val_cer, "word_acc": 0}, "final")
    history_path = logger.save_history()
    
    print(f"\n{'='*70}")
    print("  ✅ TRENING ZAVRŠEN")
    print(f"{'='*70}")
    print(f"  Najbolja epoha:     {logger.best_epoch}")
    print(f"  Najbolji Val CER:   {logger.najbolji_val_cer:.4f}")
    print(f"  Istorija sačuvana:  {history_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()