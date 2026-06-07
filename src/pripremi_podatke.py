"""
Priprema podataka za trening.

Koraci:
  1. Pokreće txt_fix_formats da popravi formate u data/srb, data/bih, data/eu
  2. Prikazuje statistiku karaktera
  3. Podeli u train/val/test (60/20/20)

Pokretanje:
    python src/pripremi_podatke.py
"""

import os
import sys
import random
import shutil
from collections import Counter
from pathlib import Path

# Dodaj da bi moglo da se pokrene iz bilo kog foldera
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.txt_fix_formats import main as fix_formate


def ocisti_karaktere(labela: str) -> str:
    """Vrati samo A-Z i 0-9 iz labele."""
    return "".join(c for c in labela.upper() if c.isalnum())


def analiziraj(data_root: Path):
    """Analizira karaktere u data/srb, data/bih, data/eu."""
    
    svi = []
    
    for folder_name in ["srb", "bih", "eu"]:
        folder = data_root / folder_name
        if not folder.exists():
            continue
        for txt in sorted(folder.glob("*.txt")):
            labela = txt.read_text(encoding="utf-8").strip()
            cisti = ocisti_karaktere(labela)
            svi.append((folder_name, txt.name, labela, cisti))
    
    counter = Counter()
    for _, _, _, cisti in svi:
        counter.update(cisti)
    
    total = sum(counter.values())
    unique = len(counter)
    
    print(f"\n{'='*60}")
    print(f"  ANALIZA KARAKTERA")
    print(f"{'='*60}")
    print(f"  Ukupno uzoraka:   {len(svi)}")
    print(f"  Ukupno karaktera: {total}")
    print(f"  Unikatnih:        {unique}")
    
    # Cifre
    print(f"\n  ── CIFRE ──")
    print(f"  {'Kar':>4} {'Broj':>6} {'Pct':>6}")
    for c in "0123456789":
        n = counter.get(c, 0)
        pct = n / total * 100 if total else 0
        print(f"  {c!r:>4} {n:>6} {pct:>5.1f}%")
    
    # Slova
    slova = sorted([(k, v) for k, v in counter.items() if k.isalpha()])
    print(f"\n  ── SLOVA ──")
    print(f"  {'Kar':>4} {'Broj':>6} {'Pct':>6}")
    for c, n in slova:
        pct = n / total * 100 if total else 0
        print(f"  {c!r:>4} {n:>6} {pct:>5.1f}%")
    
    # Balans
    if counter:
        counts = list(counter.values())
        max_c = max(counts)
        min_c = min(counts)
        print(f"\n  Balans: max/min = {max_c}/{min_c} = {max_c/min_c:.1f}x")
    
    return svi


def podeli(svi: list, data_root: Path):
    """Podeli sve uzorke u train/val/test (60/20/20)."""
    
    random.seed(42)
    random.shuffle(svi)
    
    n = len(svi)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    
    podela = {
        "train": svi[:n_train],
        "val":   svi[n_train:n_train + n_val],
        "test":  svi[n_train + n_val:],
    }
    
    # Kreiraj foldere
    for folder_name in ["train", "val", "test"]:
        (data_root / folder_name).mkdir(exist_ok=True)
    
    # Kopiraj
    for folder_name, uzorci in podela.items():
        dest = data_root / folder_name
        for tip, fname, _, _ in uzorci:
            # Ime bez ekstenzije
            base = fname.rsplit(".", 1)[0]
            
            # Pronađi sliku (bilo .jpg, .jpeg, .png)
            src_folder = data_root / tip
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = src_folder / f"{base}{ext}"
                if cand.exists():
                    img = cand
                    break
            
            if img:
                shutil.copy2(img, dest / img.name)
            
            # Kopiraj txt
            txt = src_folder / f"{base}.txt"
            if txt.exists():
                shutil.copy2(txt, dest / txt.name)
    
    # Prikaži statistiku
    print(f"\n{'='*60}")
    print(f"  P O D E L A")
    print(f"{'='*60}")
    for folder_name in ["train", "val", "test"]:
        folder = data_root / folder_name
        br_slika = len(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        br_txt = len(list(folder.glob("*.txt")))
        print(f"  {folder_name}/: {br_slika} slika, {br_txt} txt")
    
    return podela


def main():
    base = Path(__file__).parent.parent
    data = base / "data"
    
    print("=" * 60)
    print("  PRIPREMA PODATAKA")
    print("=" * 60)
    
    # 1. Popravi formate u data/srb, data/bih, data/eu
    print("\n📌 Korak 1: Popravka formata...")
    fix_formate()
    
    # 2. Analiziraj
    print("\n📌 Korak 2: Analiza karaktera...")
    svi = analiziraj(data)
    
    # 3. Podeli u train/val/test
    print("\n📌 Korak 3: Podela podataka...")
    podeli(svi, data)
    
    print(f"\n{'='*60}")
    print(f"  ✅ GOTOVO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()