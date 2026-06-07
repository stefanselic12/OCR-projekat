"""
Analiza izbalansiranosti karaktera u datasetu tablica.

Pokretanje:
    python analiza_karaktera.py

Ova skripta čita sve .txt fajlove iz data/ podfoldera, broji karaktere
i prikazuje frekvencijsku distribuciju.
"""

import os
from collections import Counter
from pathlib import Path


def analiziraj():
    data_root = Path(__file__).parent / "data"
    
    sve_label = []  # (podfolder, labela)
    
    for podfolder in sorted(os.listdir(data_root)):
        folder = data_root / podfolder
        if not folder.is_dir():
            continue
        
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".txt"):
                continue
            
            with open(folder / f, "r", encoding="utf-8") as fp:
                labela = fp.read().strip()
            
            # Ukloni crtice i razmake — samo čisti karakteri
            cisti = labela.replace("-", "").replace(" ", "")
            sve_label.append((podfolder, cisti))
    
    # ── 1. Ukupna frekvencija ─────────────────────────────────────
    counter = Counter()
    for _, cisti in sve_label:
        counter.update(cisti.upper())
    
    total = sum(counter.values())
    unique = len(counter)
    
    print("=" * 70)
    print(f"  ANALIZA KARAKTERA U DATASETU")
    print(f"  Ukupno uzoraka: {len(sve_label)}")
    print(f"  Ukupno karaktera: {total}")
    print(f"  Unikatnih karaktera: {unique}")
    print("=" * 70)
    
    # ── 2. Cifre ──────────────────────────────────────────────────
    print("\n  ── CIFRE ──")
    print(f"  {'Karakter':>8} {'Broj':>6} {'Procenat':>8}  {'Grafikon':<30}")
    print(f"  {'─'*8} {'─'*6} {'─'*8}  {'─'*30}")
    
    for c in "0123456789":
        n = counter.get(c, 0)
        pct = n / total * 100
        bar = "█" * int(pct * 2) + "░" * (30 - int(pct * 2))
        print(f"  {c!r:>8} {n:>6} {pct:>7.2f}%  {bar}")
    
    # ── 3. Slova ──────────────────────────────────────────────────
    slova = {k: v for k, v in sorted(counter.items()) if k.isalpha()}
    
    print("\n  ── SLOVA ──")
    print(f"  {'Karakter':>8} {'Broj':>6} {'Procenat':>8}  {'Grafikon':<30}")
    print(f"  {'─'*8} {'─'*6} {'─'*8}  {'─'*30}")
    
    for c, n in slova.items():
        pct = n / total * 100
        bar = "█" * int(pct * 2) + "░" * (30 - int(pct * 2))
        print(f"  {c!r:>8} {n:>6} {pct:>7.2f}%  {bar}")
    
    # ── 4. Po setovima ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STATISTIKA PO SETOVIMA")
    print("=" * 70)
    
    for set_name in ["train", "val", "test"]:
        set_counter = Counter()
        set_uzoraka = 0
        for podfolder, cisti in sve_label:
            if podfolder == set_name:
                set_counter.update(cisti.upper())
                set_uzoraka += 1
        
        if set_uzoraka == 0:
            continue
        
        n_char = sum(set_counter.values())
        n_uniq = len(set_counter)
        najcesci = set_counter.most_common(5)
        najredi = set_counter.most_common()[:-6:-1]
        
        print(f"\n  [{set_name}]")
        print(f"    Uzoraka: {set_uzoraka}")
        print(f"    Karaktera: {n_char} (unikatnih: {n_uniq})")
        print(f"    Najčešći: {', '.join(f'{c}({n})' for c, n in najcesci)}")
        print(f"    Najređi:  {', '.join(f'{c}({n})' for c, n in najredi)}")
    
    # ── 5. Procena balansa ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PROCENA IZBALANSIRANOSTI")
    print("=" * 70)
    
    counts = list(counter.values())
    max_c = max(counts)
    min_c = min(counts)
    avg = total / unique
    std = (sum((c - avg) ** 2 for c in counts) / unique) ** 0.5
    
    print(f"\n  Maks: {max_c}")
    print(f"  Min:  {min_c}")
    print(f"  Avg:  {avg:.1f}")
    print(f"  Std:  {std:.1f}")
    print(f"  Odnos max/min: {max_c/min_c:.1f}x")
    
    if max_c / min_c > 5:
        print("\n  ⚠️  VEOMA NEIZBALANSIRANO! Razmotriti:")
        print("      - Weightovanje gubitka po klasama")
        print("      - Oversampling retkih karaktera")
        print("      - Balansiranje kroz augmentaciju")
    elif max_c / min_c > 3:
        print("\n  ⚠️  Umereno neizbalansirano.")
    else:
        print("\n  ✅ Dataset je relativno izbalansiran.")


if __name__ == "__main__":
    analiziraj()
