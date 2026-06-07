"""
Balansiranje karaktera u datasetu.

Strategija:
  - Fokusiramo se SAMO na retka slova (ne cifre)
  - Za svako retko slovo generišemo sintetičke tablice
  - Samo slova se balansiraju, cifre ostaju kakve jesu

Pokretanje:
    python src/balansiraj_karaktere.py
"""

import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Konfiguracija ──────────────────────────────────────────────────────────
CILJNA_SLOVA = 50  # Ciljani broj pojavljivanja po slovu

# Tipovi tablica i njihovi obrasci
SRB_PATTERN = re.compile(r'^([A-ZČĆŽŠĐ]{2}) (\d{3})-([A-ZČĆŽŠĐ]{2})$')
BIH_PATTERN = re.compile(r'^([A-ZČĆŽŠĐ]?\d+)-([A-ZČĆŽŠĐ])-(\d+)$')
EU_PATTERN = re.compile(r'^([A-ZČĆŽŠĐ]{2}) (.+)$')

# Srpska latinica + standardna latinica
SVA_SLOVA = "ABCDEFGHIJKLMNOPRSTUVZČĆŽŠĐWYQ"


FONT_PUTANJE = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def _ucitaj_font(velicina: int) -> ImageFont.FreeTypeFont:
    for putanja in FONT_PUTANJE:
        try:
            return ImageFont.truetype(putanja, velicina)
        except OSError:
            continue
    return ImageFont.load_default()


def generisi_sliku_tablice(labela: str) -> Image.Image:
    """
    Generiše sintetičku sliku tablice sa zadatim tekstom koristeći PIL.
    Svaki poziv vraća drugačiju varijantu (boja, veličina fonta, šum)
    kako bi sintetički podaci bili što raznovrsniji.
    """
    sirina, visina = 400, 120

    boja_pozadine = random.choice([
        (255, 255, 255),
        (255, 255, 180),
        (245, 245, 245),
    ])
    img = Image.new("RGB", (sirina, visina), boja_pozadine)
    draw = ImageDraw.Draw(img)

    # Okvir tablice
    draw.rectangle([3, 3, sirina - 4, visina - 4], outline=(30, 30, 30), width=3)

    vel_fonta = random.randint(54, 68)
    font = _ucitaj_font(vel_fonta)

    bbox = draw.textbbox((0, 0), labela, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (sirina - tw) // 2 - bbox[0]
    y = (visina - th) // 2 - bbox[1]

    draw.text((x, y), labela, fill=(10, 10, 10), font=font)

    # Blagi šum
    arr = np.array(img, dtype=np.int16)
    arr += np.random.randint(-12, 13, arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def analiziraj_karaktere(data_root: Path) -> Counter:
    """Analizira frekvenciju karaktera u train/val/test."""
    counter = Counter()
    
    for podfolder in ["train", "val", "test"]:
        folder = data_root / podfolder
        if not folder.exists():
            continue
        for txt in folder.glob("*.txt"):
            labela = txt.read_text(encoding="utf-8").strip()
            cisti = "".join(c for c in labela if c.isalnum()).upper()
            counter.update(cisti)
    
    return counter


def generisi_srb_sa_slovom(slovo: str, gradovi: list, serije: list) -> str:
    """Generiše SRB: 'XX YYY-ZZ' sa retkim slovom u gradu ili seriji."""
    grad = random.choice(gradovi)
    serija = random.choice(serije)
    broj = f"{random.randint(0, 999):03d}"
    
    if random.random() < 0.5:
        # Stavi u grad (zameni jedno od 2 slova)
        poz = random.randint(0, 1)
        lst = list(grad)
        lst[poz] = slovo
        grad = "".join(lst)
    else:
        # Stavi u seriju
        poz = random.randint(0, 1)
        lst = list(serija)
        lst[poz] = slovo
        serija = "".join(lst)
    
    return f"{grad} {broj}-{serija}"


def generisi_bih_sa_slovom(slovo: str) -> str:
    """Generiše BIH: 'XXX-Y-ZZZ' sa retkim slovom u sredini."""
    levi = f"{random.randint(1, 999):0>{random.choice([2,3])}}"
    desni = f"{random.randint(1, 999):0>{random.choice([2,3])}}"
    return f"{levi}-{slovo}-{desni}"


def generisi_eu_sa_slovom(slovo: str) -> str:
    """Generiše EU: 'XX YYYZZ' sa retkim slovom u prefiksu."""
    drugo_slovo = random.choice([s for s in SVA_SLOVA if s != slovo])
    prefiks = random.choice([slovo + drugo_slovo, drugo_slovo + slovo])
    
    ostatak = "".join(random.choices("0123456789", k=random.randint(4, 6)))
    
    return f"{prefiks} {ostatak}"


def main():
    base = Path(__file__).parent.parent
    data = base / "data"
    
    print("=" * 60)
    print("  BALANSIRANJE KARAKTERA (SAMO SLOVA)")
    print("=" * 60)

    # 0. Obrisi stare synth fajlove (imali su random slike — pogresne)
    train_folder = data / "train"
    stari = list(train_folder.glob("synth_*"))
    if stari:
        print(f"\n  Brisanje {len(stari)} starih synth fajlova...")
        for f in stari:
            f.unlink()

    # 1. Analiza
    counter = analiziraj_karaktere(data)
    
    if not counter:
        print("❌ Nema podataka. Pokreni prvo pripremi_podatke.py")
        return
    
    total_slova = sum(v for k, v in counter.items() if k.isalpha())
    total_cifre = sum(v for k, v in counter.items() if k.isdigit())
    
    print(f"\n  Trenutno:")
    print(f"  Ukupno slova: {total_slova}")
    print(f"  Ukupno cifara: {total_cifre}")
    
    # 2. Pronađi retka slova
    retka_slova = {}
    for c in sorted(counter.keys()):
        if not c.isalpha():
            continue
        trenutno = counter[c]
        if trenutno < CILJNA_SLOVA:
            retka_slova[c] = CILJNA_SLOVA - trenutno
    
    if not retka_slova:
        print("  ✅ Sva slova su izbalansirana!")
        return
    
    print(f"\n  Retka slova (cilj: {CILJNA_SLOVA}):")
    for c, fali in sorted(retka_slova.items(), key=lambda x: -x[1]):
        print(f"    {c!r}: {counter[c]} → {CILJNA_SLOVA} (fali {fali})")
    
    # 3. Prikupi postojeće obrasce
    gradovi = set()
    serije = set()
    
    for podfolder in ["train", "val", "test"]:
        folder = data / podfolder
        if not folder.exists():
            continue
        for txt in folder.glob("*.txt"):
            labela = txt.read_text(encoding="utf-8").strip()
            m = SRB_PATTERN.match(labela)
            if m:
                gradovi.add(m.group(1))
                serije.add(m.group(3))
    
    if not gradovi:
        gradovi = {"IN", "VA", "KV", "NS", "BG", "SA", "VS", "BC", "SI", "CA", 
                   "LO", "ZR", "SM", "KG", "BP", "SU", "SD", "PO", "SO", "PA",
                   "NI", "VR", "BB", "PI", "UE", "TO", "RU", "BT"}
        serije = {"DN", "OR", "UL", "LV", "RR", "XJ", "NU", "LP", "BD", "ZA",
                  "UE", "VG", "GC", "AV", "CK", "KB", "HJ", "VK", "PI", "SN"}
    
    gradovi = sorted(gradovi)
    serije = sorted(serije)
    
    print(f"\n  Postojeći gradovi: {len(gradovi)}")
    print(f"  Postojeće serije: {len(serije)}")
    ukupno_fali = sum(retka_slova.values())
    print(f"  Potrebno: {ukupno_fali} sintetičkih karaktera")
    
    # 4. Generiši sintetičke labele
    print(f"\n{'='*60}")
    print("  GENERISANJE")
    print(f"{'='*60}")
    
    ukupno_gen = 0
    
    for slovo, fali in sorted(retka_slova.items(), key=lambda x: -x[1]):
        gen = 0
        for i in range(fali * 3):  # 3x pokušaja da ne bi falilo
            if gen >= fali:
                break
            
            tip = random.choices(["SRB", "BIH", "EU"], weights=[0.4, 0.3, 0.3])[0]
            
            if tip == "SRB":
                labela = generisi_srb_sa_slovom(slovo, gradovi, serije)
            elif tip == "BIH":
                labela = generisi_bih_sa_slovom(slovo)
            else:
                labela = generisi_eu_sa_slovom(slovo)
            
            if slovo not in labela.upper():
                continue
            
            naziv = f"synth_{slovo}_{gen:03d}.txt"
            txt_path = data / "train" / naziv
            txt_path.write_text(labela, encoding="utf-8")

            img = generisi_sliku_tablice(labela)
            img_path = data / "train" / f"synth_{slovo}_{gen:03d}.jpeg"
            img.save(str(img_path), "JPEG", quality=92)
            
            gen += 1
        
        ukupno_gen += gen
        print(f"  {slovo!r}: generisano {gen}")
    
    print(f"\n  Ukupno: {ukupno_gen} sintetičkih tablica")
    
    # 5. Finalna analiza (samo slova)
    print(f"\n{'='*60}")
    print("  FINALNA ANALIZA (SLOVA)")
    print(f"{'='*60}")
    
    counter2 = analiziraj_karaktere(data)
    
    # Samo slova
    slova_pre = {k: v for k, v in counter.items() if k.isalpha()}
    slova_posle = {k: v for k, v in counter2.items() if k.isalpha()}
    
    print(f"\n  {'Slovo':>5} {'Pre':>5} {'Posle':>6} {'Status'}")
    print(f"  {'─'*5} {'─'*5} {'─'*6} {'─'*10}")
    
    sva_slova_set = sorted(set(list(slova_pre.keys()) + list(slova_posle.keys())))
    max_c = 0
    min_c = float('inf')
    
    for c in sva_slova_set:
        pre = slova_pre.get(c, 0)
        posle = slova_posle.get(c, 0)
        max_c = max(max_c, posle)
        min_c = min(min_c, posle)
        
        if pre < CILJNA_SLOVA and posle >= CILJNA_SLOVA:
            status = "✅"
        elif pre < CILJNA_SLOVA:
            status = f"⚠️ fali {CILJNA_SLOVA - posle}"
        else:
            status = "✅"
        print(f"  {c!r:>5} {pre:>5} {posle:>6} {status}")
    
    print(f"\n  Max slovo: {max_c}, Min slovo: {min_c}")
    print(f"  Odnos max/min medju slovima: {max_c/min_c:.1f}x")
    
    if max_c / min_c < 3:
        print("  ✅ Slova su izbalansirana!")
    else:
        print(f"  ⚠️  Slova nisu dovoljno izbalansirana ({max_c/min_c:.1f}x)")
    
    # Cifre neka ostanu kakve jesu
    cifre_posle = {k: v for k, v in counter2.items() if k.isdigit()}
    print(f"\n  Cifre (nisu dirane):")
    for c in sorted(cifre_posle.keys()):
        print(f"    {c!r}: {cifre_posle[c]}")
    
    print(f"\n{'='*60}")
    print("  GOTOVO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
