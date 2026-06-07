"""
Popravlja formate tablica u .txt fajlovima.

Pravila:
  - SRB: "IN??054DN" → "IN 054-DN"
    (sačuvati A-Z, 0-9, srpsku latinicu ČĆŽŠĐ;
     ćirilica i ostali šum se brišu;
     format: 2 slova + razmak + 3 cifre + crtica + 2 slova)
  - BIH: "J41E514" → "J41-E-514"
    (crtica pre i posle srednjeg slova)
  - EU:  "BT81342" → "BT 81342"
    (razmak posle prva dva slova)

Radi nad data/srb, data/bih, data/eu — NE menja slike!
"""

import re
from pathlib import Path

# Dozvoljeni karakteri: A-Z, 0-9, ČĆŽŠĐ (srpska latinica)
DOZVOLJENI = re.compile(r'[^A-Za-z0-9ČčĆćŽžŠšĐđ]')


def ocisti(tekst: str) -> str:
    """Očisti tekst — ostavi samo A-Z, 0-9 i srpsku latinicu (ČĆŽŠĐ)."""
    return DOZVOLJENI.sub('', tekst).upper()


def transformisi_srb(tekst: str) -> str:
    """
    SRB: "IN??054DN" → očisti → "IN054DN" → "IN 054-DN"
    
    Očisti, pa od rezultata uzmi:
      - prva 2 slova (grad)
      - 3 cifre (broj)
      - zadnja 2 slova (serija)
    """
    cist = ocisti(tekst)
    
    # Standard: 7 karaktera: 2 slova + 3 cifre + 2 slova
    match = re.match(r'^([A-ZČĆŽŠĐ]{2})(\d{3})([A-ZČĆŽŠĐ]{2})$', cist)
    if match:
        return f"{match.group(1)} {match.group(2)}-{match.group(3)}"
    
    # Više karaktera (garbage dodao slova/cifre) — uzmi prva 2 slova, prve 3 cifre, zadnja 2 slova
    match = re.match(r'^([A-ZČĆŽŠĐ]{2})(\d{3})\d*([A-ZČĆŽŠĐ]{2})$', cist)
    if match:
        return f"{match.group(1)} {match.group(2)}-{match.group(3)}"
    
    # Ako ima 6 karaktera — verovatno fali prvo slovo (oštećen fajl)
    match = re.match(r'^([A-ZČĆŽŠĐ]{1})(\d{3})([A-ZČĆŽŠĐ]{2})$', cist)
    if match:
        return f"?{match.group(1)} {match.group(2)}-{match.group(3)}"
    
    return cist


def transformisi_bih(tekst: str) -> str:
    """
    BIH: "J41E514" → "J41-E-514" ili "059J608" → "059-J-608"
    
    Crtica pre i posle srednjeg slova.
    """
    cist = ocisti(tekst)
    
    # Slovo + brojevi + slovo + brojevi: "J41E514"
    match = re.match(r'^([A-ZČĆŽŠĐ]\d+)([A-ZČĆŽŠĐ])(\d+)$', cist)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    
    # Brojevi + slovo + brojevi: "059J608"
    match = re.match(r'^(\d+)([A-ZČĆŽŠĐ])(\d+)$', cist)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    
    return cist


def transformisi_eu(tekst: str) -> str:
    """
    EU: "BT81342" → "BT 81342"
    
    Prva 2 slova + razmak + ostatak.
    """
    cist = ocisti(tekst)
    
    # "BT81342" → prva 2 slova + ostatak
    match = re.match(r'^([A-ZČĆŽŠĐ]{2})(.*)$', cist)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    
    return cist


def main():
    base = Path(__file__).parent.parent
    data = base / "data"
    
    print("=" * 60)
    print("  POPRAVKA FORMATA TABLICA")
    print("=" * 60)
    
    for tip, folder_name, func in [
        ("SRB", "srb", transformisi_srb),
        ("BIH", "bih", transformisi_bih),
        ("EU",  "eu",  transformisi_eu),
    ]:
        folder = data / folder_name
        if not folder.exists():
            print(f"\n⚠️  Nema foldera {folder_name}/")
            continue
        
        print(f"\n📁 {tip} ({folder_name}/):")
        promenjeno = 0
        total = 0
        
        for txt in sorted(folder.glob("*.txt")):
            total += 1
            original = txt.read_text(encoding="utf-8").strip()
            novi = func(original)
            
            if novi != original:
                txt.write_text(novi, encoding="utf-8")
                promenjeno += 1
                print(f"  {original:25} → {novi}")
        
        print(f"  Promenjeno: {promenjeno}/{total}")
    
    print(f"\n{'='*60}")
    print("  ZAVRŠENO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()