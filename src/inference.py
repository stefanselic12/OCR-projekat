"""
src/inference.py — Interaktivna demonstracija modela

Pokretanje:
    python -m src.inference
    python -m src.inference --beam 5   # Beam search umesto greedy

Prihvata putanju do slike i ispisuje prepoznati tekst.
Može se koristiti i za batch obradu:
    python -m src.inference --folder data/test
"""

import os
import sys
import json
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CRNN
from src.dataset import val_transform
from src.metrics import greedy_decode, beam_search_decode


CFG = {
    "models_dir": "models",
    "rnn_hidden": 256,
    "rnn_layers": 2,
    "img_h": 64,
    "img_w": 256,
}


def ucitaj_model(device=None):
    """Učitava model i rečnik. Vraća (model, idx_to_char, device)."""
    recnik_path = os.path.join(CFG["models_dir"], "recnik.json")
    model_path = os.path.join(CFG["models_dir"], "crnn_best.pth")

    if not os.path.exists(recnik_path) or not os.path.exists(model_path):
        print(f"GREŠKA: Pokreni trening prvo (python -m src.train)")
        return None, None, None

    with open(recnik_path, "r", encoding="utf-8") as f:
        recnik = json.load(f)

    idx_to_char = {int(k): v for k, v in recnik["idx_to_char"].items()}
    num_chars = len(recnik["char_to_idx"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(num_chars, CFG["rnn_hidden"], CFG["rnn_layers"],
                 img_h=CFG["img_h"], img_w=CFG["img_w"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model, idx_to_char, device


def predvidi(model, idx_to_char, device, img_path: str,
             use_beam: bool = False, beam_width: int = 5) -> str:
    """Prepoznaje tekst sa jedne slike tablice."""
    transform = val_transform(CFG["img_h"], CFG["img_w"])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        if use_beam:
            result = beam_search_decode(output, idx_to_char, beam_width=beam_width)
        else:
            result = greedy_decode(output, idx_to_char)

    return result[0]


def main():
    parser = argparse.ArgumentParser(description="CRNN OCR inferencija")
    parser.add_argument("--image", "-i", type=str, help="Putanja do slike")
    parser.add_argument("--folder", "-f", type=str, help="Folder sa slikama za batch obradu")
    parser.add_argument("--beam", "-b", type=int, default=0,
                        help="Koristi Beam Search sa zadatom širinom (0 = greedy)")
    args = parser.parse_args()

    model, idx_to_char, device = ucitaj_model()
    if model is None:
        return

    use_beam = args.beam > 0
    beam_width = args.beam or 5

    dekoder_naziv = f"Beam Search (width={beam_width})" if use_beam else "Greedy"
    print(f"\nDekoder: {dekoder_naziv}")
    print(f"Uređaj:  {device}\n")

    if args.image:
        # Pojedinačna slika
        if not os.path.exists(args.image):
            print(f"GREŠKA: Fajl ne postoji: {args.image}")
            return
        tekst = predvidi(model, idx_to_char, device, args.image, use_beam, beam_width)
        print(f"Slika:     {args.image}")
        print(f"Tablica:   '{tekst}'")

    elif args.folder:
        # Batch obrada
        if not os.path.isdir(args.folder):
            print(f"GREŠKA: Folder ne postoji: {args.folder}")
            return

        slike = [f for f in os.listdir(args.folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"Pronađeno {len(slike)} slika u {args.folder}:\n")

        for f in sorted(slike):
            path = os.path.join(args.folder, f)
            tekst = predvidi(model, idx_to_char, device, path, use_beam, beam_width)
            # Pročitaj tačnu labelu ako postoji
            txt_path = os.path.join(args.folder, f.rsplit(".", 1)[0] + ".txt")
            tacno = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as fp:
                    tacno = fp.read().strip()
                ok = "✅" if tekst == "".join(c for c in tacno if c.isalnum()).upper() else "❌"
                print(f"  {ok} {f}: '{tekst}'  (tačno: '{tacno}')")
            else:
                print(f"  {f}: '{tekst}'")

    else:
        # Interaktivni režim
        print("Unesi putanju do slike tablice (ili 'exit'):")
        print("Primer: data/test/eu1.jpg\n")

        while True:
            unos = input("> ").strip()
            if unos.lower() in ("exit", "quit", "q"):
                break
            if not os.path.exists(unos):
                print(f"  Fajl ne postoji: {unos}")
                continue

            tekst = predvidi(model, idx_to_char, device, unos, use_beam, beam_width)
            print(f"  '{tekst}'\n")


if __name__ == "__main__":
    main()
