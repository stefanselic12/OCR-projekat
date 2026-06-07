"""
src/evaluate.py — Evaluacija modela na validacionom i test skupu

Pokretanje:
    python -m src.evaluate

Prikazuje Greedy i Beam Search rezultate uporedo.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CRNN
from src.dataset import TabliceDataset, collate_fn
from src.metrics import greedy_decode, beam_search_decode, izracunaj_metrike, dekoduj_labele, _lev_distance


CFG = {
    "val_folder": "data/val",
    "test_folder": "data/test",
    "models_dir": "models",
    "img_h": 64,
    "img_w": 256,
    "max_label_len": 12,
    "rnn_hidden": 256,
    "rnn_layers": 2,
}


def evaluiraj_skup(model, folder, idx_to_char, device, naziv):
    """Evaluira model na jednom skupu koristeći oba dekodera."""
    char_to_idx = {v: k for k, v in idx_to_char.items()}

    ds = TabliceDataset(folder, char_to_idx, CFG["max_label_len"], augment=False,
                        img_h=CFG["img_h"], img_w=CFG["img_w"])

    if len(ds) == 0:
        print(f"\n[{naziv}] Folder je prazan: {folder}")
        return None

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    sve_tacne = []
    sve_greedy = []
    sve_beam = []

    print(f"\n{'='*60}")
    print(f"  {naziv} — {len(ds)} slika")
    print(f"{'='*60}")

    with torch.no_grad():
        for imgs, labels, lengths, _ in loader:
            imgs = imgs.to(device)
            output = model(imgs)

            tacno = dekoduj_labele(labels, lengths, idx_to_char)[0]
            g = greedy_decode(output, idx_to_char)[0]
            b = beam_search_decode(output, idx_to_char, beam_width=10)[0]

            sve_tacne.append(tacno)
            sve_greedy.append(g)
            sve_beam.append(b)

            g_ok = "✅" if g == tacno else "❌"
            b_ok = "✅" if b == tacno else "❌"

            cer_g = _lev_distance(g, tacno) / max(len(tacno), 1)
            cer_b = _lev_distance(b, tacno) / max(len(tacno), 1)
            print(f"  Tačno:  '{tacno}'")
            print(f"  {g_ok} Greedy:'{g}' (CER: {cer_g:.3f})")
            print(f"  {b_ok} Beam-10:'{b}' (CER: {cer_b:.3f})")
            print()

    # Agregatne metrike
    met_greedy = izracunaj_metrike(sve_greedy, sve_tacne)
    met_beam = izracunaj_metrike(sve_beam, sve_tacne)

    print(f"\n  {'Dekoder':<15} {'CER':>8} {'Word Acc':>12}")
    print(f"  {'─'*15} {'─'*8} {'─'*12}")
    print(f"  {'Greedy':<15} {met_greedy['cer']:>8.4f} {met_greedy['word_acc']*100:>10.1f}%")
    print(f"  {'Beam-10':<15} {met_beam['cer']:>8.4f} {met_beam['word_acc']*100:>10.1f}%")

    return {"greedy": met_greedy, "beam": met_beam, "n": len(sve_tacne)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")

    recnik_path = os.path.join(CFG["models_dir"], "recnik.json")
    if not os.path.exists(recnik_path):
        print("GREŠKA: recnik.json nije pronađen. Pokrenite trening prvo!")
        return

    with open(recnik_path, "r", encoding="utf-8") as f:
        recnik = json.load(f)
    idx_to_char = {int(k): v for k, v in recnik["idx_to_char"].items()}
    num_chars = len(recnik["char_to_idx"])

    model_path = os.path.join(CFG["models_dir"], "crnn_best.pth")
    if not os.path.exists(model_path):
        print("GREŠKA: crnn_best.pth nije pronađen.")
        return

    model = CRNN(num_chars, CFG["rnn_hidden"], CFG["rnn_layers"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"Model učitan: {model_path}\n")

    val_rez = evaluiraj_skup(model, CFG["val_folder"], idx_to_char, device, "VALIDACIJA")
    test_rez = evaluiraj_skup(model, CFG["test_folder"], idx_to_char, device, "TEST")


if __name__ == "__main__":
    main()
