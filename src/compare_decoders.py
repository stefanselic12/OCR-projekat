"""
src/compare_decoders.py — Poređenje Greedy vs Beam Search dekodera

Pokretanje:
    python -m src.compare_decoders

Pokazuje:
  - CER i Word Accuracy za oba dekodera
  - Tačne primere i razlike
  - Preporuku koji koristiti
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CRNN
from src.dataset import TabliceDataset, napravi_recnik, collate_fn
from src.metrics import greedy_decode, beam_search_decode, izracunaj_metrike, dekoduj_labele


CFG = {
    "train_folder": "data/train",
    "val_folder": "data/val",
    "test_folder": "data/test",
    "models_dir": "models",
    "img_h": 64,
    "img_w": 256,
    "max_label_len": 12,
    "rnn_hidden": 256,
    "rnn_layers": 2,
    "batch_size": 1,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  POREDJENJE DEKODERA: Greedy vs Beam Search")
    print(f"  Uređaj: {device}")
    print("=" * 70)

    # Učitaj rečnik
    recnik_path = os.path.join(CFG["models_dir"], "recnik.json")
    if not os.path.exists(recnik_path):
        print("❌ Pokreni trening prvo!")
        return

    with open(recnik_path, "r", encoding="utf-8") as f:
        recnik = json.load(f)
    idx_to_char = {int(k): v for k, v in recnik["idx_to_char"].items()}
    num_chars = len(recnik["char_to_idx"])

    # Učitaj model
    model_path = os.path.join(CFG["models_dir"], "crnn_best.pth")
    model = CRNN(num_chars, CFG["rnn_hidden"], CFG["rnn_layers"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model: {model_path}\n")

    # Test na validacionom i test skupu
    for skup, folder in [("Validacija (SRB+BIH)", CFG["val_folder"]),
                          ("Test (EU)", CFG["test_folder"])]:
        print(f"\n{'─'*70}")
        print(f"  {skup}")
        print(f"{'─'*70}")

        ds = TabliceDataset(folder, recnik["char_to_idx"], CFG["max_label_len"], augment=False,
                            img_h=CFG["img_h"], img_w=CFG["img_w"])
        if len(ds) == 0:
            print(f"  Prazan folder: {folder}")
            continue

        loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate_fn)

        sve_tacne = []
        sve_greedy = []
        sve_beam5 = []
        sve_beam10 = []

        with torch.no_grad():
            for imgs, labels, lengths, _ in loader:
                imgs = imgs.to(device)
                output = model(imgs)

                tacno = dekoduj_labele(labels, lengths, idx_to_char)[0]
                sve_tacne.append(tacno)

                g = greedy_decode(output, idx_to_char)[0]
                sve_greedy.append(g)

                b5 = beam_search_decode(output, idx_to_char, beam_width=5)[0]
                sve_beam5.append(b5)

                b10 = beam_search_decode(output, idx_to_char, beam_width=10)[0]
                sve_beam10.append(b10)

        # Metrike
        met_greedy = izracunaj_metrike(sve_greedy, sve_tacne)
        met_beam5 = izracunaj_metrike(sve_beam5, sve_tacne)
        met_beam10 = izracunaj_metrike(sve_beam10, sve_tacne)

        # Tabela
        print(f"\n  {'Dekoder':<15} {'CER':>8} {'Word Acc':>12}")
        print(f"  {'─'*15} {'─'*8} {'─'*12}")
        print(f"  {'Greedy':<15} {met_greedy['cer']:>8.4f} {met_greedy['word_acc']*100:>10.1f}%")
        print(f"  {'Beam-5':<15} {met_beam5['cer']:>8.4f} {met_beam5['word_acc']*100:>10.1f}%")
        print(f"  {'Beam-10':<15} {met_beam10['cer']:>8.4f} {met_beam10['word_acc']*100:>10.1f}%")

        # Primeri gde se razlikuju
        print(f"\n  Primeri gde se Greedy i Beam-10 razlikuju:")
        for i, (g, b, t) in enumerate(zip(sve_greedy, sve_beam10, sve_tacne)):
            if g != t or b != t:
                g_ok = "✅" if g == t else "❌"
                b_ok = "✅" if b == t else "❌"
                print(f"    Tačno:  '{t}'")
                print(f"    {g_ok} Greedy: '{g}'")
                print(f"    {b_ok} Beam-10:'{b}'")
                print()

    # Preporuka
    print(f"\n{'='*70}")
    print("  ZAKLJUČAK")
    print(f"{'='*70}")
    print("""
  Greedy:
    + Brz (nekoliko ms po slici)
    - Manje precizan (ne razmatra alternativne puteve)
    - Najbolji za real-time aplikacije i trening

  Beam Search:
    + Precizniji (razmatra više kandidata)
    - Sporiji (10-50x od Greedy, zavisi od beam_width)
    + Što veći beam, to bolje (ali diminishing returns posle 5-10)

  Preporuka:
    - Tokom treninga: GREEDY (validacija na svakoj epohi)
    - Za konačnu evaluaciju: BEAM SEARCH width=10
    - Za produkciju: GREEDY (dovoljno dobar, mnogo brži)
    """)


if __name__ == "__main__":
    main()
