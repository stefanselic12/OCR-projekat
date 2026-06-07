"""
src/metrics.py — Dekoderi i metrike za CRNN/CTC

Sadrži:
  - greedy_decode: brz i jednostavan CTC dekoder
  - beam_search_decode: precizniji ali sporiji CTC dekoder
  - izracunaj_metrike: CER + Word Accuracy
  - dekoduj_labele: konvertuje indekse u tekst

Greedy vs Beam Search:
  Greedy u svakom koraku bira karakter sa najvećom verovatnoćom.
  Beam Search drži K najboljih kandidata i bira najverovatniji na kraju.
  Beam Search je 10-50x sporiji ali daje bolje rezultate.
"""

import torch
import numpy as np
from typing import List, Dict

# ── Levenshtein distance ──────────────────────────────────────────────────

try:
    import Levenshtein
    def _lev_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:
    def _lev_distance(a: str, b: str) -> int:
        """Fallback implementacija Levenshtein distance."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i] + [0] * lb
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
            prev = curr
        return prev[lb]


# ── Greedy CTC dekoder ────────────────────────────────────────────────────

def greedy_decode(
    output: torch.Tensor,
    idx_to_char: Dict[int, str],
    blank_idx: int = 0,
) -> List[str]:
    """
    Greedy CTC dekodiranje.

    Za svaki vremenski korak biramo karakter sa najvećom verovatnoćom.
    Zatim primenjujemo CTC collapse:
      1. Izbacujemo <BLANK> karaktere (blank_idx)
      2. Spajamo uzastopne duplikate
    
    Primer:
      logiti:  [A, A, _, _, B, B, _, C]
      greed:   [A, A, B, B, C]
      collapse:[A, B, C]
      rezultat:'ABC'
    """
    _, max_indices = torch.max(output, dim=2)  # (B, T)
    
    dekodovano = []
    for i in range(max_indices.size(0)):
        seq = []
        prethodni = blank_idx
        for idx in max_indices[i]:
            idx = idx.item()
            if idx != blank_idx and idx != prethodni:
                c = idx_to_char.get(idx, "")
                if c not in ("<PAD>", "<UNK>"):
                    seq.append(c)
            prethodni = idx
        dekodovano.append("".join(seq))
    
    return dekodovano


# ── Beam Search CTC dekoder ───────────────────────────────────────────────

def beam_search_decode(
    output: torch.Tensor,
    idx_to_char: Dict[int, str],
    beam_width: int = 5,
    blank_idx: int = 0,
) -> List[str]:
    """
    Beam Search CTC dekodiranje.

    Umesto da biramo samo najbolji karakter u svakom koraku,
    držimo 'beam_width' najboljih kandidata do kraja sekvence.
    """
    probs = torch.softmax(output, dim=2).cpu().numpy()  # (B, T, C)
    B, T, C = probs.shape
    
    results = []
    for b in range(B):
        # Beam: lista (sekvenca_indeksa, log_verovatnoca)
        beam = [([], 0.0)]
        
        for t in range(T):
            candidates = []
            for seq, score in beam:
                for c in range(C):
                    log_prob = np.log(probs[b, t, c] + 1e-10)
                    new_score = score + log_prob
                    
                    if c == blank_idx:
                        candidates.append((seq.copy(), new_score))
                    else:
                        new_seq = seq.copy()
                        new_seq.append(c)
                        candidates.append((new_seq, new_score))
            
            # Sortiraj po verovatnoći opadajuće
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
        
        # Izaberi najbolju sekvencu
        best_seq = beam[0][0]
        
        # CTC collapse
        collapsed = []
        prev = blank_idx
        for idx in best_seq:
            if idx != blank_idx and idx != prev:
                collapsed.append(idx)
            prev = idx
        
        result = ''.join([idx_to_char.get(idx, '?') for idx in collapsed])
        results.append(result)
    
    return results


# ── Metrike ────────────────────────────────────────────────────────────────

def izracunaj_metrike(
    predikcije: List[str],
    tacne: List[str],
) -> Dict[str, float]:
    """
    Računa CER (Character Error Rate) i Word Accuracy.
    
    CER = Levenshtein(pred, tacno) / len(tacno)
    Word Acc = broj tačnih / ukupno
    
    Returns:
        dict: {"cer": float, "word_acc": float}
    """
    if not tacne:
        return {"cer": 1.0, "word_acc": 0.0}
    
    ukupno_karaktera = sum(len(t) for t in tacne)
    ukupno_gresaka = sum(_lev_distance(p, t) for p, t in zip(predikcije, tacne))
    tacnih_reci = sum(1 for p, t in zip(predikcije, tacne) if p == t)
    
    cer = ukupno_gresaka / max(ukupno_karaktera, 1)
    word_acc = tacnih_reci / max(len(tacne), 1)
    
    return {"cer": cer, "word_acc": word_acc}


def dekoduj_labele(
    labels: torch.Tensor,
    lengths: torch.Tensor,
    idx_to_char: Dict[int, str],
) -> List[str]:
    """
    Dekoduje batch labela (1D tenzor sa svim labelama spojenim) u stringove.
    
    Args:
        labels: 1D tenzor sa svim labelama spojenim
        lengths: dužine svake labele u batch-u
        idx_to_char: mapa indeks → karakter
    
    Returns:
        Lista dekodovanih stringova
    """
    dekodovano = []
    start = 0
    for i in range(lengths.size(0)):
        duzina = lengths[i].item()
        idxs = labels[start:start + duzina].tolist()
        # Konvertuj u string, preskoči PAD (1) i BLANK (0)
        tekst_chars = []
        for idx in idxs:
            if idx not in (0, 1):  # 0=BLANK, 1=PAD
                char = idx_to_char.get(idx, "")
                if char:
                    tekst_chars.append(char)
        tekst = "".join(tekst_chars)
        dekodovano.append(tekst)
        start += duzina
    return dekodovano


def dekoduj_padovane_labele(
    labels_padded: torch.Tensor,
    lengths: torch.Tensor,
    idx_to_char: Dict[int, str],
) -> List[str]:
    """
    Dekoduje batch padovanih labela (2D) u stringove.
    
    Args:
        labels_padded: (B, max_len) - padovani tenzor
        lengths: (B,) - stvarne dužine svake labele
        idx_to_char: mapa indeks → karakter
    
    Returns:
        Lista dekodovanih stringova
    """
    dekodovano = []
    for i in range(labels_padded.size(0)):
        duzina = lengths[i].item()
        # Uzmi samo relevantne indekse (do duzina)
        indeksi = labels_padded[i, :duzina].tolist()
        # Konvertuj u string, preskoči PAD (1) i BLANK (0)
        tekst_chars = []
        for idx in indeksi:
            if idx not in (0, 1):  # 0=BLANK, 1=PAD
                char = idx_to_char.get(idx, "")
                if char:
                    tekst_chars.append(char)
        dekodovano.append("".join(tekst_chars))
    return dekodovano