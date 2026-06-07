"""
src/plot_results.py — Grafikoni treninga

Pokretanje:
    python -m src.plot_results

Generiše 4 grafikona:
  1. Loss kriva (train + val)
  2. CER kriva (train + val)
  3. Word Accuracy kriva
  4. Learning rate kroz epohe
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

LOGS_DIR = "logs"
PLOTS_DIR = "plots"


def ucitaj_istoriju() -> dict:
    """Učitava najnoviji log fajl."""
    logs_path = Path(LOGS_DIR)
    if not logs_path.exists():
        print(f"GREŠKA: {LOGS_DIR}/ ne postoji. Pokrenite trening prvo!")
        return None

    # Pronađi najnoviji JSON fajl
    json_fajlovi = sorted(logs_path.glob("trening_istorija_*.json"))
    if not json_fajlovi:
        # Pokušaj sa starim imenom
        stari = logs_path / "trening_istorija.json"
        if stari.exists():
            json_fajlovi = [stari]
        else:
            print(f"GREŠKA: Nema log fajlova u {LOGS_DIR}/")
            return None

    najnoviji = json_fajlovi[-1]
    print(f"Učitavam: {najnoviji}")
    with open(najnoviji, "r") as f:
        return json.load(f)


def plot_loss(ax, istorija):
    """Loss kriva."""
    epohe = range(1, len(istorija["train_loss"]) + 1)
    ax.plot(epohe, istorija["train_loss"], label="Train Loss", color="#2196F3", lw=2)
    ax.plot(epohe, istorija["val_loss"],   label="Val Loss",   color="#F44336", lw=2, linestyle="--")

    # Označi minimum val loss-a
    min_idx = int(np.argmin(istorija["val_loss"]))
    min_val = istorija["val_loss"][min_idx]
    ax.axvline(x=min_idx + 1, color="gray", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Best\nEp.{min_idx+1}\n{min_val:.3f}",
        xy=(min_idx + 1, min_val),
        xytext=(min_idx + 3, min_val + 0.3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="gray"
    )

    ax.set_title("CTC Loss kroz epohe", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_cer(ax, istorija):
    """CER kriva."""
    epohe = range(1, len(istorija["train_cer"]) + 1)
    ax.plot(epohe, istorija["train_cer"], label="Train CER", color="#4CAF50", lw=2)
    ax.plot(epohe, istorija["val_cer"],   label="Val CER",   color="#FF9800", lw=2, linestyle="--")

    ax.axhline(y=0.5, color="red",   linestyle=":", alpha=0.5, label="CER=0.5")
    ax.axhline(y=0.2, color="green", linestyle=":", alpha=0.5, label="CER=0.2")
    ax.axhline(y=0.0, color="blue",  linestyle=":", alpha=0.3)

    ax.set_title("Character Error Rate (CER)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("CER (niži = bolji)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)


def plot_word_acc(ax, istorija):
    """Word Accuracy kriva."""
    epohe = range(1, len(istorija["train_word_acc"]) + 1)
    train_pct = [v * 100 for v in istorija["train_word_acc"]]
    val_pct   = [v * 100 for v in istorija["val_word_acc"]]

    ax.plot(epohe, train_pct, label="Train Word Acc", color="#9C27B0", lw=2)
    ax.plot(epohe, val_pct,   label="Val Word Acc",   color="#E91E63", lw=2, linestyle="--")

    max_val = max(val_pct)
    max_idx = val_pct.index(max_val)
    ax.annotate(
        f"Max: {max_val:.1f}%",
        xy=(max_idx + 1, max_val),
        xytext=(max_idx + 3, max_val - 10),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="#E91E63", fontweight="bold"
    )

    ax.set_title("Word Accuracy (% tačnih tablica)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Tačnost (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)


def plot_lr(ax, istorija):
    """Learning Rate kriva."""
    epohe = range(1, len(istorija["lr"]) + 1)
    ax.plot(epohe, istorija["lr"], color="#607D8B", lw=2)
    ax.fill_between(epohe, 0, istorija["lr"], alpha=0.15, color="#607D8B")

    lrovi = istorija["lr"]
    for i in range(1, len(lrovi)):
        if lrovi[i] < lrovi[i - 1] - 1e-9:
            ax.axvline(x=i + 1, color="red", linestyle="--", alpha=0.6)
            ax.text(i + 1, lrovi[i], f"  ÷{int(round(lrovi[i-1]/lrovi[i]))}",
                    color="red", fontsize=8, va="center")

    ax.set_title("Learning Rate kroz epohe", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    istorija = ucitaj_istoriju()
    if istorija is None:
        return

    n_epoha = len(istorija["train_loss"])
    print(f"Metrike za {n_epoha} epoha.")

    # 2×2 grafikon
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"CRNN trening — OCR tablica  |  {n_epoha} epoha",
        fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_loss(ax1, istorija)
    plot_cer(ax2, istorija)
    plot_word_acc(ax3, istorija)
    plot_lr(ax4, istorija)

    plt.savefig(os.path.join(PLOTS_DIR, "trening_grafikon.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Sačuvano: {PLOTS_DIR}/trening_grafikon.png")

    # Samo loss detaljno
    fig2, ax = plt.subplots(figsize=(10, 5))
    plot_loss(ax, istorija)
    ax.set_title("CTC Loss — Train vs Val", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_detalj.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Sačuvano: {PLOTS_DIR}/loss_detalj.png")

    plt.show()
    print("\nGrafikoni sačuvani u 'plots/' folderu.")


if __name__ == "__main__":
    main()
