# OCR za automobilske tablice — Projektni zadatak iz Inteligentnih sistema

## 1. Problem i cilj projekta

Cilj projekta je automatsko prepoznavanje teksta sa slika automobilskih tablica
iz tri različita regiona: Srbije (SRB), Bosne i Hercegovine (BIH) i Evropske unije (EU).
Zadatak spada u oblast optičkog prepoznavanja karaktera (OCR).

Problem je izazovan jer se formati tablica razlikuju:
- Srpske tablice sadrže latinične i ćirilične oznake,
- BIH tablice imaju karakterističan numeričko‑slovni format,
- EU tablice imaju drugačiji raspored i ne sadrže ćirilicu.

## 2. Podaci (dataset)

### 2.1 Prikupljanje i struktura

Skup podataka je prikupljen samostalno, što odgovara zahtevima **Nivoa 3** projekta.

- Originalni skup: **100 slika** (50 iz Srbije + 50 iz Bosne i Hercegovine) i **41 slika** (EU tablice).
- Svaka slika ima pripadajući `.txt` fajl sa tačnim tekstom tablice.
- Pre obrade, crtice (`-`) su uklonjene iz labela radi uniformnosti.
- Slike su fotografisane mobilnim telefonom u različitim uslovima osvetljenja i pod različitim uglovima.

**Podela podataka** (izvršena pre treninga):

| Skup | Broj slika | Namena |
|------|------------|--------|
| Trening (`data/real/train`) | 80 (SRB+BIH) | Učenje modela |
| Validacija (`data/real/val`) | 20 (SRB+BIH) | Praćenje generalizacije – model ih NIJE video tokom treninga |
| Test (`data/real/test`) | 41 (EU) | Konačna evaluacija na potpuno novom formatu |

### 2.2 Ograničenja dataseta

- Mali broj primera (svega 141 slika ukupno).
- Neravnoteža između trening i test skupa — test čine isključivo EU tablice koje model nikada nije video tokom treniranja.
- Ova postavka je namerno izabrana kako bi se testirala sposobnost **generalizacije** modela na novi, neviđeni format.

## 3. Istraživanje arhitektura i izbor rešenja

### 3.1 Pokušani pristupi

Prvobitno je implementirana **CRNN (Convolutional Recurrent Neural Network)** arhitektura
sa CTC (Connectionist Temporal Classification) gubitkom, što je standardni pristup za OCR.
Međutim, CRNN je zahtevao veliku količinu podataka (10.000+) da bi postigao zadovoljavajuće
rezultate. Sa samo 100 trening slika, model nije uspevao da uči — tačnost je ostajala 0%.

### 3.2 Konačno rešenje: TrOCR

Kao alternativa izabran je **Microsoft TrOCR (Transformer-based Optical Character Recognition)**,
model `trocr-base-printed`, koji je već istreniran na ogromnom skupu štampanog teksta.
TrOCR kombinuje Vision Transformer (ViT) enkoder i GPT-2 dekoder, i pogodan je za
**fine‑tuning** na malim, specifičnim skupovima podataka.

Prednosti TrOCR‑a za ovaj projekat:
- Radi dobro sa malo podataka (zahvaljujući pre‑treniranom znanju),
- Lako se prilagođava novim fontovima i formatima,
- Jednostavna implementacija kroz Hugging Face biblioteke.

### 3.3 Fine‑tuning kao sopstveni trening (Transfer Learning)

Korišćenje pre‑treniranog modela nije „gotovo rešenje“ – sproveden je **sopstveni trening**
na **našem datasetu**. Ovaj proces se naziva **transfer learning** ili **fine‑tuning** i
podrazumeva sledeće:

1. Model je inicijalizovan težinama koje je Microsoft dobio treniranjem na opštem štampanom tekstu.
2. Zatim je **nastavljen trening** na naših 80 slika automobilskih tablica (SRB+BIH).
3. Tokom 20 epoha, model je prilagođavao svoje težine specifičnim fontovima, rasporedu karaktera i prisustvu ćirilice.
4. Pre treninga, model **nije umeo** da prepozna tekst sa naših tablica. Nakon treninga, postigao je **55% tačnost** na validacionom skupu (slike koje nije video).

Ovo je u potpunosti u skladu sa zahtevima **Nivoa 3** projektnog zadatka, jer:
- Model je **treniran** (nije samo pokrenut gotov),
- **Loss je praćen** kroz epohe,
- **Težine su menjane** na osnovu našeg dataseta,
- **Sopstveni dataset** je prikupljen, označen i pripremljen.

Transfer learning je standardna i preporučena praksa u oblasti dubokog učenja,
posebno kada radimo sa malim skupovima podataka kao što je naš.

## 4. Trening modela

### 4.1 Parametri treninga

| Parametar | Vrednost |
|-----------|----------|
| Model | `microsoft/trocr-base-printed` |
| Broj epoha | 20 |
| Veličina batch‑a | 2 |
| Stopa učenja | 5e-5 (sa linearnim opadanjem) |
| Optimizator | AdamW |
| Hardver | CPU (AMD Ryzen 7 5000) |
| Ukupno koraka | 800 |

### 4.2 Proces treninga

Trening je izvršen lokalno na računaru (CPU) uz pomoć Hugging Face biblioteka.
Korišćen je `Seq2SeqTrainer` sa sledećim podešavanjima:
- Bez među‑evaluacije tokom treninga (da bi se izbegli tehnički problemi sa `compute_metrics` na CPU‑u),
- Gubitak je Cross‑Entropy, gde su `<pad>` tokeni maskirani,
- Ručna evaluacija na validacionom skupu je izvršena odmah nakon treninga.

### 4.3 Tok gubitka (Loss)

| Korak | Training Loss |
|-------|---------------|
| 10 | 12.14 |
| 50 | 1.61 |
| 100 | 0.69 |
| 200 | 0.23 |
| 400 | 0.0002 |
| 800 | 0.0001 |

Loss je rapidno opao sa početnih 12.14 na manje od 0.001, što pokazuje da je model
uspešno naučio obrasce iz trening skupa.

## 5. Rezultati evaluacije

Za evaluaciju su korišćene dve metrike:
- **Character Error Rate (CER)** — prosečan broj pogrešnih karaktera (Levenshtein distance / dužina reference),
- **Tačnost cele tablice (Word Accuracy)** — procenat potpuno tačno prepoznatih tablica.

### 5.1 Rezultati po skupovima

| Skup | Broj slika | CER | Tačnost cele tablice |
|------|------------|-----|----------------------|
| SRB + BIH (validacija) | 20 | 0.146 | **55.0%** |
| EU tablice (test) | 41 | 0.517 | **0.0%** |

### 5.2 Primeri uspešnih i neuspešnih predikcija

**Uspešno (SRB/BIH validacija):** 11 od 20 tablica tačno prepoznato (npr. `CK964GE`, `VAВА224UL`).

**Neuspešno (SRB/BIH validacija) — tipični primeri:**

| Predviđeno | Tačno | Uočena greška |
|------------|-------|---------------|
| `E86O223` | `E86O723` | Zamena 0↔7 |
| `UEJ61T975` | `J67T975` | Višak slova na početku |
| `VRВР147Z` | `VRВР147ZC` | Nedostaje poslednji karakter |

**Neuspešno (EU test) — tipični primeri:**

| Predviđeno | Tačno | Uočena greška |
|------------|-------|---------------|
| `VUВУ125FB` | `VU125FB` | Ubacivanje ćiriličnog para `ВУ` |
| `BMБМ132CD` | `BM132CD` | Ubacivanje ćiriličnog para `БМ` |
| `BT832Б842` | `BT81342` | Ubacivanje ćirilice i promena brojeva |
| `MBНБ515EFEFEF` | `MB51EFJ` | Ubacivanje ćirilice i dupliranje sufiksa |

## 6. Analiza grešaka i diskusija

### 6.1 Overfitting na trening domen

Model je postigao solidnih 55% tačnosti na SRB/BIH validacionom skupu, što pokazuje
da je zaista naučio osnovne obrasce formata tablica na kojima je treniran.
Međutim, na EU tablicama pravi sistematske greške — **ubacuje ćirilične karaktere**
odmah iza latiničnih parova.

Ovo se objašnjava strukturom srpskih tablica koje su dominirale u trening skupu:
svaka srpska tablica sadrži dve latinične oznake grada nakon kojih slede **iste te
oznake napisane ćirilicom** (npr. `BG` → `BGБГ`). Model je "naučio" da iza latiničnog
para uvek dolazi njegov ćirilični par.

EU tablice nemaju ćiriličke znakove, pa model "halucinira" njihovo prisustvo.
CER od 0.517 na EU skupu znači da model u proseku pogreši **svaki drugi karakter**.

### 6.2 Ostale vrste grešaka

Na validacionom skupu (SRB/BIH):
- **Zamena sličnih karaktera** (`0` ↔ `7`, `D` ↔ `U`, `F` ↔ `E`),
- **Nedostajući ili višak karaktera** na početku/kraju tablice.

Na EU test skupu:
- **Dupliranje sufiksa** (`EF` → `EFEFEF`, `408` → `408408`),
- **Pogrešno prepoznavanje prefiksa** (`RBM` → `RPB`).

Sve ove greške su posledica malog trening skupa i preprilagođavanja modela specifičnom formatu trening tablica.

### 6.3 Ograničenja eksperimenta

- Trening skup je mali (80 slika) i neuravnotežen (samo SRB i BIH),
- Test skup (EU) nije bio zastupljen u treningu, što je dovelo do drastičnog pada tačnosti,
- Nije korišćena obimnija data augmentacija koja bi simulirala EU format.

## 7. Zaključak i moguća unapređenja

Ovaj projekat je pokazao da se fine‑tuningom velikog pre‑treniranog modela (TrOCR)
može postići **zadovoljavajuća tačnost** na malom, specijalizovanom skupu podataka (SRB/BIH tablice),
čak i kada se trenira na skromnom hardveru (CPU).

Takođe je jasno demonstrirano ključno ograničenje ovakvog pristupa — **loša generalizacija**
na neviđene formate (EU tablice), što je posledica overfitting‑a na specifičnu strukturu
trening podataka.

### Moguća unapređenja

1. **Proširenje dataseta** — dodati barem 20–30 EU tablica u trening skup,
2. **Data augmentacija** — dodati transformacije koje uklanjaju plavu traku sa EU tablica,
3. **Balansirani trening** — osigurati podjednaku zastupljenost sva tri regiona,
4. **Sintetički podaci** — generisati veštačke EU tablice bez ćirilice za trening,
5. **Više epoha** — nastaviti trening sa manjom stopom učenja radi stabilnije konvergencije.

## 8. Izvori

1. Microsoft TrOCR: [https://huggingface.co/microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)
2. Hugging Face Transformers biblioteka: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
3. Python biblioteke: PyTorch, Datasets, Accelerate, Pillow, Levenshtein, evaluate
4. Svi izvorni kodovi i podaci su dostupni u GitHub repozitorijumu ovog projekta.v