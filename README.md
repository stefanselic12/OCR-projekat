# OCR Projekat - Prepoznavanje registarskih tablica

Ovaj projekat služi za automatsko čitanje teksta sa registarskih tablica vozila koristeći OCR tehnologiju.

## Šta projekat radi?
* **Detektuje** tablice na slikama.
* **Ispravlja** perspektivu (ako je slika slikana pod uglom).
* **Prepoznaje** karaktere koristeći obučeni model.

## Struktura
* `data/` - Folder sa slikama za trening i testiranje.
* `src/` - Izvorni kod za procesiranje slika i model.
* `main.py` - Glavni fajl za pokretanje prepoznavanja.
* `dokumentacija.md` - Detaljan opis metodologije i rezultata.
