## OCR Automobilskih tablica 

## Tema 

Tema ovog projekta je prepoznavanje teksta sa automobilskih tablica korišćenjem CRNN (Convolutional Recurrent Neural Network) modela. Ulazni podaci modela su slike tablica (SRB, BIH i EU), dok na izlazu treba da dobijemo tekstualnu reprezentaciju registarske oznake. Projekat takođe istražuje uticaj različitih dekoder strategija (Greedy vs Beam Search) na tačnost prepoznavanja. 

## Priprema podataka 

Pošto je prikupljanje velikog broja realnih slika tablica ograničavajući faktor, dataset je proširen sintetičkim generisanjem podataka. Na osnovu realnih primera tablica napravljen je generator koji: 

- Nasumično kombinuje slova (A-Ž, uključujući Č, Ć, Š, Ž) i brojeve (0-9) u skladu sa pravilima registarskih oznaka (dužina 5–10 karaktera) 

- Generisani tekst se renderuje kao slika sa fontom sličnim tablicama 

- Dodaju se blage deformacije, šum i varijacije u osvetljenju kako bi sintetičke slike ličile na realne uslove 

Na ovaj način je generisano 513 sintetičkih slika, koje su zajedno sa realnim snimcima (ukupno 654 slike) korišćene u dataset-u. Time je značajno smanjen rizik od overfittinga, a model je naučio da generalizuje na veći broj varijacija tekstova nego što bi to bilo moguće sa samo realnim podacima. 

Podela dataset-a na trening (393), validaciju (131) i test (130)  dok su sintetički podaci ravnomerno raspoređeni kroz sva tri skupa. 

## Balansiranje 

Balansiranje karaktera: Analizom realnih podataka uočen je disbalans među slovima (npr. slovo Z se javljalo mnogo češće od slova Č). Primenjena je strategija fokusiranja samo na retka slova. 

- Cilj je postignut kada svaki karakter ima najmanje 50 pojavljivanja. 

- Za svaki karakter koji nije dostigao cilj, generisane su sintetičke tablice. 

- Na ovaj način, odnos max/min među karakterima je smanjen sa preko 10x na manje od 3x. 

## Model 

Model korišćen u ovom projektu je CRNN (Convolutional Recurrent Neural Network), izabran je zato što je standardni pristup za prepoznavanje teksta na slikama koji 

kombinuje prednosti CNN-a za izdvajanje prostornih karakteristika i RNN-a za modelovanje sekvencijalne prirode teksta. 

Delovi modela su: 

- CNN (Convolutional Neural Network) – ekstrahuje prostorne karakteristike sa slike. Konvolucioni slojevi smanjuju prostorne dimenzije dok povećavaju broj kanala, izlaz je feature map dimenzija 256×4×64 → 1024×64. 

- RNN (Recurrent Neural Network) – obrađuje sekvence karakteristika koje dolaze iz CNN-a. U ovom projektu korišćen je dvosmerni LSTM (Bidirectional LSTM) koji može da "vidi" kontekst sa obe strane. 

- CTC (Connectionist Temporal Classification) – omogućava modelu da uči bez preciznog poravnanja između ulazne slike i izlaznog teksta. CTC se koristi prilikom treninga, dok se u inferenci koriste dekoderi (Greedy ili Beam Search). 

Ukupan broj parametara modela je 3,560,679. 

## Dekoderi 

Projekat poredi dve strategije za dekodiranje CTC izlaza: 

Greedy dekoder 

- Na svakom vremenskom koraku bira karakter sa najvećom verovatnoćom 

- Zatim spaja duplikate i uklanja blank znakove 

- Jednostavan i brz 

Beam Search dekoder 

- Održava više najverovatnijih pretpostavki (beam width = 7 i 10) 

- Na kraju bira pretpostavku sa najvećom ukupnom verovatnoćom 

- Složeniji, ali teoretski može dati bolje rezultate za duže sekvence 

Zaključak analize je da, uz trenutni obim dataset-a i nivo istreniranosti, Greedy dekoder predstavlja optimalno rešenje koje minimizuje šum i greške u predikciji. 

Analiza izazova i poteškoća 

Tokom procesa razvoja identifikovano je više kjučnih izazova koja su direktno uticala na performanse sistema: 

- Disbalans karaktera: Pojedini karakteri su bili zastupljeni u znatno manjoj meri, što je rešeno ciljanim generisanjem sintetičkih uzoraka radi postizanja ravnomerne distribucije. 

- Kvalitet sintetičkih slika: Inicijalni pokušaji su ukazali na preveliku uniformnost sintetičkih slika, što je prevaziđeno uvođenjem šuma i varijacija u fontovima kako bi se simulirali realni uslovi. 

- Nepogodni uglovi snimanja: Sistem koristi dva načina da tablicu pročita pravilno, čak i kada kamera nije postavljena idealno: 

   1. **Automatsko ispravljanje (Pretprocesiranje):** Pre nego što slika stigne do modela, OpenCV kod pronalazi ivice tablice i automatski je rotira tako da stoji vodoravno. Ovim se dobija "čista" slika, a zahvaljujući kvalitetnoj obradi (Lanczos interpolacija), karakteri ostaju oštri i jasni. 

   2. **Trening na iskrivljenim slikama (Augmentacija):** Pošto se slika ponekad vidi pod uglom (kao trapez), sistem je naučen da prepoznaje i takve oblike. Tokom treninga, modelu namerno pokazujemo puno slika koje su veštački iskrivljene. Tako model "vežba" i uči da prepozna slova i brojeve bez obzira na to da li je slika savršeno ravna ili blago nagnuta. 

- Sistemski problemi _Beam Search_ dekodera: Uočeno je da Beam Search ima tendenciju ka **skraćivanju sekvenci (omission errors)** , posebno kod dužih stringova ili u uslovima vizuelne nesigurnosti. Ovo ukazuje na to da algoritam favorizuje kraće putanje sa manjim brojem "blank" karaktera, što dovodi do gubitka informacija. ( `PB9880HE -> PB980HE -` Izostavljen karakter _8_ ) 

- Vizuelne konfuzije modela (OCR karakteristike): Neke greške se ponavljaju nezavisno od dekodera, što ukazuje na to da sam **CRNN model** teško pravi distinkciju između određenih vizuelno sličnih karaktera. (H/L, 0/O, Š/S, I/1/7...) 


## Zaključak 

Razvijeni sistem uspešno demonstrira robusno prepoznavanje registarskih tablica, koristeći napredne metode geometrijske normalizacije i augmentacije za prevazilaženje izazova u realnim uslovima eksploatacije. Postignuta preciznost potvrđuje da integracija pretprocesiranja slike sa CRNN arhitekturom omogućava pouzdan rad sistema čak i pri varijacijama u osvetljenju i uglovima snimanja. Ovako koncipirano rešenje predstavlja čvrstu osnovu za implementaciju u produkcionim sistemima za automatizovanu kontrolu pristupa i nadzor. 

Stefan Selić, Luka Sikimić - FTN 

