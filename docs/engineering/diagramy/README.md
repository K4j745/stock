# Diagramy do pracy inżynierskiej

Diagramy w formacie tekstowym PlantUML (`.puml`) i wyrenderowane wersje `.svg`. Wspólny styl jest
w `_styl.iuml` i każdy diagram dołącza go przez `!include`, więc pliki renderuje się z tego
katalogu. Wersje do wklejania (ze stylem wstawionym w plik) i PNG 300 dpi są w materiałach do
tekstu pracy.

Prefiks `R` oznacza ryciny w tekście głównym, a prefiks `Z` ryciny w załączniku (osobna
numeracja Z1–Z20). Wszystkie diagramy zostały sprawdzone z kodem; daty foldów przeliczono na
danych AAPL i XOM.

## Tekst główny

| Plik | Ryc. | Diagram | Rozdział |
|---|---|---|---|
| `R01_walk_forward_schemat` | 1 | Schemat walidacji walk-forward z oknem kroczącym (252 / 21 / 21), diagram czasowy | Metodyka |
| `R02_arima_fold` | 2 | Aktywność: ARIMA w jednym foldzie (ADF, BIC, prognoza jednokrokowa) | Metodyka |
| `R03_przypadki_uzycia_przeglad` | 3 | Przypadki użycia, widok ogólny | Projekt i implementacja |
| `R04_komponenty_przeglad` | 4 | Komponenty systemu | Projekt i implementacja |
| `R05_wdrozenie` | 5 | Wdrożenie | Projekt i implementacja |
| `R06_symulacja_portfela` | 6 | Aktywność: symulacja portfela monitorującego (sygnał z sesji d−1 wykonywany na otwarciu sesji d) | Projekt i implementacja |

## Załącznik

| Plik | Ryc. | Diagram |
|---|---|---|
| `Z01a_przypadki_uzycia_serwis` | Z1 | Przypadki użycia serwisu WWW |
| `Z01b_przypadki_uzycia_potok_ml` | Z2 | Przypadki użycia potoku uczenia maszynowego |
| `Z01c_przypadki_uzycia_aktualizacja` | Z3 | Przypadki użycia cyklicznej aktualizacji i publikacji |
| `Z01d_przypadki_uzycia_generator` | Z4 | Przypadki użycia generatora danych serwisu |
| `Z02a_komponenty_stock_ml` | Z5 | Komponenty potoku ML (`stock_ml`) |
| `Z02b_komponenty_dashboard` | Z6 | Komponenty generatora danych (`dashboard`) |
| `Z03a_aktywnosc_dane_cechy` | Z7 | Aktywność: dane, cechy i etykiety |
| `Z03b_aktywnosc_walidacja_modele` | Z8 | Aktywność: walidacja walk-forward i modele finalne |
| `Z04_os_czasu_walk_forward` | Z9 | Foldy walk-forward na osi czasu na tle okna danych serwisu |
| `Z05a_sekwencja_notowania` | Z10 | Sekwencja: uruchomienie generatora i pobranie notowań |
| `Z05b1_sekwencja_sygnaly_ml` | Z11 | Sekwencja: sygnały reguł technicznych i modeli ML |
| `Z05b2_sekwencja_sygnaly_arima` | Z12 | Sekwencja: sygnały modelu ARIMA |
| `Z05c_sekwencja_portfele_eksport` | Z13 | Sekwencja: symulacja portfeli i eksport danych |
| `Z05d_sekwencja_publikacja_odczyt` | Z14 | Sekwencja: publikacja i odczyt danych przez użytkownika |
| `Z06_aktywnosc_ci_cd` | Z15 | Aktywność: workflow GitHub Actions |
| `Z07a_model_danych_sygnaly` | Z16 | Model danych: rejestr, notowania, sygnały, podsumowania modeli |
| `Z07b_model_danych_portfele` | Z17 | Model danych: portfele monitorujące |
| `Z08a_klasy_generator_dane` | Z18 | Klasy: orkiestracja, pobieranie danych i eksport |
| `Z08b_klasy_sygnaly_arima` | Z19 | Klasy: sygnały i model ARIMA |
| `Z08c_klasy_portfele` | Z20 | Klasy: symulacja portfeli i metryki |

Diagramy zastępują wcześniejszy komplet `01_…` – `11_…`. Duże diagramy (przypadki użycia,
komponenty, sekwencja, model danych, klasy, potok ML) podzielono na części czytelne na stronie A4.

## Renderowanie

Wymagana Java i PlantUML (`plantuml.jar` z [plantuml.com/download](https://plantuml.com/download)
lub z wydań na GitHubie; Graphviz jest wbudowany). Polecenia uruchamia się z tego katalogu:

```bash
java -jar plantuml.jar -charset UTF-8 -tsvg *.puml
java -DPLANTUML_LIMIT_SIZE=8192 -jar plantuml.jar -charset UTF-8 -tpng -Sdpi=300 *.puml
```
