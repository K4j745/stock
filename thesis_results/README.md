# Wyniki do pracy inżynierskiej

Tabele wygenerowane skryptem `stock_ml/thesis_results.py` (2026-11-20). Skrypt odtwarza te same
foldy walk-forward co trening (okno treningowe 252 sesje, test 21, krok 21, 165 foldów na
spółkę) dla sześciu modeli: regresja logistyczna, las losowy, XGBoost, LightGBM, model świecowy
(XGBoost na cechach świecowych i bazowych) oraz ARIMA (test ADF + dobór rzędu BIC). Nie zapisuje
modeli, więc pliki w `stock_ml/models/saved` pozostają bez zmian.

**Sprawdzenie:** dla KO przy progu 0,2% średnia dokładność każdego z pięciu wcześniej raportowanych
modeli zgadza się z `stock_ml/reports/KO_bin2_results.json` co do 1e-9.

```bash
cd stock_ml
python thesis_results.py collect --ticker AAPL --threshold 0.002   # dla każdej spółki i progu
python thesis_results.py summarize
```

## Pliki

| Plik | Zawartość |
|---|---|
| `udzial_klas.csv` | liczba obserwacji po rozgrzewce oraz udział BUY i SELL dla każdej spółki i progu, z wierszem `RAZEM` |
| `klasyfikacja_podsumowanie.csv` | średnie metryk (dokładność, precyzja, czułość, F1, ROC AUC, MCC) dla modelu i progu; `*_sd_spolki` = odchylenie standardowe średnich spółek, `*_sd_foldy` = odchylenie standardowe po wszystkich 1650 foldach |
| `klasyfikacja_spolki.csv` | te same metryki dla każdej spółki osobno |
| `backtest_podsumowanie.csv` | backtest na predykcjach spoza próby: zwrot całkowity, CAGR, zmienność, Sharpe, maks. obsunięcie, Calmar, liczba dni |
| `backtest_spolki.csv` | backtest dla każdej spółki osobno, dodatkowo ekspozycja (udział dni z pozycją) i liczba zmian pozycji |
| `metadane.json` | parametry walk-forward, okres backtestu, lista modeli |
| `raw/{TICKER}_{TAG}_predictions.csv` | predykcje spoza próby: data, zwrot następnego dnia, etykieta, decyzja i prawdopodobieństwo każdego modelu |
| `raw/{TICKER}_{TAG}_folds.json` | metryki każdego foldu z datami okien; dla ARIMA także wybrany rząd |
| `dashboard_migawka_2026-11-19_v2/` | migawka portfeli dashboardu po poprawce momentu wykonania, z porównaniem przed i po (patrz niżej) |
| `dashboard_migawka_2026-11-19/` | migawka z CI sprzed poprawki, archiwum |

Progi: `bin2` = 0,2%, `bin5` = 0,5%, `bin10` = 1,0%. Dla 1,0% istnieją tylko wyniki walk-forward;
na dashboard nie zapisano modeli dla tego progu.

## Zasady backtestu

- Strategia modelu: pozycja długa w spółce, gdy model przewiduje BUY, w przeciwnym razie gotówka.
- Decyzja z dnia t korzysta wyłącznie z danych do zamknięcia dnia t i daje zwrot od zamknięcia t
  do zamknięcia t+1.
- Portfel: równe wagi 10 spółek, rebalansowany codziennie (średnia dziennych zwrotów spółek).
- Wspólny okres dla wszystkich strategii: dni, dla których każdy model ma predykcję spoza próby,
  czyli 2012-06-12 – 2026-03-24 (3465 sesji).
- Punkty odniesienia w tym samym okresie: SPY kup i trzymaj oraz koszyk 10 spółek kup i trzymaj
  (równe wagi).
- **Bez kosztów transakcyjnych i poślizgu** — ani ten backtest, ani symulacja portfeli na
  dashboardzie ich nie uwzględniają (warianty z kosztami: punkt 3 niżej).
- Sharpe: średni dzienny zwrot / odchylenie standardowe × √252, stopa wolna od ryzyka 0.

## Dodatkowe analizy (`stock_ml/thesis_extras.py`)

Liczone z tych samych predykcji spoza próby co tabele powyżej (`raw/`), bez ponownego trenowania.
Uzasadnienie metodyczne i źródła: `Praca_inz/Rekomendacje_metodyczne_z_literatura_2026-11-20.md`.

```bash
cd stock_ml
python thesis_extras.py all        # albo: pooled | costs | random | shap
```

### 1. Metryki łączne (`klasyfikacja_laczna_podsumowanie.csv`, `klasyfikacja_laczna_spolki.csv`)

- **Metoda:** precyzja, czułość, F1 i MCC liczone z jednej macierzy pomyłek zsumowanej po wszystkich
  foldach. W podsumowaniu macierz sumowana jest też po spółkach, czyli 34 650 predykcji na model i próg.
- **Kolumny:** `tp`, `fp`, `fn`, `tn`, `buy_share_true` (udział BUY w danych), `buy_share_predicted`
  (udział dni, w których model wskazał BUY).
- **Dlaczego:** średnie po foldach z `klasyfikacja_podsumowanie.csv` liczą precyzję jako 0 w foldzie
  bez ani jednej predykcji BUY (`zero_division=0` w `models/evaluate.py`), co je zaniża.
  - W metrykach łącznych taki przypadek daje wartość nieokreśloną (puste pole), a nie 0.
- **ROC AUC** zostaje średnią po foldach (`roc_auc_fold_mean`). Łączenie AUC między foldami
  wymaga spójnej kalibracji prawdopodobieństw (Forman i Scholz 2010).
- **Stabilność:** średnie i odchylenia po foldach z `klasyfikacja_podsumowanie.csv` nadal ją opisują.
- **Dokładność** łączna jest równa średniej po foldach, bo foldy mają tę samą długość.

### 2. Klasyfikator „zawsze SELL” (`klasyfikator_zawsze_sell.csv`, kolumna `always_sell_accuracy`)

Dokładność klasyfikatora, który zawsze przewiduje SELL, wynosi 1 − udział BUY. W osobnym pliku jest
liczona na dniach ocenianych przez cztery główne modele, dla każdej spółki i łącznie (`RAZEM`).
W tabelach łącznych jest liczona na dniach danego modelu.

### 3. Backtest z kosztami (`backtest_koszty_podsumowanie.csv`, `backtest_koszty_spolki.csv`)

- **Warianty kosztów:** 0, 5 i 10 punktów bazowych za każde wejście lub wyjście z pozycji
  (półobrót, ang. *half-turn*). Za Krauss i in. (2017) oraz Fischer i Krauss (2018), którzy
  przyjmują 5 pb za półobrót.
- **Naliczanie:** koszt odejmowany jest od zwrotu dnia, w którym zmienia się pozycja. Otwarcie
  pozycji pierwszego dnia też się liczy, zamknięcia na końcu okresu się nie dolicza.
- **Kup i trzymaj** (SPY i koszyk) płaci tylko jedno wejście.
- **Pozostałe zasady** są jak w backtestcie bez kosztów: te same kolumny, wariant 0 pb jest
  identyczny z `backtest_podsumowanie.csv`.

### 4. Strategie losowe o tej samej ekspozycji (`losowe_strategie.csv`)

Dla każdego modelu, progu i wariantu kosztów jest 1000 losowań (ziarno 20261120). Oba warianty
mają dokładnie tę samą ekspozycję (udział dni w pozycji) i liczbę zmian pozycji co model, więc
ponoszą te same koszty. Różnią się od modelu tylko momentem wejść i wyjść. Kolumna `null` wskazuje
wariant:

- **`shared_shift` (wariant główny):** pozycje modelu przesunięte cyklicznie w czasie o losowe
  opóźnienie, wspólne dla wszystkich spółek, co najmniej 21 sesji.
  - Zachowane zostaje także to, że model często trzyma wiele spółek jednocześnie, czyli
    zmienność portfela jest porównywalna.
- **`run_shuffle`:** osobno dla każdej spółki odcinki „w pozycji” i „w gotówce” ustawione
  w losowej kolejności.
  - Rozbija to jednoczesność pozycji między spółkami, przez co losowy portfel jest bardziej
    zdywersyfikowany i ma niższą zmienność.
  - Ten wariant zaniża więc ocenę modelu w Sharpe i obsunięciu; podano go dla porównania.

Kolumny:

- Dla Sharpe’a, CAGR i maksymalnego obsunięcia podano wartość modelu (`*_model`), średnią
  oraz percentyle 5, 50 i 95 rozkładu losowego.
- `*_percentile` to odsetek losowań, które model pobił lub wyrównał. Dla obsunięcia „lepiej”
  oznacza wartość bliższą zeru.

Interpretacja:

- **Sens testu:** modele są w gotówce przez 60–65% dni, więc mniejsze obsunięcie niż kup
  i trzymaj może wynikać z samej ekspozycji. Percentyl bliski 50 oznacza brak przewagi
  w wyczuciu momentu; za istotną przewagę można uznać percentyl ≥ 95.
- **Percentyl CAGR** prawie nie zależy od kosztów, bo model i losowania płacą tyle samo.
- **Percentyl Sharpe’a** rośnie z kosztami. To efekt mechaniczny: modele mają wyższą zmienność
  niż losowe przesunięcia, bo częściej są w pozycji w okresach dużych wahań, a ten sam koszt
  obniża Sharpe’a tym mniej, im wyższa zmienność. Wzrost nie oznacza więc lepszego wyczucia
  momentu.

### 5. SHAP dla LightGBM (`wykresy_shap_lightgbm/`, `shap_top10_xgboost_lightgbm.csv`, `shap_top10_zgodnosc.csv`)

- **Wykresy:** zbiorczy i słupkowy SHAP dla LightGBM, z polskimi opisami osi, dla 10 spółek i progów
  0,2% i 0,5%. Liczone na modelach z dashboardu i ostatnich 500 obserwacjach, tak jak dla XGBoost,
  którego wykresy są w `docs/charts/plots/`.
- **`shap_top10_xgboost_lightgbm.csv`:** 10 najważniejszych cech obu modeli według średniej |SHAP|.
- **`shap_top10_zgodnosc.csv`:** ile cech z pierwszej dziesiątki modele mają wspólnych i czy zgadza
  się cecha najważniejsza.
- Las losowy pominięto (opcjonalny, do załącznika).

### 6. Reguły techniczne bez uczenia maszynowego (strategia `technical_rules`)

- **Sygnały:** dokładnie funkcja `dashboard/lib/signals.py::technical_rule_based` z ustawieniami
  z `dashboard/config.json` (`technical_signals`), czyli te same reguły co portfel `reguly_tech`
  na dashboardzie. Liczona w `stock_ml/thesis_results.py::technical_rule_positions`.
  - Cztery reguły na stronę: RSI(14) poniżej 35 / powyżej 65, przecięcie MACD z linią sygnału,
    zamknięcie poza wstęgą Bollingera (20, 2), zamknięcie powyżej / poniżej SMA20 i SMA50.
  - Decyduje wynik netto (liczba reguł kupna minus sprzedaży); przy remisie kupno, jeśli
    zamknięcie jest powyżej SMA50.
- **Bez informacji z przyszłości:** wskaźniki (biblioteka `ta`) to okna wsteczne, więc decyzja
  z dnia t korzysta wyłącznie z danych do zamknięcia dnia t. Sprawdza to test
  `stock_ml/tests/test_thesis_rules.py`: pozycje policzone na danych uciętych są identyczne
  z pozycjami policzonymi na pełnych danych.
- **Zasady backtestu:** jak dla modeli — pozycja długa przy BUY, inaczej gotówka, zwrot od
  zamknięcia t do zamknięcia t+1, ten sam wspólny okres (2012-06-12 – 2026-03-24, 3465 sesji),
  równe wagi i codzienne równoważenie. Notowania pochodzą z tego samego źródła co dane badawcze.
- **Progi etykiet:** reguły od nich nie zależą, więc ich wiersze są takie same dla `bin2`, `bin5`
  i `bin10`. Powtórzono je w każdym progu dla spójności tabel.
- **Gdzie są wyniki:** `technical_rules` jako strategia w `backtest_podsumowanie.csv`,
  `backtest_spolki.csv`, `backtest_koszty_*.csv` (0/5/10 pb) i `losowe_strategie.csv` (oba
  warianty, z własną ekspozycją). Reguły nie są klasyfikatorem etykiet, więc nie ma ich
  w tabelach klasyfikacji.
- **Aktywność:** w podsumowaniach backtestu doszły kolumny `exposure` (średni po spółkach udział
  dni w pozycji) i `position_changes` (łączna liczba zmian pozycji w 10 spółkach) dla każdej
  strategii.
- **Losowania:** reguły mają własny strumień liczb losowych (ziarno 20261121), więc losowania
  dla modeli są identyczne jak przed dodaniem reguł.

Wyniki (bez kosztów, okres wspólny):

| Strategia | Zwrot całkowity | CAGR | Sharpe | Maks. obsunięcie | Ekspozycja | Zmiany pozycji |
|---|---|---|---|---|---|---|
| SPY kup i trzymaj | 531,6% | 14,3% | 0,886 | −33,7% | — | — |
| Koszyk kup i trzymaj | 748,6% | 16,8% | 1,130 | −30,9% | — | — |
| Reguły techniczne | 298,7% | 10,6% | 1,255 | −9,0% | 61,5% | 5902 |
| LightGBM, próg 0,2% (najlepszy model) | 163,6% | 7,3% | 0,897 | −13,3% | 38,6% | 11 250 |

- **Z kosztami** Sharpe reguł wynosi 0,996 przy 5 pb i 0,737 przy 10 pb, bo zmieniają pozycję
  rzadziej niż modele drzewiaste.
- **Wobec losowego wyczucia momentu** (wariant `shared_shift`, 0 pb) percentyl CAGR wynosi
  tylko 64, czyli brak przewagi w zwrocie. Percentyl Sharpe’a to 97, a obsunięcia 100.
- **Interpretacja:** przewaga reguł polega na ograniczaniu ryzyka. Filtr trendu (SMA50)
  wychodzi z rynku w spadkach, więc zmienność i obsunięcie są dużo mniejsze niż w losowym
  wyczuciu momentu o tej samej ekspozycji. To odwrotność modeli ML, które częściej trzymają
  pozycję w okresach dużych wahań.

## Migawka dashboardu po poprawce (`dashboard_migawka_2026-11-19_v2/`) — do cytowania

- **Błąd i poprawka:** do 2026-11-20 symulator portfeli (`dashboard/lib/portfolio.py`) wykonywał
  sygnał z dnia d po cenie otwarcia tego samego dnia. Sygnał korzysta jednak z zamknięcia dnia d,
  czyli z informacji, której w chwili transakcji jeszcze nie było (*lookahead*). Po poprawce
  sygnał z dnia d jest wykonywany po otwarciu następnej sesji. Kup i trzymaj nie korzysta
  z informacji rynkowej, więc nadal wchodzi pierwszego dnia.
- **Jak liczono:** skrypt `dashboard/thesis_snapshot.py` pobiera notowania tak jak `generate.py`,
  ucina je na 2026-11-19 przed jakimkolwiek obliczeniem (tylko kompletne sesje), liczy sygnały
  i uruchamia 10 portfeli dwa razy: po poprawce i z wykonaniem tego samego dnia. Nic nie zapisuje
  do `docs/data`.
  ```bash
  python dashboard/thesis_snapshot.py --end 2026-11-19 --out thesis_results/dashboard_migawka_2026-11-19_v2 --compare-same-day
  ```
- **Pliki:**
  - `portfele_metryki.csv`: wyniki po poprawce, te same kolumny co w starej migawce.
  - `portfele_metryki_ten_sam_dzien.csv`: wyniki z wykonaniem tego samego dnia.
  - `porownanie_przed_po_poprawce.csv`: obie wersje i różnica.
  - `zrodlo.json`: źródła sygnałów, commit, data.
- **Model świecowy:** tu działa prawdziwy model (`candle`), nie sygnał zastępczy.
- **Kontrola:** wariant z wykonaniem tego samego dnia odtwarza migawkę z CI
  (`dashboard_migawka_2026-11-19/`) co do 4 miejsc po przecinku dla wszystkich portfeli poza
  `candle` (tam był proxy) i `ensemble` (głosuje m.in. sygnałem świecowym). Różnica przed i po
  wynika więc wyłącznie z momentu wykonania.

**Wielkość błędu (zwrot w całym oknie, przed → po):**

| Portfel | Przed | Po |
|---|---|---|
| rf | 65,4% | 63,9% |
| xgb | 19,1% | 17,8% |
| candle | 36,7% | 35,7% |
| ensemble | 50,8% | 49,6% |
| reguly_tech | 29,8% | 29,4% |
| lr | 25,8% | 25,9% |
| lgbm | 38,4% | 38,5% |
| arima | 2,2% | 2,2% |

Sharpe zmienia się najwyżej o 0,03. Od 2026-02-24 różnice są poniżej 0,5 pp.

**Dlaczego błąd jest mały:** portfele modeli zaczynają od 10 akcji każdej spółki (pozycje
zalążkowe) i na każdy sygnał kupują lub sprzedają tylko 1 akcję. Większość wyniku pochodzi więc
z pozycji zalążkowych, a nie z sygnałów. Z tego samego powodu wyniki portfeli dashboardu słabo
mierzą jakość modeli; do tego służy backtest badawczy opisany wyżej.

**Uwaga:** `arima` od 2026-02-24 prawie nie handluje (zwrot ok. 0), więc jego Sharpe w tym okresie
(−0,27 przed, −1,50 po) nie ma sensownej interpretacji.

## Migawka dashboardu z CI (`dashboard_migawka_2026-11-19/`) — archiwum, sprzed poprawki

- **Źródło:** dane wygenerowane przez GitHub Actions 2026-11-19 23:27 UTC, commit `f18686ed`,
  po zamknięciu sesji, więc z kompletnymi świecami. Okno: 3 lata, 2023-11-20 – 2026-11-19.
- **`portfele_metryki.csv`:** metryki 10 portfeli dla całego okna i osobno od 2026-02-24.
  Od tej daty dane są nowe dla modeli finalnych, które trenowano na 2025-02-21 – 2026-02-23.
  Okres od 2026-02-24 liczony jest od stanu portfela z tego dnia, bez resetu pozycji.
- **Zastrzeżenia — nie cytować:**
  - Portfele z sygnałami wykonywały transakcje tego samego dnia, korzystając z zamknięcia,
    którego jeszcze nie było (patrz wyżej).
  - Portfel świecowy korzystał z sygnału zastępczego (`candle_proxy`) z powodu błędu ładowania
    modelu przy pandas 3. Błąd poprawiono w `stock_ml/models/candle_model.py`.
- **Kopie plików źródłowych:** `portfolios_index.json`, `meta.json`, `zrodlo.json` (zawiera m.in.
  źródło sygnału każdego modelu).
