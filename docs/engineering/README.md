# Dokumentacja inżynierska systemu StocKK Forecast

Katalog `docs/engineering/` zawiera dokumentację projektową powstałą na potrzeby rozdziału
„Inżynieria oprogramowania" pracy inżynierskiej *Prognozowanie krótkoterminowych zmian cen akcji
giełdowych z wykorzystaniem metod uczenia maszynowego oraz symulacja strategii inwestycyjnej*
(Kajetan Kaczyński, Uniwersytet Przyrodniczy w Poznaniu).

## Zawartość katalogu

| Plik | Opis |
|------|------|
| `uml_use_case.puml` | Diagram przypadków użycia w notacji PlantUML |
| `uml_sequence.puml` | Diagram sekwencji dla cyklicznej aktualizacji dashboardu i odczytu danych przez użytkownika |
| `uml_components.puml` | Diagram komponentów przedstawiający warstwy systemu |
| `README.md` | Ten dokument — te same trzy diagramy w notacji Mermaid, renderowane bezpośrednio przez GitHub |
| `WYMAGANIA.md` | Rozdział z wymaganiami funkcjonalnymi i niefunkcjonalnymi wraz z macierzą pokrycia |

Pliki `.puml` są źródłem diagramów w jakości nadającej się do druku. Można je wygenerować lokalnie:

```bash
plantuml -tsvg docs/engineering/uml_use_case.puml
plantuml -tsvg docs/engineering/uml_sequence.puml
plantuml -tsvg docs/engineering/uml_components.puml
```

Poniższe wersje w notacji Mermaid są renderowane przez GitHub bez dodatkowych narzędzi i służą
do szybkiego podglądu w przeglądarce.

## Krótka charakterystyka systemu

StocKK Forecast składa się z trzech niezależnie uruchamianych części:

1. **Pipeline uczenia maszynowego (`stock_ml/`)** — uruchamiany lokalnie przez autora, pobiera
   notowania z Yahoo Finance, buduje macierz cech (ok. 30 wskaźników technicznych oraz cechy
   świecowe), trenuje pięć wariantów klasyfikatorów w schemacie walidacji kroczącej
   (*walk-forward*), zapisuje modele oraz skalery na dysku lokalnym i generuje raporty.
2. **Generator dashboardu (`dashboard/`)** — uruchamiany cyklicznie przez GitHub Actions, pobiera
   bieżące notowania, wyznacza sygnały BUY/SELL (reguły techniczne, modele ML, głosowanie
   większościowe, benchmark *buy & hold*) i zapisuje wyniki jako statyczne pliki JSON i CSV
   w katalogu `docs/data/`.
3. **Serwis WWW (`docs/`)** — statyczne strony hostowane na GitHub Pages. Cała logika działa
   po stronie przeglądarki; system nie ma back-endu, bazy danych ani kont użytkowników.

---

## 1. Diagram przypadków użycia

Mermaid nie posiada natywnej notacji diagramu przypadków użycia, dlatego poniższy diagram
odwzorowuje jego strukturę za pomocą grafu skierowanego: aktorzy znajdują się po lewej stronie,
przypadki użycia zostały pogrupowane w trzy granice systemu, a relacje `<<include>>`
i `<<extend>>` opisano etykietami krawędzi. Wersja kanoniczna znajduje się w pliku
`uml_use_case.puml`.

```mermaid
flowchart LR
    U(("Użytkownik strony"))
    B(("Autor / Badacz"))
    CI(("Harmonogram CI<br/>GitHub Actions"))
    YF(("Yahoo Finance<br/>yfinance"))

    subgraph WWW["Serwis WWW - GitHub Pages, pliki statyczne"]
        UC01["UC-01 Przeglądanie panelu<br/>sygnałów BUY/SELL"]
        UC02["UC-02 Filtrowanie sygnałów"]
        UC03["UC-03 Pobranie eksportu CSV"]
        UC04["UC-04 Wykresy cen<br/>świece, RSI, MACD"]
        UC05["UC-05 Wykresy analizy ML"]
        UC06["UC-06 Symulacja portfela"]
        UC07["UC-07 Zapis portfela<br/>w localStorage"]
        UC08["UC-08 Eksport / import<br/>portfela JSON"]
        UC09["UC-09 Przełączenie waluty<br/>USD / PLN"]
        UC10["UC-10 Zmiana języka PL / EN"]
        UC11["UC-11 Zmiana motywu"]
        UC12["UC-12 Lektura instrukcji<br/>i metodologii"]
        UC13["UC-13 Wczytanie artefaktów<br/>JSON z docs/data"]
        UC14["UC-14 Podgląd metadanych"]
        UC15["UC-15 Przegląd rynku<br/>KPI, mapa ciepła"]
    end

    subgraph ML["Pipeline ML - stock_ml, uruchamiany offline"]
        UC20["UC-20 Pobranie danych OHLCV"]
        UC21["UC-21 Archiwizacja historii<br/>do CSV"]
        UC22["UC-22 Budowa macierzy cech"]
        UC23["UC-23 Trening modeli<br/>LR, RF, XGBoost, LightGBM"]
        UC24["UC-24 Walidacja walk-forward<br/>252 / 21 / 21"]
        UC25["UC-25 Strojenie hiperparametrów<br/>Optuna"]
        UC26["UC-26 Trening modelu świecowego"]
        UC27["UC-27 Ewaluacja modeli"]
        UC28["UC-28 Backtest strategii"]
        UC29["UC-29 Wykresy i raporty"]
        UC30["UC-30 Analiza SHAP"]
    end

    subgraph GEN["Generator dashboardu i publikacja - dashboard/, CI"]
        UC40["UC-40 Generowanie artefaktów<br/>statycznych"]
        UC41["UC-41 Wczytanie artefaktów ML"]
        UC42["UC-42 Zejście do proxy<br/>przy braku artefaktów"]
        UC43["UC-43 Wyznaczenie sygnałów"]
        UC44["UC-44 Cykliczna aktualizacja<br/>cron pn-pt 17:00 UTC"]
        UC45["UC-45 Publikacja dashboardu"]
        UC46["UC-46 Pobranie kursu USD/PLN"]
    end

    U --- UC15
    U --- UC01
    U --- UC04
    U --- UC05
    U --- UC06
    U --- UC10
    U --- UC11
    U --- UC12
    U --- UC14

    B --- UC20
    B --- UC21
    B --- UC23
    B --- UC26
    B --- UC27
    B --- UC28
    B --- UC29
    B --- UC30
    B --- UC40

    CI --- UC44
    UC20 --- YF
    UC40 --- YF
    UC46 --- YF

    UC15 -. include .-> UC13
    UC01 -. include .-> UC13
    UC04 -. include .-> UC13
    UC05 -. include .-> UC13
    UC06 -. include .-> UC13
    UC06 -. include .-> UC07
    UC14 -. include .-> UC13
    UC23 -. include .-> UC22
    UC23 -. include .-> UC24
    UC26 -. include .-> UC22
    UC22 -. include .-> UC20
    UC27 -. include .-> UC22
    UC28 -. include .-> UC22
    UC30 -. include .-> UC22
    UC40 -. include .-> UC43
    UC40 -. include .-> UC41
    UC40 -. include .-> UC46
    UC43 -. include .-> UC41
    UC44 -. include .-> UC40
    UC44 -. include .-> UC45

    UC02 -. extend .-> UC01
    UC03 -. extend .-> UC01
    UC08 -. extend .-> UC06
    UC09 -. extend .-> UC06
    UC25 -. extend .-> UC23
    UC42 -. extend .-> UC41
    UC21 -. extend .-> UC20
```

### Jak czytać ten diagram

- **Aktorzy.** *Użytkownik strony* to anonimowy odwiedzający — system nie zna jego tożsamości,
  ponieważ nie istnieje mechanizm rejestracji. *Autor / Badacz* uruchamia pipeline ML na własnej
  stacji roboczej. *Harmonogram CI* jest aktorem systemowym inicjującym aktualizację danych.
  *Yahoo Finance* jest systemem zewnętrznym, z którego pobierane są notowania.
- **Granice systemu.** Trzy prostokąty odpowiadają trzem katalogom repozytorium (`docs/`,
  `stock_ml/`, `dashboard/`) i jednocześnie trzem odrębnym środowiskom uruchomieniowym:
  przeglądarce użytkownika, stacji autora oraz maszynie wirtualnej GitHub Actions.
- **Relacja `<<include>>`** oznacza zachowanie obowiązkowe. Każde wyświetlenie danych na stronie
  wymaga wcześniejszego wczytania artefaktów JSON (UC-13), a trening modelu zawsze obejmuje
  budowę macierzy cech (UC-22) i walidację kroczącą (UC-24).
- **Relacja `<<extend>>`** oznacza zachowanie opcjonalne, uruchamiane po spełnieniu warunku
  rozszerzenia. Najistotniejszym przypadkiem jest UC-42: gdy w środowisku CI brakuje plików
  modeli lub zestaw cech nie zgadza się z zapisanym skalerem, generator przechodzi na
  deterministyczne funkcje proxy.
- **Rozdzielenie UC-23 i UC-40** odzwierciedla fakt, że modele nie są trenowane w CI —
  katalog `stock_ml/models/saved/` jest objęty plikiem `.gitignore`.

---

## 2. Diagram sekwencji

Scenariusz obejmuje pełny cykl życia danych: od uruchomienia zadania cyklicznego, przez pobranie
notowań, wyznaczenie sygnałów i publikację artefaktów, aż po wczytanie plików JSON przez
przeglądarkę odwiedzającego. Wersja kanoniczna znajduje się w pliku `uml_sequence.puml`.

```mermaid
sequenceDiagram
    autonumber
    actor CI as Harmonogram CI
    participant GEN as generate.py
    participant CFG as config_loader
    participant FETCH as data_fetcher
    participant IND as indicators
    participant MLL as ml_loader
    participant SIG as signals
    participant EXP as exporters
    participant YF as yfinance / Yahoo Finance
    participant ART as stock_ml/models/saved
    participant DOCS as Artefakty docs/data
    participant REPO as Repozytorium Git / Pages
    participant BR as Przeglądarka
    actor User as Użytkownik strony

    rect rgb(245, 247, 249)
    note over CI, EXP: Faza 1-2: uruchomienie zadania i pobranie danych rynkowych
    CI->>GEN: python dashboard/generate.py
    GEN->>CFG: load_config("dashboard/config.json")
    CFG-->>GEN: tickery, modele, label_version=bin2, progi

    loop dla każdego tickera
        GEN->>FETCH: fetch_prices(ticker)
        FETCH->>YF: yfinance.download(ticker)
        YF-->>FETCH: ramka OHLCV lub pusty wynik
        alt pobranie udane
            FETCH-->>GEN: DataFrame OHLCV
        else błąd sieci / brak biblioteki / pusty wynik
            FETCH-->>GEN: None - ticker pominięty
        end
    end

    alt żaden ticker nie został pobrany
        GEN-->>CI: ERROR "No prices downloaded - aborting"
        note right of GEN: Poprzednia wersja docs/data<br/>pozostaje opublikowana
    end
    end

    rect rgb(245, 247, 249)
    note over GEN, ART: Faza 3: cechy techniczne i prawdopodobieństwa modeli
    loop dla każdego pobranego tickera
        GEN->>IND: add_indicators(df)
        IND-->>GEN: SMA, EMA, MACD, RSI, BB, ATR, OBV
        GEN->>MLL: model_probabilities(ticker, model, df)
        MLL->>ART: wczytaj model i scaler - tag bin2
        ART-->>MLL: artefakt lub brak pliku
        alt artefakty obecne i zgodne z zestawem cech
            MLL-->>GEN: prawdopodobieństwa, source=model
        else brak artefaktów lub niezgodność cech
            MLL-->>GEN: proxy, model_version=proxy-1.0
            note right of MLL: Flaga use_proxy_when_artifacts_missing.<br/>W CI to jest gałąź domyślna
        end
        GEN->>SIG: technical_rule_based, from_probabilities, buy_and_hold
        SIG-->>GEN: sygnały cząstkowe
        GEN->>SIG: ensemble_majority(min_votes=2)
        SIG-->>GEN: sygnał zespołowy
    end
    GEN->>FETCH: fetch_fx("USDPLN=X")
    alt kurs pobrany
        FETCH-->>GEN: kurs USD/PLN
    else brak danych
        FETCH-->>GEN: wartość zapasowa 4.0
    end
    end

    rect rgb(245, 247, 249)
    note over GEN, REPO: Faza 4-5: eksport artefaktów i publikacja
    GEN->>EXP: write_signals_bundle, write_price_series,<br/>write_registry, write_global_exports
    EXP->>DOCS: zapis JSON i CSV
    DOCS-->>EXP: potwierdzenie
    GEN->>DOCS: meta.json, last_updated.json
    GEN-->>CI: kod wyjścia 0
    alt wykryto zmiany w docs/
        CI->>REPO: git add, commit, push
        REPO-->>CI: publikacja na GitHub Pages
    else brak zmian
        CI->>CI: pominięcie commita
    end
    end

    rect rgb(245, 247, 249)
    note over User, DOCS: Faza 6: wejście użytkownika na stronę
    User->>BR: otwarcie adresu GitHub Pages
    BR->>REPO: GET index.html, app.js, style.css
    REPO-->>BR: pliki statyczne
    BR->>BR: odczyt localStorage - język, motyw, waluta, portfele
    par żądania o dane
        BR->>DOCS: GET data/signals/latest.json
        DOCS-->>BR: najnowsze sygnały
    and
        BR->>DOCS: GET data/registry/*.json, data/fx.json
        DOCS-->>BR: słowniki i kurs walutowy
    and
        BR->>DOCS: GET data/meta.json, data/last_updated.json
        DOCS-->>BR: metadane przebiegu
    end
    alt odpowiedź HTTP 200
        BR-->>User: panel z aktualnymi sygnałami
    else błąd pobrania pliku
        BR->>DOCS: GET data/signals.json - format zapasowy
        DOCS-->>BR: starszy format lub błąd
        BR-->>User: komunikat o braku danych
    end
    opt symulacja portfela
        User->>BR: dodanie transakcji kupna / sprzedaży
        BR->>BR: przeliczenie pozycji i zapis w localStorage
        BR-->>User: zaktualizowana wycena
        note right of BR: Dane portfela nie opuszczają przeglądarki
    end
    end
```

### Jak czytać ten diagram

- **Podział na fazy.** Sześć faz odpowiada kolejnym etapom przebiegu: konfiguracja, pobranie
  danych, przetwarzanie, eksport, publikacja i konsumpcja danych przez użytkownika. Fazy 1–5
  wykonują się bez udziału człowieka, faza 6 dopiero po wejściu odwiedzającego na stronę.
- **Rozdzielenie w czasie.** Między fazą 5 a 6 może upłynąć dowolnie dużo czasu — użytkownik
  zawsze ogląda stan zamrożony w momencie ostatniego commita, a nie stan bieżący rynku.
  Znacznik czasu tej wersji danych publikowany jest w pliku `last_updated.json` (UC-14).
- **Pierwszy blok `alt`** dotyczy pojedynczego tickera: nieudane pobranie powoduje jego
  pominięcie, a nie przerwanie całego przebiegu. Dopiero brak *wszystkich* notowań przerywa
  zadanie bez commita, dzięki czemu poprawne dane nie zostają nadpisane pustym zestawem.
- **Drugi blok `alt`** przedstawia zejście do proxy. Jest to kluczowa decyzja architektoniczna:
  ponieważ artefakty ML nie trafiają do repozytorium, w środowisku CI wykonywana jest gałąź
  proxy, a wynikowy sygnał jest oznaczany wartością `proxy-1.0` w polu `model_version`.
- **Blok `par`** oznacza równoległe żądania HTTP wykonywane przez przeglądarkę. Pliki są od
  siebie niezależne, więc nie występuje efekt kaskady żądań.
- **Blok `opt`** obejmuje symulację portfela — funkcję opcjonalną, w całości klienta,
  bez komunikacji z jakimkolwiek serwerem aplikacyjnym.

---

## 3. Diagram komponentów

Diagram przedstawia sześć warstw systemu wraz z kierunkami zależności oraz interfejsami,
przez które warstwy się komunikują. Wersja kanoniczna znajduje się w pliku `uml_components.puml`.

```mermaid
flowchart TB
    subgraph L1["1. Warstwa danych zewnętrznych"]
        YF["Yahoo Finance<br/>serwis zewnętrzny"]
        YFLIB["yfinance<br/>klient HTTP"]
        I_OHLCV(["Interfejs: OHLCV + kurs USD/PLN"])
        YF --- I_OHLCV
        YFLIB --- I_OHLCV
    end

    subgraph L2["2. Pipeline ML - stock_ml, offline"]
        ML_CLI["main.py<br/>interfejs CLI"]
        ML_CFG["config.py<br/>tickery, progi, okna walk-forward"]
        ML_DATA["data<br/>pobieranie i archiwizacja CSV"]
        ML_FEAT["features<br/>indicators, candles, validation"]
        ML_MODELS["models<br/>train, tune, evaluate, candle_model"]
        ML_BT["backtest<br/>run, portfolio"]
        ML_REP["reports<br/>generate, plots, shap_analysis"]
        ART[("models/saved/<br/>modele i skalery<br/>poza kontrolą wersji")]
        ML_OUT[("reports/<br/>CSV, JSON, PNG")]
        I_ART(["Interfejs: artefakty modeli"])
        I_PNG(["Interfejs: wykresy PNG"])

        ML_CLI --> ML_DATA
        ML_CLI --> ML_MODELS
        ML_CLI --> ML_BT
        ML_CLI --> ML_REP
        ML_CFG -.-> ML_DATA
        ML_CFG -.-> ML_MODELS
        ML_DATA --> ML_FEAT
        ML_FEAT --> ML_MODELS
        ML_FEAT --> ML_BT
        ML_MODELS --> ART
        ML_BT --> ML_OUT
        ML_REP --> ML_OUT
        ART --- I_ART
        ML_OUT --- I_PNG
    end

    subgraph L3["3. Generator dashboardu - dashboard/"]
        GEN["generate.py<br/>orkiestracja przebiegu"]
        CFG["config_loader"]
        FETCH["data_fetcher"]
        IND["indicators"]
        MLL["ml_loader"]
        SIG["signals<br/>reguły, ML, ensemble, proxy"]
        PORT["portfolio"]
        EXP["exporters"]
        AUD["audit"]

        GEN --> CFG
        GEN --> FETCH
        GEN --> IND
        GEN --> MLL
        GEN --> SIG
        GEN --> PORT
        GEN --> EXP
        GEN --> AUD
        MLL --> SIG
        IND --> SIG
        SIG --> EXP
        PORT --> EXP
    end

    subgraph L4["4. Repozytorium artefaktów statycznych - docs/"]
        DOCSDATA[("docs/data/<br/>signals, prices, registry,<br/>fx, meta, CSV")]
        DOCSPNG[("docs/charts/plots/<br/>PNG + plots_index.json")]
        I_HTTP(["Interfejs: HTTP GET - JSON / CSV / PNG"])
        DOCSDATA --- I_HTTP
        DOCSPNG --- I_HTTP
    end

    subgraph L5["5. Warstwa prezentacji - przeglądarka"]
        HTML["Strony HTML<br/>index, signals, prices,<br/>charts, portfolio, methodology"]
        APPJS["assets/app.js<br/>i18n PL/EN, motyw, formatowanie"]
        UIPORT["Moduł portfela"]
        PLOTLY["Plotly.js 2.27"]
        CHARTJS["Chart.js 4.4"]
        LS[("localStorage<br/>portfele, język, motyw, waluta")]

        HTML --> APPJS
        HTML --> PLOTLY
        HTML --> CHARTJS
        HTML --> UIPORT
        UIPORT --> LS
        APPJS --> LS
    end

    subgraph L6["6. Automatyzacja i hosting"]
        CI["GitHub Actions<br/>dashboard.yml"]
        GIT["Repozytorium Git"]
        PAGES["GitHub Pages<br/>serwer plików statycznych"]
        CI --> GIT
        GIT --> PAGES
    end

    ML_DATA --> I_OHLCV
    FETCH --> I_OHLCV
    MLL --> I_ART
    GEN --> I_PNG
    EXP --> DOCSDATA
    AUD --> DOCSDATA
    GEN --> DOCSPNG
    HTML --> I_HTTP
    CI --> GEN
    PAGES --> I_HTTP
    PAGES -.-> HTML
```

### Jak czytać ten diagram

- **Kierunek zależności.** Strzałki prowadzą wyłącznie „w dół" łańcucha przetwarzania: warstwa
  prezentacji zależy od repozytorium artefaktów, repozytorium od generatora, generator od
  pipeline'u ML i źródła danych. Nie występuje zależność zwrotna — strona nie może wywołać
  żadnej funkcji generatora ani modelu.
- **Interfejsy.** Zaokrąglone elementy oznaczają punkty styku między warstwami. Interfejs
  `HTTP GET` jest jedynym sposobem, w jaki przeglądarka komunikuje się z danymi; interfejs
  `artefakty modeli` jest dostępny wyłącznie na stacji autora, ponieważ katalog
  `stock_ml/models/saved/` nie jest wersjonowany.
- **Brak komponentów serwerowych.** Na diagramie nie ma serwera aplikacyjnego ani bazy danych,
  co jest świadomą decyzją projektową. Konsekwencją jest brak kont użytkowników i przechowywanie
  stanu portfela wyłącznie w `localStorage` przeglądarki.
- **Rola warstwy 6.** GitHub Actions pełni funkcję planisty, a GitHub Pages funkcję serwera
  plików statycznych. Obie usługi są wymienne — system można opublikować na dowolnym hostingu
  statycznym, co opisano w wymaganiu przenośności w pliku `WYMAGANIA.md`.
- **Powiązanie z pipeline'em ML.** Warstwa 2 jest jedynym producentem artefaktów modeli.
  Ponieważ w środowisku CI te artefakty są niedostępne, komponent `ml_loader` implementuje
  ścieżkę zapasową opisaną na diagramie sekwencji.
