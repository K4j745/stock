# Inżynieria oprogramowania

Rozdział opisuje proces wytwórczy systemu StocKK Forecast oraz specyfikację wymagań, jakie postawiono
przed jego trzema podsystemami: pipeline'em uczenia maszynowego, generatorem artefaktów danych
oraz statycznym serwisem WWW. Wymagania zostały spisane na podstawie analizy zaimplementowanego
kodu, a przypisany im status realizacji odzwierciedla rzeczywisty stan repozytorium, nie zaś
pierwotne zamierzenia projektowe.

## 1. Proces wytwórczy

Projekt realizowany był jednoosobowo, w modelu iteracyjno-przyrostowym. Zamiast pełnego cyklu
kaskadowego, w którym całość wymagań ustalana jest przed rozpoczęciem implementacji, przyjęto
podział prac na krótkie iteracje kończące się działającym przyrostem funkcjonalności. Kolejne
iteracje obejmowały:

1. pozyskanie i archiwizację danych giełdowych oraz weryfikację ich kompletności;
2. budowę warstwy cech (wskaźniki techniczne, cechy świecowe) wraz z kontrolą przecieku danych;
3. implementację treningu modeli i walidacji kroczącej;
4. ewaluację, backtest oraz warstwę raportową;
5. budowę generatora artefaktów statycznych i automatyzację w środowisku ciągłej integracji;
6. budowę warstwy prezentacji oraz symulatora portfela;
7. dopracowanie dokumentacji, dwujęzyczności i warstwy informacyjno-prawnej.

Ze względu na jednoosobowy charakter zespołu zrezygnowano z formalnych artefaktów zarządczych
(rejestr sprintów, przeglądy zespołowe) na rzecz narzędzi lekkich:

- **Kontrola wersji.** Repozytorium Git z rozgałęzieniami tematycznymi dla poszczególnych iteracji.
  Historia commitów pełni funkcję dziennika prac; identyfikator commitu jest zapisywany
  w metadanych każdego przebiegu generatora (`docs/data/meta.json`), co pozwala odtworzyć,
  która wersja kodu wygenerowała daną wersję danych.
- **Ciągła integracja i dostarczanie.** GitHub Actions (`.github/workflows/dashboard.yml`)
  uruchamia generator według harmonogramu (`cron: 0 17 * * 1-5`, po zamknięciu sesji na giełdach
  amerykańskich) oraz na żądanie (`workflow_dispatch`). Zadanie kończy się automatycznym commitem
  i wypchnięciem katalogu `docs/`, co jest równoznaczne z publikacją nowej wersji serwisu
  na GitHub Pages.
- **Testy.** Testy jednostkowe w środowisku `pytest`, uruchamiane lokalnie.
- **Dokumentacja towarzysząca kodowi.** Pliki `stock_ml/README.md`, `docs/README.md`,
  `docs/data/README.md`, `docs/LICENSING.md` oraz niniejszy katalog `docs/engineering/`.

Istotną decyzją procesową było **rozdzielenie środowiska treningu i środowiska publikacji**.
Trening modeli odbywa się wyłącznie na stacji roboczej autora, a katalog `stock_ml/models/saved/`
został objęty plikiem `.gitignore`. Środowisko ciągłej integracji nie trenuje modeli — wykonuje
jedynie generator danych. Konsekwencje tej decyzji dla wymagań funkcjonalnych opisano w części 2.2.

## 2. Wymagania funkcjonalne

Wymagania oznaczono identyfikatorami `FR-nn`. Priorytet określono w trzystopniowej skali:
**krytyczny** (bez niego system nie realizuje celu pracy), **wysoki** (istotny dla wartości
merytorycznej lub użytkowej), **średni/niski** (funkcja uzupełniająca). Status realizacji
przyjmuje wartości: **zrealizowane**, **częściowo** (funkcja działa w ograniczonym zakresie lub
nie jest udostępniona użytkownikowi końcowemu), **planowane** (funkcja nie została zaimplementowana).

### 2.1. Pipeline uczenia maszynowego (`stock_ml/`)

| ID | Nazwa | Opis | Priorytet | Status |
|----|-------|------|-----------|--------|
| FR-01 | Pozyskanie notowań historycznych | System pobiera dzienne szeregi OHLCV dla zdefiniowanego koszyka dziesięciu spółek oraz indeksu odniesienia SPY, począwszy od 1 kwietnia 2011 r., korzystając z biblioteki `yfinance`. | krytyczny | zrealizowane |
| FR-02 | Archiwizacja danych źródłowych | Pobrane szeregi zapisywane są w lokalnym archiwum CSV, co umożliwia powtórzenie eksperymentu bez ponownego odpytywania źródła zewnętrznego. | średni | zrealizowane |
| FR-03 | Wyznaczenie wskaźników technicznych | Moduł cech oblicza około trzydziestu zmiennych objaśniających: średnie kroczące SMA i EMA, MACD wraz z linią sygnału i histogramem, RSI, oscylator stochastyczny, ROC, wstęgi Bollingera, ATR, OBV, względny wolumen, zwroty opóźnione oraz rozpiętość dzienną. | krytyczny | zrealizowane |
| FR-04 | Wyznaczenie cech formacji świecowych | System wylicza cechy opisujące geometrię świecy (korpus, cienie, położenie zamknięcia) wykorzystywane przez odrębny model świecowy. | wysoki | zrealizowane |
| FR-05 | Etykietowanie kierunku zmiany ceny | Zmienna objaśniana powstaje przez porównanie zwrotu z następnej sesji z zadanym progiem (podstawowy 0,2%, warianty porównawcze 0,5% i 1,0%). Wersja etykiety odwzorowywana jest w nazwie artefaktu jako znacznik `bin2`, `bin5` lub `bin10`. | krytyczny | zrealizowane |
| FR-06 | Zapobieganie przeciekowi danych | Z macierzy cech usuwane są kolumny bezpośrednio zależne od zmiennej objaśnianej, a standaryzacja dopasowywana jest wyłącznie na podzbiorze uczącym danego podziału. | krytyczny | zrealizowane |
| FR-07 | Walidacja krocząca | Ocena modeli prowadzona jest w schemacie walk-forward z oknem przesuwnym: 252 sesje uczące, 21 sesji testowych, krok 21 sesji, bez mieszania obserwacji w czasie. | krytyczny | zrealizowane |
| FR-08 | Trening zestawu klasyfikatorów | System trenuje regresję logistyczną, las losowy, XGBoost oraz LightGBM na wspólnej macierzy cech i wspólnym podziale czasowym. | krytyczny | zrealizowane |
| FR-09 | Trening modelu świecowego | Odrębny model wykorzystujący wyłącznie cechy formacji świecowych, stanowiący punkt odniesienia dla modeli wskaźnikowych. | wysoki | zrealizowane |
| FR-10 | Strojenie hiperparametrów | Optymalizacja hiperparametrów z użyciem biblioteki Optuna, z zapisem wariantów artefaktów oznaczonych przyrostkiem `_tuned`. | średni | zrealizowane |
| FR-11 | Utrwalanie artefaktów modeli | Model oraz dopasowany do niego skaler zapisywane są na dysku w formacie właściwym dla biblioteki (`.json`, `.txt`, `.joblib`), z nazwą zawierającą ticker i znacznik etykiety. | krytyczny | zrealizowane |
| FR-12 | Ewaluacja zapisanych modeli | Wyznaczenie miar jakości klasyfikacji: dokładność, precyzja, czułość, miara F1, współczynnik korelacji Matthewsa oraz pole pod krzywą ROC, w podziale na kolejne okna walidacji. | krytyczny | zrealizowane |
| FR-13 | Backtest strategii inwestycyjnej | Symulacja strategii opartej na sygnałach modelu z wyznaczeniem zwrotu całkowitego i zannualizowanego, zmienności, wskaźnika Sharpe'a, maksymalnego obsunięcia kapitału oraz wskaźnika Calmara, w zestawieniu ze strategią kup-i-trzymaj oraz indeksem SPY. | wysoki | częściowo |
| FR-14 | Raporty tabelaryczne | Eksport wyników walidacji i backtestów do plików CSV oraz JSON w katalogu raportów. | wysoki | zrealizowane |
| FR-15 | Wykresy analityczne | Generowanie wykresów: krzywa kapitału, macierz pomyłek, ważność cech, porównanie modeli. | wysoki | zrealizowane |
| FR-16 | Analiza wyjaśnialności SHAP | Wyznaczenie wartości SHAP i wygenerowanie wykresów zbiorczego, słupkowego oraz wodospadowego dla wskazanego modelu. | średni | częściowo |

**Uzasadnienie statusów odbiegających od pełnej realizacji:**

- **FR-13 — częściowo.** Moduł backtestu jest kompletny pod względem metryk portfelowych, jednak
  interfejs wiersza poleceń przyjmuje dla backtestu wersję etykiety ograniczoną do wartości
  `A` i `B`, podczas gdy artefakty modeli binarnych zapisywane są ze znacznikiem `bin2`
  (odpowiednio `bin5`, `bin10`). W efekcie ścieżki plików budowane w module backtestu nie wskazują
  na artefakty produkowane przez bieżący tryb etykietowania, a uruchomienie backtestu na modelach
  binarnych wymaga ręcznego wskazania znacznika. Ujednolicenie sposobu wyznaczania znacznika
  we wszystkich poleceniach interfejsu pozostaje do wykonania.
- **FR-16 — częściowo.** Analiza SHAP jest zaimplementowana i produkuje trzy typy wykresów,
  jednak wyniki nie trafiają do serwisu WWW. Zadanie ciągłej integracji wywołuje wyłącznie
  polecenia generujące raporty i wykresy standardowe, a katalog wykresów publikowanych
  (`docs/charts/plots/`) nie zawiera plików SHAP. Analiza pozostaje zatem narzędziem
  wykorzystywanym lokalnie na potrzeby opisu wyników w pracy, a nie funkcją serwisu.

### 2.2. Generator artefaktów danych (`dashboard/`)

| ID | Nazwa | Opis | Priorytet | Status |
|----|-------|------|-----------|--------|
| FR-17 | Wczytanie i walidacja konfiguracji | Generator odczytuje plik `dashboard/config.json` zawierający listę tickerów, listę modeli, wersję etykiety, progi reguł technicznych oraz definicje portfeli, uzupełniając brakujące klucze wartościami domyślnymi. | krytyczny | zrealizowane |
| FR-18 | Pobranie bieżących notowań | Dla każdego tickera pobierany jest aktualny szereg OHLCV; wynik jest buforowany na czas trwania przebiegu, aby ograniczyć liczbę zapytań do źródła. | krytyczny | zrealizowane |
| FR-19 | Pobranie kursu wymiany USD/PLN | Kurs pobierany jest z tego samego źródła i publikowany jako odrębny artefakt; w razie niepowodzenia stosowana jest jawnie zdefiniowana wartość zapasowa. | średni | zrealizowane |
| FR-20 | Wczytanie artefaktów modeli i kontrola zgodności | Generator próbuje wczytać model wraz ze skalerem oraz zweryfikować, czy zestaw cech odtworzony z danych bieżących odpowiada zestawowi zapamiętanemu w skalerze. | krytyczny | zrealizowane |
| FR-21 | Awaryjne wyznaczenie prawdopodobieństw | Gdy artefakty są niedostępne lub niezgodne, stosowana jest deterministyczna funkcja zastępcza, a wynik oznaczany jest w polu źródła sygnału oraz w wersji modelu jako pochodzący z proxy. Zachowanie sterowane jest flagą konfiguracyjną. | wysoki | zrealizowane |
| FR-22 | Sygnały z reguł technicznych | Wyznaczenie sygnału kupna lub sprzedaży na podstawie zestawu reguł opartych na wskaźnikach, z wynikiem netto i progiem liczby spełnionych reguł, z rozstrzygnięciem remisu według kierunku trendu. | krytyczny | zrealizowane |
| FR-23 | Sygnały z prawdopodobieństw modeli | Przekształcenie prawdopodobieństwa klasy wzrostowej w decyzję binarną według progu 0,5, dla każdego z pięciu strumieni modelowych. | krytyczny | zrealizowane |
| FR-24 | Sygnał zespołowy | Wyznaczenie decyzji zbiorczej metodą głosowania większościowego, z minimalną liczbą zgodnych głosów równą dwa. | wysoki | zrealizowane |
| FR-25 | Strategia odniesienia kup-i-trzymaj | Publikacja strumienia sygnałów odpowiadającego strategii pasywnej, stanowiącego punkt odniesienia dla pozostałych strumieni. | wysoki | zrealizowane |
| FR-26 | Wycena portfeli modelowych | Wyznaczenie i publikacja historii wartości portfeli prowadzonych według sygnałów poszczególnych modeli. | średni | częściowo |
| FR-27 | Eksport artefaktów JSON | Zapis pakietu sygnałów (najnowsze i historyczne), szeregów cenowych dla poszczególnych tickerów, rejestrów pomocniczych, kursu walutowego oraz metadanych do katalogu `docs/data/`. | krytyczny | zrealizowane |
| FR-28 | Eksport zbiorczy CSV | Publikacja zbiorczych plików CSV z sygnałami, umożliwiających dalszą analizę poza serwisem. | wysoki | zrealizowane |
| FR-29 | Metadane przebiegu | Zapis identyfikatora commitu, wersji użytych bibliotek, wersji etykiety, progów decyzyjnych, informacji o obecności artefaktów ML oraz znacznika czasu zakończenia przebiegu. | wysoki | zrealizowane |
| FR-30 | Zabezpieczenie przed publikacją pustego zestawu | Jeżeli nie udało się pobrać notowań dla żadnego tickera, przebieg jest przerywany z komunikatem błędu i bez wykonania commita, dzięki czemu poprzednia poprawna wersja danych pozostaje opublikowana. | wysoki | zrealizowane |
| FR-31 | Praca w trybie bez dostępu do sieci | Odtworzenie zestawu danych z lokalnego archiwum w sytuacji, gdy źródło zewnętrzne jest niedostępne. | średni | zrealizowane |
| FR-32 | Cykliczna aktualizacja i publikacja | Uruchomienie generatora według harmonogramu oraz na żądanie, wykrycie zmian w katalogu `docs/` i automatyczna publikacja nowej wersji serwisu. | krytyczny | zrealizowane |
| FR-33 | Udostępnienie wykresów analitycznych w serwisie | Skopiowanie wykresów wytworzonych przez pipeline ML do katalogu publikowanego wraz z indeksem opisującym dostępne pliki. | średni | zrealizowane |

**Uzasadnienie statusów odbiegających od pełnej realizacji:**

- **FR-26 — częściowo.** Moduł wyceny portfeli jest zaimplementowany, jednak w bieżącej
  konfiguracji lista portfeli jest pusta. Metadane przebiegu raportują zerową liczbę portfeli,
  a publikowany indeks portfeli jest pustą tablicą. Funkcja nie jest zatem widoczna dla
  użytkownika serwisu; jej włączenie wymaga wyłącznie uzupełnienia konfiguracji, co czyni
  to wymaganie kandydatem do domknięcia w kolejnej iteracji.
- **FR-31 — zrealizowane.** Moduł pobierania danych ponawia nieudane pobranie trzykrotnie
  z narastającym opóźnieniem, weryfikuje kompletność kolumn OHLCV, a po wyczerpaniu prób
  sięga po zatwierdzoną w repozytorium kopię CSV z katalogu archiwum i przycina ją do
  żądanego okresu. Archiwum jest utrzymywane przyrostowo: przebieg automatyczny dopisuje
  nowe notowania do istniejących plików zamiast je nadpisywać, dzięki czemu historia nie
  ulega skróceniu nawet przy krótkim oknie pobrania. Odporność na awarię źródła realizowana
  jest zatem zarówno na poziomie pozyskania danych, jak i publikacji (wymaganie FR-30),
  co usuwa wcześniejszą rozbieżność wobec deklaracji na stronie informacyjnej.

### 2.3. Serwis WWW (`docs/`)

| ID | Nazwa | Opis | Priorytet | Status |
|----|-------|------|-----------|--------|
| FR-34 | Panel przeglądowy rynku | Strona główna prezentuje wskaźniki zbiorcze, mapę ciepła zmian oraz zestawienie liderów wzrostów i spadków. | krytyczny | zrealizowane |
| FR-35 | Tabela sygnałów | Prezentacja sygnałów BUY/SELL w postaci tabeli z informacją o tickerze, sektorze, modelu, źródle sygnału, cenie i dacie. | krytyczny | zrealizowane |
| FR-36 | Filtrowanie i wyszukiwanie sygnałów | Zawężanie zbioru wyników według tickera, modelu, typu sygnału, zakresu dat, portfela oraz frazy wyszukiwania. | wysoki | zrealizowane |
| FR-37 | Oznaczenie źródła sygnału | Wizualne wyróżnienie sygnałów pochodzących z funkcji zastępczej proxy w odróżnieniu od sygnałów wyznaczonych przez wytrenowany model. | wysoki | zrealizowane |
| FR-38 | Macierz zgodności modeli | Zestawienie pokazujące, w jakim stopniu poszczególne strumienie sygnałów są ze sobą zgodne. | średni | zrealizowane |
| FR-39 | Pobranie eksportu CSV | Udostępnienie z poziomu strony odnośnika do zbiorczego pliku CSV z sygnałami. | średni | zrealizowane |
| FR-40 | Interaktywne wykresy cen | Wykres świecowy z wolumenem oraz panelami wskaźników RSI i MACD, z możliwością przybliżania i odczytu wartości. | wysoki | zrealizowane |
| FR-41 | Galeria wykresów analizy ML | Prezentacja opublikowanych wykresów analitycznych na podstawie indeksu plików, wraz z komunikatem stanu pustego, gdy wykresy nie zostały wygenerowane. | średni | zrealizowane |
| FR-42 | Symulator portfela | Rejestrowanie transakcji kupna i sprzedaży, obsługa wielu portfeli równolegle, bieżąca wycena pozycji oraz wynik zrealizowany i niezrealizowany. | wysoki | zrealizowane |
| FR-43 | Trwałość stanu portfela | Zapamiętanie zawartości portfeli między sesjami przeglądarki w pamięci lokalnej, bez przesyłania danych na serwer. | wysoki | zrealizowane |
| FR-44 | Eksport i import portfela | Zapis stanu portfela do pliku JSON, wczytanie stanu z pliku oraz wyczyszczenie danych. | średni | zrealizowane |
| FR-45 | Przełączanie waluty prezentacji | Przeliczanie wartości między dolarem amerykańskim a złotym na podstawie opublikowanego kursu, z zapamiętaniem wyboru. | średni | zrealizowane |
| FR-46 | Dwujęzyczność interfejsu | Przełączanie języka interfejsu między polskim a angielskim, z zapamiętaniem wyboru użytkownika. | wysoki | częściowo |
| FR-47 | Motyw jasny i ciemny | Przełączanie schematu kolorystycznego z zapamiętaniem wyboru oraz spójnym stosowaniem zmiennych CSS. | średni | zrealizowane |
| FR-48 | Strona metodologii | Opis źródła danych, sposobu konstrukcji cech, zasad etykietowania, schematu walidacji, zestawu modeli oraz sposobu wyznaczania sygnałów, w wersji polskiej i angielskiej. | wysoki | zrealizowane |
| FR-49 | Strona informacyjna „O stronie" | Informacje o projekcie, autorze i uczelni, pochodzeniu danych, licencji oraz zastrzeżeniu prawnym, w wersji polskiej i angielskiej. | wysoki | zrealizowane |
| FR-50 | Prezentacja aktualności danych | Wyświetlenie znacznika czasu ostatniej aktualizacji oraz wersji zestawu danych na podstawie opublikowanych metadanych. | wysoki | zrealizowane |
| FR-51 | Zastrzeżenie inwestycyjne | Widoczna w stopce każdej strony informacja, że prezentowane treści nie stanowią porady inwestycyjnej. | krytyczny | zrealizowane |
| FR-52 | Automatyczne odświeżanie widoku | Cykliczne ponowne pobranie artefaktów przez otwartą kartę przeglądarki. | niski | zrealizowane |
| FR-53 | Sekcja najczęściej zadawanych pytań | Zwięzły zestaw pytań i odpowiedzi wyjaśniających sposób interpretacji sygnałów oraz ograniczenia systemu. | niski | zrealizowane |
| FR-54 | Powiadomienia o nowych sygnałach | Mechanizm informowania odwiedzającego o pojawieniu się nowych sygnałów dla obserwowanych spółek. | niski | planowane |

**Uzasadnienie statusów odbiegających od pełnej realizacji:**

- **FR-46 — częściowo.** Mechanizm tłumaczeń oparty jest na atrybutach `data-i18n` i słowniku
  osadzonym w skrypcie wspólnym. Słownik obejmuje około stu kluczy, co pokrywa nawigację,
  nagłówki, stopkę i najważniejsze etykiety. Strony metodologii oraz informacyjna zawierają
  pełne wersje dwujęzyczne w postaci równoległych bloków treści. Poza tym zakresem pozostaje
  jednak część etykiet kolumn, podpowiedzi i komunikatów o błędach, które wyświetlane są
  wyłącznie w jednym języku niezależnie od wyboru użytkownika.
- **FR-53 — zrealizowane.** Strona instrukcji zawiera sekcję pomocy technicznej podzieloną
  na trzy grupy: pytania ogólne, interpretacja sygnałów oraz problemy techniczne. Sekcja
  występuje w obu wersjach językowych i odsyła do uzasadnienia doboru spółek oraz do
  dokumentu licencyjnego.
- **FR-54 — planowane.** Mechanizm powiadomień nie występuje w repozytorium. Byłby możliwy
  do zrealizowania wyłącznie po stronie klienta (porównanie zawartości pamięci lokalnej
  z nowo pobranym zestawem sygnałów), ponieważ architektura bezserwerowa wyklucza wysyłkę
  wiadomości.

## 3. Wymagania niefunkcjonalne

| ID | Kategoria | Wymaganie | Miara / kryterium akceptacji | Status |
|----|-----------|-----------|------------------------------|--------|
| NFR-01 | Wydajność | Czas wczytania strony przeglądowej nie przekracza dwóch sekund przy typowym łączu szerokopasmowym. | Sumaryczny rozmiar artefaktów pobieranych przez stronę główną (najnowsze sygnały, rejestry, kurs walutowy, metadane) nie przekracza 200 kB; plik z najnowszymi sygnałami ma około 105 kB. | zrealizowane |
| NFR-02 | Wydajność | Pojedynczy artefakt szeregu cenowego pozostaje na tyle mały, by wykres renderował się bez odczuwalnego opóźnienia. | Rozmiar pliku OHLCV dla jednego tickera mieści się w przedziale 370–380 kB. | zrealizowane |
| NFR-03 | Wydajność | Widok pełnej historii sygnałów nie wymaga pobrania całego zbioru danych na raz. | Strona sygnałów pobiera pojedynczy plik historii o rozmiarze rzędu 80 MB; wymagane jest wprowadzenie stronicowania lub podziału na pliki okresowe. | planowane |
| NFR-04 | Wydajność | Pełny przebieg generatora mieści się w limicie czasu zadania ciągłej integracji. | Przebieg kończy się powodzeniem w standardowym limicie usługi; w repozytorium nie prowadzi się jednak pomiaru czasu poszczególnych etapów. | częściowo |
| NFR-05 | Wydajność | Rozmiar publikowanego katalogu danych jest kontrolowany świadomie. | Generator domyślnie nie zapisuje pomocniczych plików cząstkowych sygnałów, ponieważ zwiększały one rozmiar repozytorium o około 165 MB. | zrealizowane |
| NFR-06 | Niezawodność i dostępność | Dostępność serwisu nie zależy od infrastruktury utrzymywanej przez autora. | Serwis jest zbiorem plików statycznych na GitHub Pages; brak własnego serwera aplikacyjnego i bazy danych oznacza brak punktów awarii po stronie projektu. | zrealizowane |
| NFR-07 | Niezawodność i dostępność | Awaria źródła danych nie powoduje utraty ani zafałszowania opublikowanych treści. | Pobranie jest ponawiane trzykrotnie z narastającym opóźnieniem, następnie wykorzystywana jest zatwierdzona kopia CSV; przy braku obu źródeł przebieg jest przerywany bez commita, a poprzednia wersja danych pozostaje dostępna. | zrealizowane |
| NFR-08 | Niezawodność i dostępność | Brak artefaktów modeli nie przerywa procesu generowania danych. | Moduł wczytujący modele przechodzi na deterministyczne funkcje zastępcze, oznaczając wynik odpowiednim identyfikatorem wersji. | zrealizowane |
| NFR-09 | Niezawodność i dostępność | Wyniki eksperymentów są odtwarzalne. | Ustalone ziarno generatora liczb losowych, deterministyczny podział czasowy oraz zapis wersji bibliotek w metadanych przebiegu. | zrealizowane |
| NFR-10 | Niezawodność i dostępność | Nieudany przebieg automatyczny jest sygnalizowany osobie odpowiedzialnej. | Brak dedykowanego kroku powiadamiania; wykrycie awarii wymaga ręcznego przeglądu historii uruchomień. | planowane |
| NFR-11 | Bezpieczeństwo | System nie przetwarza danych osobowych i nie prowadzi kont użytkowników. | Brak formularzy rejestracji, brak identyfikacji odwiedzającego, brak mechanizmu sesji. | zrealizowane |
| NFR-12 | Bezpieczeństwo | Dane wprowadzone przez użytkownika nie opuszczają jego urządzenia. | Portfele, wybór języka, motywu i waluty przechowywane są wyłącznie w pamięci lokalnej przeglądarki; nie istnieje kanał przesyłania ich na serwer. | zrealizowane |
| NFR-13 | Bezpieczeństwo | Repozytorium nie zawiera sekretów ani kluczy dostępowych. | Źródło danych nie wymaga klucza API, a zadanie ciągłej integracji korzysta wyłącznie z wbudowanego tokenu środowiska z jawnie ograniczonym uprawnieniem zapisu zawartości. | zrealizowane |
| NFR-14 | Bezpieczeństwo | Dane wstawiane do drzewa dokumentu są zabezpieczone przed wstrzyknięciem znaczników. | W module portfela zastosowano funkcję ucieczki znaków specjalnych; pozostałe widoki renderują dane pochodzące wyłącznie z własnego generatora, jednak nie stosują tego zabezpieczenia konsekwentnie. | częściowo |
| NFR-15 | Bezpieczeństwo | Zasoby pobierane z sieci dostarczania treści są weryfikowane, a polityka bezpieczeństwa treści ograniczona. | Biblioteki wykresów ładowane są z sieci CDN bez atrybutu kontroli integralności; nie zdefiniowano polityki bezpieczeństwa treści. | planowane |
| NFR-16 | Użyteczność i dostępność | Interfejs dostępny jest w języku polskim i angielskim. | Przełącznik języka działa i zapamiętuje wybór, jednak nie wszystkie ciągi znaków są objęte słownikiem tłumaczeń. | częściowo |
| NFR-17 | Użyteczność i dostępność | Użytkownik może wybrać jasny lub ciemny wariant kolorystyczny. | Motyw ustawiany atrybutem na elemencie głównym dokumentu i zapamiętywany w pamięci lokalnej; oba warianty zdefiniowane zmiennymi CSS. | zrealizowane |
| NFR-18 | Użyteczność i dostępność | Serwis pozostaje czytelny na urządzeniach mobilnych. | Reguły zapytań medialnych zdefiniowano dla arkuszy wspólnych oraz stron przeglądu, wykresów, cen i portfela; najszerszy widok — tabela sygnałów — nie ma dedykowanego układu mobilnego. | częściowo |
| NFR-19 | Użyteczność i dostępność | Kontrast tekstu i tła spełnia poziom AA normy WCAG 2.1. | Paleta barw projektowana z deklarowanym celem zgodności na poziomie AA; nie przeprowadzono jednak formalnego audytu ani nie udokumentowano jego wyników. | częściowo |
| NFR-20 | Użyteczność i dostępność | Interfejs jest obsługiwalny z klawiatury i opisany semantycznie dla technologii wspomagających. | W dokumentach występują pojedyncze atrybuty roli i etykiet dostępnościowych; brak systematycznego zarządzania kolejnością fokusu w elementach interaktywnych. | częściowo |
| NFR-21 | Użyteczność i dostępność | Animacje są ograniczane zgodnie z preferencją użytkownika dotyczącą redukcji ruchu. | Arkusze stylów zawierają regułę wyłączającą efekty przy ustawieniu preferencji ograniczenia ruchu. | zrealizowane |
| NFR-22 | Użyteczność i dostępność | Brak danych komunikowany jest zrozumiałym opisem, a nie pustym ekranem. | Widoki wykresów i sygnałów wyświetlają komunikat stanu pustego wraz z wyjaśnieniem przyczyny. | zrealizowane |
| NFR-23 | Przenośność | Serwis można opublikować na dowolnym serwerze plików statycznych. | Brak zależności od funkcji specyficznych dla hostingu; wszystkie odwołania do danych są względne. | zrealizowane |
| NFR-24 | Przenośność | Warstwa prezentacji nie wymaga etapu budowania ani środowiska uruchomieniowego. | Zastosowano czysty HTML, CSS i JavaScript bez frameworków i bez menedżera pakietów po stronie front-endu. | zrealizowane |
| NFR-25 | Przenośność | Część serwerowa uruchamia się na systemach Windows i Linux. | Środowisko określone wersją Pythona 3.11 oraz plikami zależności dla obu podsystemów; ścieżki budowane niezależnie od systemu operacyjnego. | zrealizowane |
| NFR-26 | Przenośność | Serwis działa w aktualnych wersjach przeglądarek opartych na silnikach Blink, Gecko i WebKit. | Wykorzystano wyłącznie standardowe API przeglądarkowe; nie przeprowadzono jednak testów wieloprzeglądarkowych. | częściowo |
| NFR-27 | Utrzymywalność | Kod podzielony jest na moduły o jednej odpowiedzialności. | Podsystemy rozdzielone na katalogi tematyczne; moduły generatora odpowiadają pojedynczym etapom przetwarzania. | zrealizowane |
| NFR-28 | Utrzymywalność | Parametry sterujące wydzielone są z kodu. | Progi, listy tickerów i parametry walidacji zdefiniowano w module konfiguracyjnym pipeline'u oraz w pliku konfiguracyjnym generatora. | zrealizowane |
| NFR-29 | Utrzymywalność | Kluczowa logika przetwarzania objęta jest testami jednostkowymi. | W repozytorium znajduje się jeden moduł testowy obejmujący osiem przypadków dla cech świecowych; moduły generatora sygnałów i eksportu nie mają testów. | częściowo |
| NFR-30 | Utrzymywalność | Testy uruchamiane są automatycznie przy każdej zmianie kodu. | Jedyny zdefiniowany przepływ pracy ciągłej integracji odpowiada za generowanie danych; nie zawiera kroku uruchamiania testów. | planowane |
| NFR-31 | Utrzymywalność | Każdy przebieg jest identyfikowalny i możliwy do powiązania z wersją kodu. | Metadane przebiegu zawierają identyfikator commitu, wersje bibliotek, wersję etykiety oraz informację o obecności artefaktów modeli. | zrealizowane |
| NFR-32 | Utrzymywalność | Projekt posiada dokumentację techniczną i projektową. | Dokumentacja modułów, opis struktury katalogu danych, opis metodologii, analiza licencyjna oraz niniejszy katalog dokumentacji inżynierskiej z trzema diagramami UML. | zrealizowane |
| NFR-33 | Zgodność prawna | Źródło danych jest jednoznacznie wskazane wraz ze statusem wykorzystywanego narzędzia. | Strona informacyjna wskazuje Yahoo Finance jako źródło oraz zaznacza, że biblioteka pobierająca nie jest oficjalnym interfejsem tego serwisu ani nie jest z nim powiązana. | zrealizowane |
| NFR-34 | Zgodność prawna | Serwis zawiera zastrzeżenie o braku charakteru doradztwa inwestycyjnego. | Zastrzeżenie umieszczone w stopce wszystkich stron, w opisie metodologii oraz jako wyróżniony komunikat na stronie informacyjnej. | zrealizowane |
| NFR-35 | Zgodność prawna | Ryzyko redystrybucji danych zostało przeanalizowane i udokumentowane. | Odrębny dokument omawia podstawę wykorzystania danych, licencję biblioteki pobierającej, regulamin dostawcy, status prawny notowań, przyjęte środki ostrożności oraz alternatywne źródła. | zrealizowane |
| NFR-36 | Zgodność prawna | Warunki wykorzystania kodu źródłowego są jawnie określone. | Repozytorium nie zawiera pliku licencji kodu, mimo że warunki wykorzystania danych zostały opisane szczegółowo. | planowane |

## 4. Macierz pokrycia

Macierz wiąże przypadki użycia z diagramu przypadków użycia (`uml_use_case.puml`) z wymaganiami
funkcjonalnymi. Pozwala to wykazać, że każdy zidentyfikowany przypadek użycia ma odpowiadające mu
wymaganie oraz że żadne wymaganie krytyczne nie pozostaje poza modelem zachowania systemu.

| Przypadek użycia | Wymagania funkcjonalne | Status pokrycia |
|------------------|------------------------|-----------------|
| UC-01 Przeglądanie panelu sygnałów | FR-35, FR-37, FR-38 | pełne |
| UC-02 Filtrowanie sygnałów | FR-36 | pełne |
| UC-03 Pobranie eksportu CSV | FR-39, FR-28 | pełne |
| UC-04 Przeglądanie wykresów cen | FR-40 | pełne |
| UC-05 Przeglądanie wykresów analizy ML | FR-41, FR-33, FR-15 | częściowe (brak wykresów SHAP — FR-16) |
| UC-06 Symulacja portfela | FR-42 | pełne |
| UC-07 Zapis stanu portfela | FR-43 | pełne |
| UC-08 Eksport / import portfela | FR-44 | pełne |
| UC-09 Przełączenie waluty | FR-45, FR-19 | pełne |
| UC-10 Zmiana języka | FR-46 | częściowe |
| UC-11 Zmiana motywu | FR-47 | pełne |
| UC-12 Lektura instrukcji i metodologii | FR-48, FR-49, FR-51, FR-53 | pełne |
| UC-13 Wczytanie artefaktów JSON | FR-27 | pełne |
| UC-14 Podgląd metadanych | FR-50, FR-29 | pełne |
| UC-15 Przeglądanie rynku | FR-34 | pełne |
| UC-20 Pobranie danych OHLCV | FR-01 | pełne |
| UC-21 Archiwizacja historii | FR-02 | pełne |
| UC-22 Budowa macierzy cech | FR-03, FR-04, FR-05, FR-06 | pełne |
| UC-23 Trening modeli | FR-08, FR-11 | pełne |
| UC-24 Walidacja walk-forward | FR-07 | pełne |
| UC-25 Strojenie hiperparametrów | FR-10 | pełne |
| UC-26 Trening modelu świecowego | FR-09 | pełne |
| UC-27 Ewaluacja modeli | FR-12 | pełne |
| UC-28 Backtest strategii | FR-13 | częściowe |
| UC-29 Generowanie wykresów i raportów | FR-14, FR-15 | pełne |
| UC-30 Analiza SHAP | FR-16 | częściowe |
| UC-40 Generowanie artefaktów statycznych | FR-17, FR-18, FR-27, FR-28, FR-29, FR-33 | pełne |
| UC-41 Wczytanie artefaktów ML | FR-20 | pełne |
| UC-42 Zejście do proxy | FR-21, FR-37 | pełne |
| UC-43 Wyznaczenie sygnałów | FR-22, FR-23, FR-24, FR-25, FR-26 | częściowe (portfele modelowe niepublikowane — FR-26) |
| UC-44 Cykliczna aktualizacja danych | FR-32, FR-30 | pełne |
| UC-45 Publikacja dashboardu | FR-32 | pełne |
| UC-46 Pobranie kursu USD/PLN | FR-19 | pełne |

Wymagania nieprzypisane do żadnego przypadku użycia to FR-31 (praca bez dostępu do sieci),
FR-52 (automatyczne odświeżanie widoku) oraz FR-54 (powiadomienia). FR-31 i FR-52 realizują
zachowania systemowe niewymagające udziału aktora, a FR-54 pozostaje w fazie planowania,
dlatego świadomie pominięto je na diagramie przypadków użycia.

## 5. Podsumowanie stanu realizacji

Zdefiniowano **54 wymagania funkcjonalne** oraz **36 wymagań niefunkcjonalnych**. Wśród wymagań
funkcjonalnych 49 ma status zrealizowanych, cztery zrealizowano częściowo (FR-13, FR-16, FR-26,
FR-46), a jedno pozostaje w fazie planowania (FR-54). Wszystkie wymagania oznaczone
priorytetem krytycznym są zrealizowane w pełni, co oznacza, że system realizuje cel pracy
w zakładanym zakresie: pozyskuje dane, buduje cechy, trenuje i waliduje modele bez przecieku
informacji z przyszłości, wyznacza sygnały inwestycyjne, symuluje strategię i publikuje wyniki
w postaci dostępnego publicznie serwisu.

Wśród wymagań niefunkcjonalnych 23 uznano za zrealizowane, osiem za zrealizowane częściowo,
a pięć za planowane (NFR-03, NFR-10, NFR-15, NFR-30, NFR-36). Największy dług techniczny dotyczy
trzech obszarów: rozmiaru artefaktu historii sygnałów pobieranego przez przeglądarkę, pokrycia
testami automatycznymi oraz formalnego potwierdzenia zgodności z wytycznymi dostępności.
Obszary te wyznaczają naturalny kierunek dalszego rozwoju systemu.
