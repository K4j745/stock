# Zasady i licencje wykorzystania danych źródłowych w projekcie „Stock4caster”

> **Zastrzeżenie.** Niniejszy dokument ma charakter informacyjny i porządkujący na potrzeby
> pracy inżynierskiej. **Nie stanowi porady prawnej.** Zawarte tu oceny są opinią autora
> opartą na lekturze publicznie dostępnych regulaminów i licencji; w razie wątpliwości co do
> dopuszczalności konkretnego sposobu wykorzystania danych należy zasięgnąć opinii prawnika
> lub jednostki prawnej uczelni.

> **Status weryfikacji źródeł.** Treść regulaminów serwisów internetowych bywa zmieniana bez
> ogłoszenia. Sekcje oznaczone jako **[do weryfikacji]** opisują postanowienia zgodnie z ich
> znanym brzmieniem i ogólnym charakterem, jednak przed złożeniem pracy **należy otworzyć
> adresy podane w bibliografii i potwierdzić aktualne brzmienie cytowanych klauzul**, a datę
> dostępu zaktualizować. Tam, gdzie dokładnego brzmienia nie udało się ustalić, zaznaczono to
> wprost, zamiast rekonstruować treść postanowienia.

---

## 1. Źródło danych i podstawa prawna wykorzystania

### 1.1. Skąd pochodzą dane

Wszystkie dane rynkowe wykorzystane w projekcie pochodzą z jednego źródła — serwisu
**Yahoo Finance** (Yahoo Inc.). Dane pobierane są automatycznie, przy użyciu otwartoźródłowej
biblioteki języka Python **`yfinance`** (autor: Ran Aroussi, repozytorium
`github.com/ranaroussi/yfinance`), zadeklarowanej w plikach `stock_ml/requirements.txt`
(`yfinance>=0.2.31`) oraz `dashboard/requirements.txt`.

Pobieranie realizują trzy moduły:

| Plik | Rola | Zapisywany artefakt |
|---|---|---|
| `stock_ml/data/download.py` | pobranie i cache danych treningowych (`auto_adjust=True`, kolumny OHLCV) | `data/raw/*.parquet`, `*.csv` — katalog wyłączony z repozytorium (`.gitignore`) |
| `stock_ml/data/download_historical.py` | kopia zapasowa pełnej historii (`auto_adjust=False`, OHLCV + `Adj Close`) | `stock_ml/data/historical/{TICKER}_historical.csv` — **commitowane do repozytorium** |
| `dashboard/lib/data_fetcher.py` | doraźne pobranie serii na potrzeby generowania widoków dashboardu | dane w pamięci, dalej przetwarzane do `docs/data/` |

### 1.2. Zakres pobieranych danych

* **Interwał:** dzienny (`1d`).
* **Zakres czasowy:** od `2011-04-01` (`DATA_START`) do `2026-04-01` (`DATA_END`).
* **Instrumenty (11):** AAPL, MSFT, JPM, XOM, JNJ, UNH, PG, WMT, KO, NEE oraz SPY
  (ETF na indeks S&P 500, wykorzystywany wyłącznie jako benchmark strategii „kup i trzymaj”).
* **Pola:** `Open`, `High`, `Low`, `Close`, `Volume`, a w kopii historycznej dodatkowo
  `Adj Close`.

### 1.3. Charakter prawny dostępu — kwestia kluczowa

Należy jednoznacznie odnotować fakt istotny dla całej dalszej analizy:

> **Yahoo Finance nie udostępnia obecnie publicznego, udokumentowanego i licencjonowanego API
> do danych giełdowych dla użytkowników zewnętrznych.** Historyczne API Yahoo Finance
> (oparte m.in. o YQL) zostało wycofane w 2017 r. Biblioteka `yfinance` korzysta z
> **nieudokumentowanych, wewnętrznych punktów końcowych** serwisu Yahoo Finance,
> wykorzystywanych przez jego interfejs webowy.

Wynikają z tego dwie konsekwencje:

1. **Brak umowy licencyjnej na dane.** Użytkownik `yfinance` nie zawiera z Yahoo żadnej
   odrębnej umowy licencyjnej dotyczącej danych. Jedyną podstawą relacji są **ogólne warunki
   korzystania z usług Yahoo (Terms of Service)**, akceptowane w sposób dorozumiany przez
   korzystanie z serwisu.
2. **Brak gwarancji ciągłości.** Yahoo może w każdej chwili zmienić lub zamknąć wykorzystywane
   punkty końcowe — co w praktyce wielokrotnie miało miejsce i wymuszało kolejne wydania
   `yfinance`. Właśnie z tego powodu projekt utrzymuje commitowaną kopię historyczną
   (`stock_ml/data/historical/`), co jednak — jak omówiono w rozdz. 5 — samo w sobie rodzi
   odrębne ryzyko licencyjne.

**Rozróżnienie fundamentalne dla dalszych rozważań:** licencja **biblioteki** (`yfinance`) i
licencja **danych** (Yahoo Finance) to dwie całkowicie odrębne kwestie. Otwartoźródłowa,
liberalna licencja narzędzia **nie przenosi** żadnych praw do danych, które za jego pomocą
pobrano. Jest to najczęstsze nieporozumienie w pracach dyplomowych opartych o `yfinance`.

---

## 2. Licencja biblioteki `yfinance`

### 2.1. Licencja

Biblioteka `yfinance` jest rozpowszechniana na **licencji Apache License, Version 2.0**
(plik `LICENSE.txt` w repozytorium `ranaroussi/yfinance`; klasyfikator PyPI:
*License :: OSI Approved :: Apache Software License*). **[do weryfikacji — potwierdzić w pliku
`LICENSE.txt` aktualnego wydania biblioteki zainstalowanego w projekcie]**

Apache 2.0 to licencja permisywna. W zakresie **kodu biblioteki** zezwala ona na:

* korzystanie, zwielokrotnianie i modyfikowanie kodu, w tym w celach komercyjnych,
* rozpowszechnianie kodu oryginalnego i pochodnego,

pod warunkiem m.in.:

* zachowania informacji o prawach autorskich i treści licencji,
* oznaczenia wprowadzonych zmian w plikach zmodyfikowanych,
* zachowania pliku `NOTICE`, jeżeli występuje.

Licencja zawiera też wyraźne wyłączenie rękojmi i odpowiedzialności (sekcje 7 i 8 Apache 2.0:
oprogramowanie dostarczane jest „AS IS”, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND).

**Wykorzystanie `yfinance` w niniejszej pracy mieści się bez zastrzeżeń w granicach licencji
Apache 2.0** — biblioteka jest jedynie importowana i wywoływana, nie jest modyfikowana ani
redystrybuowana w postaci zmienionej.

### 2.2. Disclaimer autora biblioteki

Autor biblioteki wprost zastrzega, że `yfinance` nie jest oficjalnym API Yahoo, i przenosi na
użytkownika ciężar zapoznania się z warunkami korzystania z samych danych. Sekcja
*„Important Legal Disclaimer”* w pliku `README.md` repozytorium `ranaroussi/yfinance` stanowi
w szczególności **[cytat wg znanego brzmienia README — przed złożeniem pracy potwierdzić
dokładne sformułowanie w aktualnej wersji pliku]**:

> „**Yahoo!, Y!Finance, and Yahoo! finance are registered trademarks of Yahoo, Inc.**
>
> yfinance is **not affiliated, endorsed, or vetted by Yahoo, Inc.** It's an open-source tool
> that uses Yahoo's publicly available APIs, and is intended for **research and educational
> purposes**.
>
> **You should refer to Yahoo!'s terms of use** […] **for details on your rights to use the
> actual data downloaded. Remember - the Yahoo! finance API is intended for personal use
> only.**”

Tłumaczenie robocze:

> „Yahoo!, Y!Finance oraz Yahoo! finance są zastrzeżonymi znakami towarowymi Yahoo, Inc.
> yfinance nie jest powiązany z Yahoo, Inc., nie jest przez nią wspierany ani zweryfikowany.
> Jest to narzędzie otwartoźródłowe korzystające z publicznie dostępnych API Yahoo,
> przeznaczone do celów badawczych i edukacyjnych. Należy zapoznać się z warunkami korzystania
> Yahoo! […] w celu ustalenia zakresu przysługujących uprawnień do korzystania z samych
> pobranych danych. Należy pamiętać, że API Yahoo! finance przeznaczone jest wyłącznie do
> użytku osobistego.”

### 2.3. Wnioski z disclaimera

Disclaimer jest dla niniejszej pracy istotny z trzech powodów:

1. **Potwierdza dopuszczalność celu badawczo-edukacyjnego** — autor narzędzia wskazuje
   dokładnie taki jego zamierzony sposób użycia, jaki realizuje ta praca.
2. **Nie udziela żadnych praw do danych** — autor `yfinance` nie jest dysponentem danych
   Yahoo i nie mógłby ich udzielić. Licencja Apache 2.0 obejmuje **wyłącznie kod**.
3. **Zawiera ostrzeżenie o użytku osobistym** (*„intended for personal use only”*), które
   stanowi zasadniczy argument przeciwko publicznej redystrybucji danych surowych — zob.
   rozdz. 5.

---

## 3. Regulamin Yahoo Finance

### 3.1. Właściwe dokumenty

Korzystanie z Yahoo Finance regulują w szczególności:

* **Yahoo Terms of Service (ATOS)** — ogólne warunki korzystania z usług Yahoo:
  <https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html>
* **Yahoo APIs Terms of Use (Yahoo Developer Network)** — warunki korzystania z interfejsów
  programistycznych:
  <https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm>
* **Yahoo Privacy / Policies hub** — <https://policies.yahoo.com/us/en/yahoo/terms/index.htm>

Są to dokładnie te trzy adresy, do których odsyła disclaimer w README biblioteki `yfinance`.
**[do weryfikacji — adresy Yahoo bywają przekierowywane; potwierdzić docelowy URL i datę
ostatniej aktualizacji dokumentu]**

### 3.2. Czego dotyczą ograniczenia — ocena ostrożna

Poniższy opis oddaje **ogólny charakter** postanowień typowych dla warunków korzystania z
usług Yahoo. **Nie przytaczam dosłownego brzmienia klauzul, ponieważ w toku przygotowania
niniejszego opracowania nie było możliwe bezpośrednie pobranie i zacytowanie aktualnej treści
regulaminu.** Dosłowne cytaty należy uzupełnić po samodzielnym otwarciu powyższych adresów.

Regulaminy Yahoo — zgodnie z ich znanym, wieloletnim brzmieniem — obejmują w szczególności
następujące grupy ograniczeń:

**a) Dostęp zautomatyzowany (scraping).**
Warunki korzystania z usług Yahoo zawierają postanowienia zakazujące korzystania z usług przy
użyciu zautomatyzowanych środków (robotów, „spiderów”, scraperów) bez uprzedniej zgody
usługodawcy. Oznacza to, że **automatyczne pobieranie danych przez `yfinance` — czyli sposób,
w jaki działa cały pipeline tego projektu — należy traktować co najmniej jako obszar sporny na
gruncie regulaminu.** Jest to ocena ostrożnościowa; nie ma znanego autorowi publicznego
stanowiska Yahoo kwalifikującego wprost użycie `yfinance` do celów naukowych.

**b) Cel niekomercyjny / użytek osobisty.**
Zarówno disclaimer `yfinance` (rozdz. 2.2), jak i warunki korzystania z API Yahoo wskazują na
**użytek osobisty i niekomercyjny** jako zamierzony zakres korzystania. Wykorzystanie danych do
pracy dyplomowej i badań własnych mieści się w tak zakreślonym celu **wyraźnie lepiej** niż
jakiekolwiek zastosowanie zarobkowe.

**c) Redystrybucja danych.**
To postanowienie ma dla niniejszego projektu znaczenie rozstrzygające. Warunki korzystania z
usług i API Yahoo **ograniczają dalsze rozpowszechnianie treści uzyskanych z serwisu**
— w typowym brzmieniu poprzez zakaz zwielokrotniania, rozpowszechniania, publicznego
udostępniania, odsprzedaży lub tworzenia usług konkurencyjnych w oparciu o pozyskane treści,
bez uprzedniej pisemnej zgody. **[do weryfikacji — dosłowne brzmienie i numer klauzuli]**

**Czego nie udało się ustalić i czego nie należy twierdzić w pracy:**

* dokładnego brzmienia i numeracji klauzul dotyczących redystrybucji w aktualnej wersji ATOS;
* czy w aktualnych warunkach występuje limit czasowy przechowywania (cache) pobranych treści
  — w regulaminach różnych API pojawiają się klauzule typu „nie dłużej niż 24 godziny”, jednak
  **nie należy przypisywać takiego postanowienia Yahoo bez jego zweryfikowania**;
* czy Yahoo formułuje odrębny, złagodzony reżim dla zastosowań akademickich (autorowi nie jest
  znane istnienie takiego reżimu).

### 3.3. Podsumowanie stanu prawnego dostępu

| Zagadnienie | Ocena |
|---|---|
| Pobieranie danych do analizy własnej | Dopuszczalne pod względem celu (badawczo-edukacyjny); sposób (automatyzacja) pozostaje obszarem spornym na gruncie regulaminu |
| Przechowywanie lokalnej kopii do celów odtwarzalności badania | Uzasadnione metodologicznie; ryzyko regulaminowe niskie, dopóki kopia nie jest publiczna |
| **Publiczne udostępnianie danych surowych** | **Obszar najwyższego ryzyka — zob. rozdz. 5** |
| Publikowanie danych pochodnych (sygnały, metryki, wykresy) | Ryzyko istotnie niższe — zob. rozdz. 5.3 |
| Wykorzystanie komercyjne | Poza zakresem projektu; wymagałoby odrębnej licencji |

---

## 4. Zakres wykorzystania w niniejszej pracy

Autor niniejszej pracy oświadcza, że:

1. **Cel wyłącznie naukowo-dydaktyczny.** Dane rynkowe pobrane z Yahoo Finance za pomocą
   biblioteki `yfinance` wykorzystywane są wyłącznie na potrzeby pracy inżynierskiej
   realizowanej na Uniwersytecie Przyrodniczym w Poznaniu — do trenowania, walidacji i
   ewaluacji modeli uczenia maszynowego oraz do prezentacji wyników tych eksperymentów.

2. **Brak jakiejkolwiek komercjalizacji.** Projekt „Stock4caster” nie jest i nie będzie
   przedmiotem sprzedaży, odpłatnego udostępniania, subskrypcji ani innej formy monetyzacji.
   Dashboard nie zawiera reklam, mechanizmów płatności ani zbierania danych użytkowników.
   Kod i wyniki udostępniane są nieodpłatnie w celach edukacyjnych.

3. **Brak konkurencji wobec źródła.** Publikowane materiały nie są i nie mają być substytutem
   serwisu Yahoo Finance ani komercyjnego dostawcy danych rynkowych. Nie są udostępniane dane
   w czasie rzeczywistym, nie jest oferowane API, nie jest budowana usługa dystrybucji danych.

4. **Brak porady inwestycyjnej.** Prezentowane sygnały, prognozy, metryki i wyniki
   backtestów mają charakter wyłącznie **badawczy i poglądowy**. Nie stanowią rekomendacji
   inwestycyjnej ani doradztwa inwestycyjnego w rozumieniu przepisów o obrocie instrumentami
   finansowymi (w szczególności ustawy z dnia 29 lipca 2005 r. o obrocie instrumentami
   finansowymi oraz rozporządzenia MAR), ani oferty nabycia bądź zbycia jakiegokolwiek
   instrumentu finansowego. Wyniki historyczne i wyniki backtestów **nie stanowią gwarancji
   wyników przyszłych**. Autor nie ponosi odpowiedzialności za decyzje podjęte na podstawie
   prezentowanych treści.

5. **Atrybucja źródła.** Yahoo, Yahoo! Finance i powiązane oznaczenia są znakami towarowymi
   Yahoo Inc. Projekt nie jest powiązany z Yahoo Inc. ani przez nią wspierany, zatwierdzony
   czy zweryfikowany. Nie jest też powiązany z autorem biblioteki `yfinance`.

---

## 5. Status prawny notowań giełdowych — fakty a ochrona baz danych

Poniższa analiza ma charakter ostrożny i uproszczony; jej celem jest wskazanie, **dlaczego
sama „faktyczność” kursów nie zamyka sprawy**.

### 5.1. Pojedynczy fakt rynkowy nie jest chroniony prawem autorskim

Prawo autorskie chroni **sposób wyrażenia**, nie zaś fakty, idee, odkrycia czy dane jako takie
(zob. art. 1 ust. 2¹ ustawy z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych;
analogicznie art. 9 ust. 2 Porozumienia TRIPS oraz art. 2 Traktatu WIPO o prawie autorskim).
Informacja, że kurs zamknięcia akcji AAPL w danym dniu wyniósł określoną wartość, jest
**faktem rynkowym** i jako pojedyncza informacja nie jest przedmiotem prawa autorskiego.

### 5.2. Zbiór danych może jednak podlegać ochronie sui generis

Odrębny reżim ustanawia **dyrektywa 96/9/WE Parlamentu Europejskiego i Rady z dnia 11 marca
1996 r. w sprawie ochrony prawnej baz danych**, wdrożona w Polsce **ustawą z dnia 27 lipca
2001 r. o ochronie baz danych** (Dz.U. 2001 nr 128 poz. 1402 ze zm.). Prawo *sui generis*
przysługuje producentowi bazy danych, który wykazał **istotny co do jakości lub ilości nakład
inwestycyjny w sporządzenie, weryfikację lub prezentację** zawartości bazy, i pozwala mu
zakazać pobierania lub wtórnego wykorzystania istotnej części bazy (a także powtarzającego się
pobierania części nieistotnych, sprzecznego z normalnym korzystaniem z bazy).

Dla oceny sytuacji istotne są dwa zastrzeżenia:

* **Nakład na *wytworzenie* danych nie jest nakładem na ich *pozyskanie*.** Trybunał
  Sprawiedliwości UE w wyrokach z 9 listopada 2004 r. — *The British Horseracing Board v.
  William Hill* (C-203/02) oraz w sprawach *Fixtures Marketing* (C-46/02, C-338/02, C-444/02)
  — przesądził, że środki przeznaczone na **tworzenie** danych będących zawartością bazy nie
  są uwzględniane przy ocenie istotności nakładu. Jest to argument osłabiający ochronę
  *sui generis* po stronie podmiotu, który dane **generuje** (np. giełdy w toku obrotu).
* **Nie osłabia to jednak pozycji agregatora.** Podmiot, który dane z wielu rynków **zbiera,
  weryfikuje, normalizuje i prezentuje** (a taką rolę pełni serwis finansowy), może wykazać
  nakład kwalifikowany. Wykluczenie ochrony *sui generis* dla Yahoo Finance nie jest zatem
  wnioskiem oczywistym i nie należy go w pracy zakładać.

### 5.3. Wyjątek na eksplorację tekstów i danych (TDM)

Dyrektywa (UE) 2019/790 (DSM) wprowadziła w **art. 3** obowiązkowy wyjątek na eksplorację
tekstów i danych **prowadzoną przez organizacje badawcze w celach badań naukowych**, którego
nie można wyłączyć umownie, oraz w **art. 4** wyjątek ogólny z możliwością zastrzeżenia
uprawnionego. Przepisy te zostały wdrożone do polskiego porządku prawnego nowelizacją prawa
autorskiego z 2024 r. **[do weryfikacji — dokładna jednostka redakcyjna, tj. numeracja
artykułów w znowelizowanej ustawie]**

Znaczenie tego wyjątku dla projektu jest realne, ale **wąskie**:

* **przemawia za dopuszczalnością** zwielokrotniania danych na potrzeby ich analizy
  (trenowanie modeli, backtesting) w kontekście badań prowadzonych w ramach uczelni;
* **nie legalizuje natomiast publicznego udostępniania samego korpusu danych.** Wyjątek TDM
  obejmuje zwielokrotnianie i bezpieczne przechowywanie na potrzeby eksploracji, nie zaś
  rozpowszechnianie zbioru wśród nieoznaczonego kręgu odbiorców.

### 5.4. Wniosek: rzeczywistym ograniczeniem jest umowa, nie prawo autorskie

Nawet gdyby przyjąć, że pojedyncze notowania nie są chronione prawem autorskim, a ochrona
*sui generis* jest wątpliwa, **pozostaje wiążący stosunek umowny** wynikający z akceptacji
warunków korzystania z serwisu Yahoo (rozdz. 3). Naruszenie regulaminu jest kwestią
**odpowiedzialności kontraktowej**, niezależną od tego, czy dane są przedmiotem prawa
własności intelektualnej. **To właśnie regulamin, a nie prawo autorskie, jest w tym projekcie
głównym źródłem ryzyka.**

---

## 6. Ryzyko redystrybucji i przyjęte środki ostrożności

### 6.1. Czy publikacja danych w repozytorium to redystrybucja?

**Tak — i jest to ocena, przed którą projekt nie powinien się uchylać.**

Repozytorium GitHub jest publiczne, a dashboard opublikowany na GitHub Pages jest dostępny
bez uwierzytelnienia dla dowolnego użytkownika sieci. Umieszczenie w nim plików z danymi
oznacza **publiczne udostępnianie utrwalonej kopii treści pozyskanych z Yahoo Finance** —
czyli redystrybucję, a nie „użytek osobisty”, o którym mowa w disclaimerze `yfinance`.
Nie zmienia tego okoliczność, że publikacja jest nieodpłatna i motywowana celem naukowym:
kryterium regulaminowym jest **udostępnienie treści osobom trzecim**, a nie osiągnięcie zysku.

### 6.2. Które artefakty projektu są problematyczne

| Artefakt | Charakter | Ocena ryzyka |
|---|---|---|
| `stock_ml/data/historical/{TICKER}_historical.csv` | **pełna surowa historia OHLCV + Adj Close, ~15 lat, 11 instrumentów, commitowana w publicznym repo** | **Wysokie.** To najbliższy odpowiednik „kopii bazy danych źródłowej”. Format CSV czyni ją bezpośrednio nadającą się do ponownego użycia — a więc potencjalnie substytucyjną wobec źródła |
| `docs/data/{TICKER}_ohlcv.json` oraz `docs/data/tickers/{TICKER}/ohlcv.json` | surowe szeregi OHLCV serwowane przez GitHub Pages | **Średnie–wysokie.** Zależy od długości publikowanego okna; im dłuższy szereg, tym bliżej „istotnej części” zbioru |
| `docs/data/exports/*.csv` (`all_signals`, `all_transactions`, `portfolio_metrics`, `model_metrics`) | **dane pochodne** — wyniki modeli, transakcje, metryki | **Niskie.** Produkt własnej pracy twórczej i obliczeniowej autora; nie odtwarzają danych źródłowych i nie zastępują ich |
| Wykresy PNG z pipeline’u ML | **dane pochodne** — wizualizacje wyników | **Niskie.** Prezentacja graficzna wyników badania, nie zbiór danych |
| Kod źródłowy pipeline’u | własny utwór autora | **Brak.** Nie zawiera danych Yahoo |

**Kluczowa granica przebiega między danymi surowymi a pochodnymi.** Sygnały, metryki modeli i
wykresy są rezultatem istotnego przetworzenia — nie pozwalają odtworzyć źródłowych szeregów i
nie konkurują z serwisem źródłowym. Publikowanie ich jest nieporównanie mniej ryzykowne niż
publikowanie kompletnych plików OHLCV.

### 6.3. Rzeczywista skala ryzyka — ocena uczciwa

Nie należy tego ryzyka ani wyolbrzymiać, ani bagatelizować.

**Argumenty za niskim prawdopodobieństwem materializacji ryzyka:**

* skala projektu jest znikoma (11 instrumentów, dane dzienne, brak danych intraday i
  czasu rzeczywistego);
* dane dzienne EOD są powszechnie dostępne z wielu źródeł i mają niewielką wartość rynkową;
* cel jest wyłącznie akademicki, brak jakiejkolwiek monetyzacji;
* w serwisie GitHub istnieją tysiące repozytoriów zawierających analogiczne zbiory i autorowi
  nie są znane przypadki egzekwowania roszczeń wobec projektów studenckich tej skali.

**Argumenty za istnieniem realnego ryzyka:**

* niskie prawdopodobieństwo to nie to samo co zgodność z regulaminem — **projekt opisywany w
  pracy dyplomowej powinien być zgodny z zasadami, a nie jedynie statystycznie bezpieczny**;
* najbardziej prawdopodobne konsekwencje to nie proces sądowy, lecz **żądanie usunięcia treści
  (notice-and-takedown) skierowane do GitHub** albo **zablokowanie dostępu (IP/rate limiting)
  przez Yahoo** — co w praktyce oznaczałoby awarię działającego dashboardu, potencjalnie tuż
  przed obroną;
* promotor lub recenzent może zasadnie zakwestionować podstawę publikacji danych, jeżeli praca
  nie zawiera świadomej analizy tego zagadnienia — **posiadanie niniejszego rozdziału jest
  samo w sobie środkiem zaradczym**.

### 6.4. Przyjęte i rekomendowane środki ostrożności

**Środki przyjęte:**

1. **Ograniczenie do danych dziennych (EOD).** Projekt nie pobiera i nie publikuje danych
   w czasie rzeczywistym ani danych intraday, których wartość rynkowa i ochrona umowna są
   nieporównanie wyższe.
2. **Minimalizacja zakresu.** Zbiór ograniczono do 11 instrumentów dobranych metodologicznie
   (dywersyfikacja sektorowa + benchmark), a nie do możliwie największej próby.
3. **Rozdzielenie warstw.** Katalog roboczy `data/raw/` jest wyłączony z repozytorium
   (`.gitignore`); publikowane są przede wszystkim artefakty pochodne w `docs/data/`.
4. **Jawna atrybucja i disclaimer.** Niniejszy dokument oraz sekcja „O stronie” dashboardu
   wskazują źródło danych, brak powiązania z Yahoo Inc. i brak charakteru rekomendacji
   inwestycyjnej.
5. **Udokumentowanie analizy prawnej.** Niniejszy rozdział stanowi dowód świadomego
   rozważenia zagadnienia, zgodnie z wymaganiem promotora.

**Rekomendacje — działania zalecane przed publikacją/obroną, w kolejności priorytetu:**

1. **Rozważyć usunięcie z repozytorium pełnych plików
   `stock_ml/data/historical/{TICKER}_historical.csv`** i zastąpienie ich **skryptem
   odtwarzającym** dane (`download_historical.py` już istnieje) wraz z zapisem sumy
   kontrolnej (hash) oraz dokładnej daty i wersji `yfinance` użytych do pobrania.
   Odtwarzalność badania zostaje wówczas zachowana — recenzent może wygenerować identyczny
   zbiór — a repozytorium przestaje redystrybuować dane źródłowe. **To pojedyncza zmiana o
   największym efekcie i najniższym koszcie.**
   *(Uwaga techniczna: samo usunięcie plików z bieżącego commitu nie usuwa ich z historii Gita;
   pełne usunięcie wymaga przepisania historii, np. `git filter-repo`. Decyzję o tym należy
   podjąć świadomie, ważąc korzyść wobec ryzyka naruszenia integralności repozytorium pracy.)*
2. **Ograniczyć okno publikowanych danych surowych** w `docs/data/*_ohlcv.json` do zakresu
   rzeczywiście niezbędnego do działania wykresów dashboardu (np. 1–3 lata zamiast pełnych
   15 lat). Publikowany fragment jest wtedy trudniejszy do zakwalifikowania jako „istotna
   część” zbioru w rozumieniu ochrony *sui generis* i wyraźnie niesubstytucyjny wobec źródła.
3. **Umieścić widoczną notę atrybucyjną i disclaimer** na stronie dashboardu oraz w pliku
   `README.md` repozytorium (gotowy tekst — zob. rozdz. 8).
4. **Dodać plik `NOTICE`** lub sekcję w `README.md` z informacją o wykorzystaniu `yfinance`
   na licencji Apache 2.0 wraz z odesłaniem do jej treści.
5. **W dalszych pracach rozważyć migrację do źródła o jaśniejszej licencji** (rozdz. 7),
   ewentualnie w modelu równoległym: Yahoo/`yfinance` jako źródło robocze, a źródło o jasnej
   licencji jako to, z którego dane są faktycznie publikowane.

### 6.5. Praktyka w pracach dyplomowych

Ustalenia dotyczące utrwalonej praktyki należy formułować ostrożnie, ponieważ nie istnieje
jeden wiążący standard. Na podstawie obserwacji publicznie dostępnych prac i repozytoriów
akademickich można jednak wskazać powtarzalny wzorzec:

* **Yahoo Finance i `yfinance` są w pracach dyplomowych wykorzystywane bardzo powszechnie** —
  ze względu na brak opłat, brak konieczności rejestracji i wygodę API. Samo wykorzystanie
  tego źródła do analiz nie jest w środowisku akademickim kwestionowane.
* **Standardem jest wykorzystanie lokalne z podaniem źródła**, a nie republikacja zbioru.
  Prace zwykle opisują sposób pozyskania danych i podają parametry umożliwiające ich
  odtworzenie, zamiast dołączać kompletny zbiór jako załącznik publiczny.
* **W publikacjach udostępnia się przede wszystkim wyniki**: tabele metryk, wykresy, sygnały,
  kod — czyli dokładnie te artefakty, które w tym projekcie zaklasyfikowano jako pochodne.
* **Rosnącym standardem jest sekcja „Data availability statement”**, wskazująca źródło,
  sposób i datę pozyskania danych oraz ograniczenia w ich udostępnianiu. Niniejszy dokument
  pełni tę funkcję.

Wniosek: projekt mieści się w akceptowanej praktyce akademickiej **w zakresie pozyskiwania i
analizy** danych. Elementem odbiegającym od typowej praktyki jest **commitowanie kompletnych
surowych plików CSV w publicznym repozytorium** — i to właśnie ten element wskazano wyżej jako
priorytet do korekty.

---

## 7. Alternatywne źródła danych o jaśniejszej licencji

> **[do weryfikacji]** Poniższe zestawienie opisuje modele licencyjne zgodnie z ich znanym
> kształtem. **Warunki dostawców danych zmieniają się często**, a przed przyjęciem
> któregokolwiek źródła jako podstawy publikacji należy bezwzględnie przeczytać jego aktualny
> regulamin i — w razie potrzeby — wystąpić o potwierdzenie na piśmie (dostawcy zwykle
> odpowiadają na zapytania akademickie).

| Źródło | Model dostępu | Publikacja danych **surowych** | Publikacja danych **pochodnych** | Uwagi dla studenta |
|---|---|---|---|---|
| **Yahoo Finance** (przez `yfinance`) — *stan obecny* | Nieoficjalne, nieudokumentowane endpointy; bez klucza | **Nie zalecane** — regulamin ogranicza dalsze rozpowszechnianie; disclaimer wskazuje na użytek osobisty | Ryzyko niskie | Wygodne i darmowe, ale bez podstawy licencyjnej do redystrybucji |
| **Stooq** (stooq.pl / stooq.com) | Darmowe pobieranie CSV po URL, bez klucza; dane GPW i rynków zagranicznych | **Wymaga sprawdzenia regulaminu** — serwis udostępnia dane do użytku własnego; zakres zgody na redystrybucję nie jest oczywisty | Ryzyko niskie | Bardzo dobra opcja robocza dla pracy w Polsce: polskie źródło, łatwe do zacytowania, prosty format CSV, brak rejestracji. **Przed publikacją danych surowych napisać do serwisu z zapytaniem** |
| **Nasdaq Data Link** (dawniej Quandl) | Darmowy klucz API + zbiory płatne; część zbiorów na **jawnych licencjach otwartych** | **Zależy od zbioru** — poszczególne zbiory mają własne, wyraźnie wskazane licencje. Historyczny zbiór *WIKI Prices* był udostępniony na **CC BY 4.0** (redystrybucja dozwolona przy podaniu źródła), lecz **przestał być aktualizowany w 2018 r.** | Dozwolone zgodnie z licencją zbioru | **Jedyne źródło z tej listy dające jednoznaczną, pisemną zgodę na redystrybucję** — dla zbiorów objętych CC BY. Wada: brak aktualnych notowań w zbiorach otwartych |
| **Alpha Vantage** | Darmowy klucz API, ostry limit zapytań w planie bezpłatnym | **Nie** w planie bezpłatnym — regulamin zastrzega redystrybucję; wymaga planu komercyjnego/zgody | Zwykle dozwolone | Dobre do pozyskiwania danych, nie do ich publikowania. Limity dzienne bywają uciążliwe dla pełnej historii 11 instrumentów |
| **Tiingo** | Darmowy klucz API dla użytku osobistego; plany płatne | **Nie** — redystrybucja wymaga odrębnej licencji komercyjnej | Zwykle dozwolone | Dobra jakość danych EOD, przyjazny dla projektów niekomercyjnych; publikacja surowych szeregów wykluczona bez licencji |
| **EODHD** (EOD Historical Data) | Płatna subskrypcja (dostępne taryfy studenckie/tanie) | **Nie** w planach podstawowych — redystrybucja objęta odrębną, droższą licencją | Zwykle dozwolone | Wygodne API i szeroki zakres; koszt i zakaz redystrybucji ograniczają przydatność dla tego projektu |
| **SEC EDGAR** (sec.gov) | Otwarty, oficjalny; bez opłat | **Tak** — materiały urzędowe rządu USA, w domenie publicznej | Tak | **Wzorzec źródła bez ograniczeń licencyjnych.** Zawiera jednak dane sprawozdawcze (fundamentalne), **nie notowania giełdowe** — nie zastąpi OHLCV, ale doskonale nadaje się jako uzupełniające źródło o bezspornym statusie |

### 7.1. Rekomendacja dla studenta — dwie realne opcje

**Opcja A (rekomendowana — najmniejszy koszt, największy efekt): pozostać przy `yfinance`,
ale przestać publikować dane surowe.**

Zachować obecny pipeline bez zmian, natomiast:

* usunąć z publicznego repozytorium commitowane pliki `*_historical.csv`, zastępując je
  skryptem odtwarzającym wraz z metadanymi pobrania (data, wersja `yfinance`, suma kontrolna);
* ograniczyć okno danych w `docs/data/*_ohlcv.json` do minimum niezbędnego dla wykresów;
* publikować bez zmian wszystkie artefakty pochodne (sygnały, metryki, transakcje, wykresy).

Rozwiązanie to **eliminuje główne ryzyko przy zerowej ingerencji w metodykę badawczą i bez
konieczności ponownego przeliczania wyników**. Odtwarzalność pracy zostaje zachowana.

**Opcja B (jeżeli publikacja pełnych szeregów surowych jest wymagana):**

* **na potrzeby części historycznej** — wykorzystać zbiór z **Nasdaq Data Link** objęty
  licencją **CC BY 4.0**, który wprost dopuszcza redystrybucję przy podaniu źródła (ograniczenie:
  brak danych po 2018 r., co pozwala pokryć jedynie wcześniejszą część zakresu 2011–2026);
* **na potrzeby części bieżącej** — wykorzystać **Stooq**, po uprzednim potwierdzeniu zakresu
  dozwolonego wykorzystania w regulaminie serwisu lub w drodze zapytania e-mail (odpowiedź
  warto załączyć do dokumentacji pracy).

Opcja B jest solidniejsza licencyjnie, ale wymaga migracji warstwy pobierania danych,
ponownego przeliczenia całego pipeline’u i weryfikacji zgodności wyników — jest to zauważalny
nakład pracy tuż przed obroną. **Jeżeli czas jest ograniczony, należy wybrać Opcję A.**

---

## 8. Bibliografia i źródła

Data dostępu do wszystkich źródeł: **26 sierpnia 2026 r.**

> **Uwaga metodyczna.** Wskazane niżej adresy należy otworzyć samodzielnie i potwierdzić
> aktualność cytowanych treści przed złożeniem pracy. Fragmenty oznaczone w tekście jako
> **[do weryfikacji]** wymagają uzupełnienia o dosłowne brzmienie klauzul.

### Źródła pierwotne — licencje i regulaminy

1. **yfinance — repozytorium projektu**, R. Aroussi, <https://github.com/ranaroussi/yfinance>
2. **yfinance — plik licencji (Apache License 2.0)**,
   <https://github.com/ranaroussi/yfinance/blob/main/LICENSE.txt>
3. **yfinance — README, sekcja „Important Legal Disclaimer”**,
   <https://github.com/ranaroussi/yfinance/blob/main/README.md>
4. **yfinance — dokumentacja**, <https://ranaroussi.github.io/yfinance/>
5. **yfinance — pakiet w PyPI**, <https://pypi.org/project/yfinance/>
6. **Apache License, Version 2.0**, Apache Software Foundation,
   <https://www.apache.org/licenses/LICENSE-2.0>
7. **Yahoo Terms of Service (ATOS)**,
   <https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html>
8. **Yahoo APIs Terms of Use (Yahoo Developer Network)**,
   <https://policies.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.htm>
9. **Yahoo — zbiór regulaminów i polityk**,
   <https://policies.yahoo.com/us/en/yahoo/terms/index.htm>

### Akty prawne

10. **Dyrektywa 96/9/WE** Parlamentu Europejskiego i Rady z dnia 11 marca 1996 r. w sprawie
    ochrony prawnej baz danych, <https://eur-lex.europa.eu/eli/dir/1996/9/oj>
11. **Dyrektywa (UE) 2019/790** (DSM) — w szczególności art. 3 i 4 (eksploracja tekstów
    i danych), <https://eur-lex.europa.eu/eli/dir/2019/790/oj>
12. **Ustawa z dnia 27 lipca 2001 r. o ochronie baz danych**, Dz.U. 2001 nr 128 poz. 1402
    ze zm., <https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20011281402>
13. **Ustawa z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych**,
    tekst jednolity, <https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU19940240083>

### Orzecznictwo

14. Wyrok TSUE z 9 listopada 2004 r., **C-203/02**, *The British Horseracing Board Ltd i in.
    przeciwko William Hill Organization Ltd* — nakład na wytworzenie danych a nakład na ich
    pozyskanie.
15. Wyroki TSUE z 9 listopada 2004 r. w sprawach **C-46/02, C-338/02, C-444/02**
    (*Fixtures Marketing*) — przesłanki ochrony *sui generis* baz danych.

### Alternatywne źródła danych

16. **Stooq** — <https://stooq.pl/> ; <https://stooq.com/db/>
17. **Nasdaq Data Link** (dawniej Quandl) — <https://data.nasdaq.com/>
18. **Alpha Vantage** — <https://www.alphavantage.co/> ; regulamin:
    <https://www.alphavantage.co/terms_of_service/>
19. **Tiingo** — <https://www.tiingo.com/> ; <https://www.tiingo.com/about/terms>
20. **EOD Historical Data (EODHD)** — <https://eodhd.com/>
21. **SEC EDGAR** — <https://www.sec.gov/edgar> ; zasady dostępu automatycznego:
    <https://www.sec.gov/os/webmaster-faq#developers>
22. **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
    <https://creativecommons.org/licenses/by/4.0/legalcode.pl>

---

## Skrót do wykorzystania na stronie WWW

### Wersja polska

Dane rynkowe prezentowane w serwisie Stock4caster pochodzą z serwisu Yahoo Finance i zostały
pobrane przy użyciu otwartoźródłowej biblioteki `yfinance` (licencja Apache 2.0), która nie
jest oficjalnym API Yahoo i nie jest z Yahoo Inc. powiązana. Projekt ma charakter wyłącznie
naukowo-dydaktyczny — powstał jako część pracy inżynierskiej i nie jest w żaden sposób
komercjalizowany. Prezentowane sygnały, prognozy i wyniki backtestów służą wyłącznie celom
badawczym i poglądowym; **nie stanowią rekomendacji ani doradztwa inwestycyjnego**, a wyniki
historyczne nie są gwarancją wyników przyszłych. Yahoo i Yahoo! Finance są znakami towarowymi
Yahoo Inc.; niniejszy serwis nie jest przez tę spółkę wspierany ani zatwierdzony. Szczegółowe
omówienie zasad i licencji obowiązujących przy pozyskiwaniu oraz udostępnianiu danych zawiera
dokument [LICENSING.md](LICENSING.md).

### English version

The market data presented on Stock4caster is sourced from Yahoo Finance and retrieved using
the open-source `yfinance` library (Apache 2.0 licence), which is not an official Yahoo API
and is not affiliated with Yahoo Inc. This project is strictly academic and educational — it
was created as part of an engineering thesis and is not commercialised in any way. The
signals, forecasts and backtest results shown here are for research and illustrative purposes
only; **they do not constitute investment advice or a recommendation**, and past performance
is no guarantee of future results. Yahoo and Yahoo! Finance are trademarks of Yahoo Inc.;
this site is neither endorsed by nor affiliated with that company. A detailed discussion of
the terms and licences governing the acquisition and redistribution of the underlying data is
available in [LICENSING.md](LICENSING.md).
