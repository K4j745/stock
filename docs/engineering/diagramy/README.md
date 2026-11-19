# Diagramy do pracy inżynierskiej

Diagramy w formacie tekstowym PlantUML (`.puml`). Wspólny styl jest w `_styl.iuml` i każdy
diagram dołącza go przez `!include`, więc pliki renderuje się z tego katalogu.

| Plik | Diagram | Rozdział pracy |
|---|---|---|
| `01_przypadki_uzycia.puml` | przypadki użycia (UML) | wymagania funkcjonalne |
| `02_komponenty.puml` | komponenty (UML) | architektura |
| `03_sekwencja_generowania.puml` | sekwencja generowania danych dashboardu (UML) | implementacja |
| `04_wdrozenie.puml` | wdrożenie (UML) | architektura, wdrożenie |
| `05_walk_forward.puml` | walidacja walk-forward na osi czasu (wykres Gantta) | metodyka |
| `06_aktywnosc_potok_ml.puml` | aktywność: trening i ewaluacja modeli (UML) | metodyka |
| `07_aktywnosc_arima.puml` | aktywność: ARIMA (ADF, dobór rzędu BIC, prognoza) (UML) | metodyka |
| `08_aktywnosc_symulacja_portfela.puml` | aktywność: symulacja portfela (UML) | symulacja strategii |
| `09_model_danych.puml` | model danych artefaktów JSON (UML, klasy) | projekt danych |
| `10_aktywnosc_ci_cd.puml` | aktywność: workflow GitHub Actions (UML) | wdrożenie, automatyzacja |
| `11_klasy_dashboard.puml` | klasy i moduły generatora oraz rdzenia ARIMA (UML) | implementacja |

Diagramy 01–03 zastępują wcześniejsze `uml_use_case.puml`, `uml_components.puml` i
`uml_sequence.puml`, zaktualizowane o model ARIMA, modele zapisane w repozytorium i HOLD
przy braku prognozy.

## Renderowanie

Wymagana Java (jest w systemie) i PlantUML:

- **VS Code:** rozszerzenie *PlantUML* (jebbs.plantuml), podgląd `Alt+D`, eksport PNG/SVG.
- **Wiersz poleceń** (z [plantuml.com/download](https://plantuml.com/download) pobierz `plantuml.jar`):

```bash
java -jar plantuml.jar -charset UTF-8 -tpng docs/engineering/diagramy/*.puml
```

Do druku w pracy lepszy jest wektorowy SVG albo PDF (`-tsvg`, `-tpdf`).
