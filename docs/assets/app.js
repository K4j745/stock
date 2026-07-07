/* ============================================================================
   Stock4caster — shared front-end chrome: PL/EN i18n + copyright footer.
   Default language is Polish (thesis language). The choice is persisted in
   localStorage under 'stock4caster-lang'. Static UI is translated via
   [data-i18n] / [data-i18n-html] / [data-i18n-attr] attributes; a small
   window.S4C API lets the per-page inline scripts translate the few strings
   they render dynamically.
   ========================================================================== */
(function () {
  "use strict";

  var LANG_KEY = "stock4caster-lang";
  var DEFAULT_LANG = "pl";
  var COPYRIGHT =
    "© 2025 Kajetan Kaczyński — Praca inżynierska / " +
    "Uniwersytet Przyrodniczy w Poznaniu / Wszelkie prawa zastrzeżone.";

  var DICT = {
    pl: {
      "nav.overview": "Przegląd",
      "nav.prices": "Ceny",
      "nav.charts": "Wykresy",
      "nav.signals": "Sygnały",
      "nav.portfolios": "Portfele",
      "nav.methodology": "Metodologia",

      "kpi.total": "Wszystkie sygnały",
      "kpi.buy": "Sygnały BUY",
      "kpi.sell": "Sygnały SELL",
      "kpi.hold": "Sygnały HOLD",

      "card.marketSignals": "Sygnały rynkowe",
      "card.signalSummary": "Podsumowanie sygnałów",
      "card.bySector": "Wg sektora",
      "card.universeBySector": "Uniwersum wg sektora",

      "label.model": "Model:",
      "label.autoRefresh": "Auto-odświeżanie: 60s",
      "label.loading": "Ładowanie danych rynkowych...",

      "h1.charts": "Wykresy analizy ML",
      "sub.charts": "Skuteczność modeli, ważność cech i wizualizacje predykcji",
      "h1.signals": "Panel sygnałów",
      "sub.signals": "Porównaj decyzje BUY / SELL / HOLD dla modeli technicznych i ML. Filtruj wg tickera, modelu, typu sygnału, daty lub portfela.",
      "h1.portfolio": "Symulator portfela",
      "sub.portfolio": "Handluj wirtualnie po cenach zamknięcia z ostatniej sesji. Portfele są przechowywane wyłącznie w Twojej przeglądarce — bez serwera i bez połączeń sieciowych poza wczytaniem lokalnych plików JSON.",

      "footer.index": "Dane: Yahoo Finance — To nie jest porada inwestycyjna",
      "footer.prices": "Interaktywne wykresy cen (Plotly.js)",
      "footer.charts": "Wykresy generowane automatycznie z pipeline'u ML",
      "footer.signals": "Panel sygnałów — To nie jest porada inwestycyjna",
      "footer.portfolio": "Symulator wirtualny — To nie jest porada inwestycyjna",
      "footer.methodology": "Metodologia projektu — praca inżynierska",

      "lang.aria": "Przełącz język (polski / angielski)"
    },
    en: {
      "nav.overview": "Overview",
      "nav.prices": "Prices",
      "nav.charts": "Charts",
      "nav.signals": "Signals",
      "nav.portfolios": "Portfolios",
      "nav.methodology": "Methodology",

      "kpi.total": "Total Signals",
      "kpi.buy": "BUY Signals",
      "kpi.sell": "SELL Signals",
      "kpi.hold": "HOLD Signals",

      "card.marketSignals": "Market Signals",
      "card.signalSummary": "Signal Summary",
      "card.bySector": "By Sector",
      "card.universeBySector": "Universe by sector",

      "label.model": "Model:",
      "label.autoRefresh": "Auto-refresh: 60s",
      "label.loading": "Loading market data...",

      "h1.charts": "ML Analysis Charts",
      "sub.charts": "Model performance, feature importance, and prediction analysis visualizations",
      "h1.signals": "Signals Panel",
      "sub.signals": "Compare BUY / SELL / HOLD calls across technical and ML models. Filter by ticker, model, signal type, date, or portfolio.",
      "h1.portfolio": "Portfolio Simulator",
      "sub.portfolio": "Trade virtually using last-session close prices from this project. Portfolios are stored in your browser only — no server, no network calls beyond loading the local JSON files.",

      "footer.index": "Data: Yahoo Finance — Not financial advice",
      "footer.prices": "Interactive price charts powered by Plotly.js",
      "footer.charts": "Charts auto-generated from the ML pipeline",
      "footer.signals": "Signals panel — Not financial advice",
      "footer.portfolio": "Virtual simulator — Not financial advice",
      "footer.methodology": "Project methodology — engineering thesis",

      "lang.aria": "Switch language (Polish / English)"
    }
  };

  var lang = localStorage.getItem(LANG_KEY) || DEFAULT_LANG;
  if (lang !== "pl" && lang !== "en") lang = DEFAULT_LANG;

  function tr(key) {
    var d = DICT[lang] || DICT[DEFAULT_LANG];
    if (d && d[key] != null) return d[key];
    if (DICT.en[key] != null) return DICT.en[key];
    return key;
  }

  function applyTranslations(root) {
    root = root || document;
    root.querySelectorAll("[data-i18n]").forEach(function (el) {
      var v = tr(el.getAttribute("data-i18n"));
      if (v != null && el.textContent !== v) el.textContent = v;
    });
    root.querySelectorAll("[data-i18n-html]").forEach(function (el) {
      var v = tr(el.getAttribute("data-i18n-html"));
      if (v != null) el.innerHTML = v;
    });
    root.querySelectorAll("[data-i18n-attr]").forEach(function (el) {
      el.getAttribute("data-i18n-attr").split(";").forEach(function (pair) {
        var p = pair.split(":");
        var attr = (p[0] || "").trim();
        var key = (p[1] || "").trim();
        if (!attr || !key) return;
        var v = tr(key);
        if (v != null) el.setAttribute(attr, v);
      });
    });
  }

  function injectCopyright() {
    document.querySelectorAll(".site-footer").forEach(function (f) {
      if (f.querySelector(".vw-copyright")) return;
      var span = document.createElement("span");
      span.className = "vw-copyright";
      span.textContent = COPYRIGHT;
      f.appendChild(span);
    });
  }

  function renderLangToggle() {
    var pl = lang === "pl";
    return (
      '<span class="lang-pl ' + (pl ? "lang-on" : "lang-off") + '">PL</span>' +
      '<span class="lang-sep"> / </span>' +
      '<span class="lang-en ' + (pl ? "lang-off" : "lang-on") + '">EN</span>'
    );
  }

  function mountLangToggle() {
    var host = document.querySelector(".site-header .header-right");
    if (!host || document.getElementById("s4cLangToggle")) return;
    var btn = document.createElement("button");
    btn.className = "lang-toggle";
    btn.id = "s4cLangToggle";
    btn.type = "button";
    btn.setAttribute("data-i18n-attr", "aria-label:lang.aria");
    btn.setAttribute("aria-label", tr("lang.aria"));
    btn.innerHTML = renderLangToggle();
    // Place the language switch first so it reads left-to-right with theme.
    host.insertBefore(btn, host.firstChild);
    btn.addEventListener("click", function () {
      setLang(lang === "pl" ? "en" : "pl");
    });
  }

  function setLang(next) {
    lang = next === "en" ? "en" : "pl";
    localStorage.setItem(LANG_KEY, lang);
    document.documentElement.setAttribute("lang", lang);
    var btn = document.getElementById("s4cLangToggle");
    if (btn) btn.innerHTML = renderLangToggle();
    applyTranslations(document);
    document.dispatchEvent(new CustomEvent("s4c:langchange", { detail: { lang: lang } }));
  }

  window.S4C = {
    t: tr,
    get lang() { return lang; },
    setLang: setLang,
    apply: applyTranslations
  };

  function init() {
    document.documentElement.setAttribute("lang", lang);
    mountLangToggle();
    injectCopyright();
    applyTranslations(document);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
