/* ============================================================================
   StocKK Forecast — animated market background.
   A decorative candlestick chart drawn on a canvas behind every panel. The
   prices are synthetic and show no real data.

   The series is periodic: its last close equals its first open, so the chart
   joins itself with no visible jump when the loop restarts. Time comes from the
   wall clock, so the chart carries on from the same place between pages.
   ========================================================================== */
(function () {
  "use strict";

  var PAUSE_KEY = "stockk-forecast-bg-paused";
  var SEED = 20260407;
  var CANDLE_COUNT = 240;
  var SECONDS_PER_CANDLE = 1.5;
  var LOOP_SECONDS = CANDLE_COUNT * SECONDS_PER_CANDLE;
  var FRAME_MS = 1000 / 30;
  var MAX_PIXEL_RATIO = 1.5;
  var CENTER_WINDOW = 26;
  var TRADE_WINDOW = 7;
  var MIN_TRADE_GAP = 9;
  var TRADE_FADE_CANDLES = 2.5;
  var TRADE_LABEL_OPACITY = 0.6;
  var RIGHT_MARGIN_CANDLES = 7;
  var GRID_ROWS = 9;
  var GRID_EVERY_CANDLES = 4;
  var PRICE_UNITS_ON_SCREEN = 84;
  var WICK_RANGE = 4.5;
  var NOISE_RANGE = 9;
  /** [waves per loop, size in price units]. */
  var WAVE_HARMONICS = [[1, 13], [2, 10], [3, 8], [5, 5], [11, 7], [17, 8], [23, 6]];

  var PALETTES = {
    dark: {
      up: "0,229,138",
      down: "255,51,102",
      grid: "rgba(0,229,138,0.13)",
      gridStrong: "rgba(0,229,138,0.22)"
    },
    light: {
      up: "0,145,90",
      down: "214,31,82",
      grid: "rgba(0,145,90,0.16)",
      gridStrong: "rgba(0,145,90,0.28)"
    }
  };

  var ICON_PAUSE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
  var ICON_PLAY = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4l13 8-13 8z"/></svg>';

  /** Seeded generator, so every visitor and every page sees the same series. */
  function createRandom(seed) {
    var state = seed;
    return function () {
      state = (state + 0x6d2b79f5) | 0;
      var t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function wrap(index) {
    return ((index % CANDLE_COUNT) + CANDLE_COUNT) % CANDLE_COUNT;
  }

  /**
   * Build closes from waves that fit the loop a whole number of times, plus
   * per-candle noise. Slow waves make the trends, fast ones the runs of red
   * and green candles. Every part repeats exactly every CANDLE_COUNT candles.
   * @returns {number[]}
   */
  function buildCloses(random) {
    var waves = WAVE_HARMONICS.map(function (wave) {
      return {
        harmonic: wave[0],
        size: wave[1] * (0.8 + random() * 0.4),
        phase: random() * Math.PI * 2
      };
    });

    var closes = [];
    for (var c = 0; c < CANDLE_COUNT; c += 1) {
      var angle = (c / CANDLE_COUNT) * Math.PI * 2;
      var wave = waves.reduce(function (sum, w) {
        return sum + Math.sin(angle * w.harmonic + w.phase) * w.size;
      }, 0);
      closes.push(100 + wave + (random() - 0.5) * NOISE_RANGE);
    }
    return closes;
  }

  /** @returns {{open:number, close:number, high:number, low:number}[]} */
  function buildCandles(closes, random) {
    return closes.map(function (close, index) {
      var open = closes[wrap(index - 1)];
      return {
        open: open,
        close: close,
        high: Math.max(open, close) + random() * WICK_RANGE,
        low: Math.min(open, close) - random() * WICK_RANGE
      };
    });
  }

  /** Circular moving average: the level the view follows. */
  function buildCenters(closes) {
    return closes.map(function (_, index) {
      var total = 0;
      for (var k = -CENTER_WINDOW; k <= CENTER_WINDOW; k += 1) {
        total += closes[wrap(index + k)];
      }
      return total / (CENTER_WINDOW * 2 + 1);
    });
  }

  /** Buys sit on local lows and sells on local highs, like a good trader. */
  function buildTrades(closes, random) {
    var trades = {};
    var lastTrade = -MIN_TRADE_GAP;
    for (var i = 0; i < CANDLE_COUNT; i += 1) {
      var isLow = true;
      var isHigh = true;
      for (var k = -TRADE_WINDOW; k <= TRADE_WINDOW; k += 1) {
        var other = closes[wrap(i + k)];
        if (other < closes[i]) isLow = false;
        if (other > closes[i]) isHigh = false;
      }
      if ((isLow || isHigh) && i - lastTrade >= MIN_TRADE_GAP) {
        var big = random() > 0.5;
        trades[i] = {
          isBuy: isLow,
          text: big ? String(Math.round(closes[i] * 80)) : (closes[i] / 9).toFixed(1)
        };
        lastTrade = i;
      }
    }
    return trades;
  }

  function readStoredPause() {
    try {
      return localStorage.getItem(PAUSE_KEY) === "1";
    } catch (error) {
      return false;
    }
  }

  function storePause(isPaused) {
    try {
      localStorage.setItem(PAUSE_KEY, isPaused ? "1" : "0");
    } catch (error) {
      /* Private mode: the choice simply lasts for this page. */
    }
  }

  function translate(key, fallback) {
    return window.S4C ? window.S4C.t(key) : fallback;
  }

  function init() {
    var host = document.createElement("div");
    host.className = "market-bg";
    host.setAttribute("aria-hidden", "true");
    var canvas = document.createElement("canvas");
    host.appendChild(canvas);
    var ctx = canvas.getContext("2d");
    if (!ctx) return;
    document.body.insertBefore(host, document.body.firstChild);

    var random = createRandom(SEED);
    var closes = buildCloses(random);
    var candles = buildCandles(closes, random);
    var centers = buildCenters(closes);
    var trades = buildTrades(closes, random);

    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var isPaused = readStoredPause();
    var clockOffset = 0;
    var frozenSeconds = wallSeconds();
    var frameId = 0;
    var lastFrame = 0;
    var width = 0;
    var height = 0;
    var palette = PALETTES.dark;
    var button = null;

    function wallSeconds() {
      return Date.now() / 1000;
    }

    function currentSeconds() {
      if (isPaused || reducedMotion.matches) return frozenSeconds;
      return wallSeconds() + clockOffset;
    }

    function readPalette() {
      var theme = document.documentElement.getAttribute("data-theme");
      palette = PALETTES[theme] || PALETTES.dark;
    }

    function resize() {
      var ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function centerAt(head) {
      var base = Math.floor(head);
      var part = head - base;
      return centers[wrap(base)] * (1 - part) + centers[wrap(base + 1)] * part;
    }

    /** The newest candle grows from its open towards its final close. */
    function formingCandle(candle, progress) {
      var eased = progress * progress * (3 - 2 * progress);
      var wobble = Math.sin(progress * Math.PI * 5) * 1.1 * (1 - progress);
      var close = candle.open + (candle.close - candle.open) * eased + wobble;
      return {
        open: candle.open,
        close: close,
        high: Math.max(candle.open, close) + (candle.high - Math.max(candle.open, candle.close)) * eased,
        low: Math.min(candle.open, close) - (Math.min(candle.open, candle.close) - candle.low) * eased
      };
    }

    function drawGrid(step, head, center, scale) {
      var rowGap = height / GRID_ROWS;
      var rowShift = (center * scale) % rowGap;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.strokeStyle = palette.grid;
      for (var y = rowShift - rowGap; y < height + rowGap; y += rowGap) {
        ctx.moveTo(0, Math.round(y) + 0.5);
        ctx.lineTo(width, Math.round(y) + 0.5);
      }
      ctx.stroke();

      var columnGap = step * GRID_EVERY_CANDLES;
      var columnShift = (head * step) % columnGap;
      ctx.beginPath();
      ctx.strokeStyle = palette.gridStrong;
      for (var x = width - columnShift + columnGap; x > -columnGap; x -= columnGap) {
        ctx.moveTo(Math.round(x) + 0.5, 0);
        ctx.lineTo(Math.round(x) + 0.5, height);
      }
      ctx.stroke();
    }

    function drawCandle(candle, x, bodyWidth, priceToY) {
      var rgb = candle.close >= candle.open ? palette.up : palette.down;
      var top = priceToY(Math.max(candle.open, candle.close));
      var bottom = priceToY(Math.min(candle.open, candle.close));
      var bodyHeight = Math.max(bottom - top, 2);

      var column = ctx.createLinearGradient(0, bottom, 0, height);
      column.addColorStop(0, "rgba(" + rgb + ",0.16)");
      column.addColorStop(1, "rgba(" + rgb + ",0)");
      ctx.fillStyle = column;
      ctx.fillRect(x - bodyWidth / 2, bottom, bodyWidth, height - bottom);

      ctx.fillStyle = "rgba(" + rgb + ",0.75)";
      ctx.fillRect(x - 0.75, priceToY(candle.high), 1.5, priceToY(candle.low) - priceToY(candle.high));

      ctx.fillStyle = "rgba(" + rgb + ",0.18)";
      ctx.fillRect(x - bodyWidth / 2 - 3, top - 3, bodyWidth + 6, bodyHeight + 6);
      ctx.fillStyle = "rgba(" + rgb + ",0.95)";
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, bodyHeight);
    }

    function drawTrade(trade, candle, x, opacity, priceToY) {
      var rgb = trade.isBuy ? palette.up : palette.down;
      var y = trade.isBuy ? priceToY(candle.low) + 24 : priceToY(candle.high) - 14;
      ctx.fillStyle = "rgba(" + rgb + "," + opacity + ")";
      ctx.fillText((trade.isBuy ? "▲ " : "▼ ") + trade.text, x, y);
    }

    function draw() {
      if (!width || !height) return;
      var head = (currentSeconds() % LOOP_SECONDS) / SECONDS_PER_CANDLE;
      var newest = Math.floor(head);
      var progress = head - newest;

      var step = Math.max(14, Math.min(26, width / 62));
      var bodyWidth = step * 0.62;
      var scale = height / PRICE_UNITS_ON_SCREEN;
      var rightEdge = width - step * RIGHT_MARGIN_CANDLES;
      var visible = Math.ceil(rightEdge / step) + 3;
      var center = centerAt(head - visible / 2);
      var middle = height * 0.5;
      function priceToY(price) {
        return middle - (price - center) * scale;
      }

      ctx.clearRect(0, 0, width, height);
      drawGrid(step, head, center, scale);

      ctx.font = "600 " + Math.max(12, step * 0.6) + "px 'Share Tech Mono', ui-monospace, monospace";
      ctx.textAlign = "center";

      for (var age = visible; age >= 0; age -= 1) {
        var index = wrap(newest - age);
        var x = rightEdge - (age + progress) * step;
        var candle = age === 0 ? formingCandle(candles[index], progress) : candles[index];
        drawCandle(candle, x, bodyWidth, priceToY);

        var trade = trades[index];
        if (trade && age > 0) {
          var opacity = Math.min((age - 1 + progress) / TRADE_FADE_CANDLES, 1) * TRADE_LABEL_OPACITY;
          drawTrade(trade, candle, x, opacity, priceToY);
        }
      }
    }

    function shouldRun() {
      return !isPaused && !reducedMotion.matches && !document.hidden;
    }

    function tick(now) {
      frameId = window.requestAnimationFrame(tick);
      if (now - lastFrame < FRAME_MS) return;
      lastFrame = now;
      draw();
    }

    function sync() {
      window.cancelAnimationFrame(frameId);
      frameId = 0;
      if (shouldRun()) {
        frameId = window.requestAnimationFrame(tick);
      } else {
        draw();
      }
    }

    function renderButton() {
      if (!button) return;
      var key = isPaused ? "bg.play" : "bg.pause";
      var label = translate(key, isPaused ? "Resume background animation" : "Pause background animation");
      button.innerHTML = isPaused ? ICON_PLAY : ICON_PAUSE;
      button.setAttribute("aria-label", label);
      button.setAttribute("title", label);
      button.setAttribute("aria-pressed", isPaused ? "true" : "false");
      button.hidden = reducedMotion.matches;
    }

    function setPaused(next) {
      if (next) {
        frozenSeconds = currentSeconds();
      } else {
        clockOffset = frozenSeconds - wallSeconds();
      }
      isPaused = next;
      storePause(next);
      renderButton();
      sync();
    }

    function mountButton() {
      var footer = document.querySelector(".site-footer");
      if (!footer) return;
      button = document.createElement("button");
      button.type = "button";
      button.className = "bg-toggle";
      button.addEventListener("click", function () {
        setPaused(!isPaused);
      });
      footer.appendChild(button);
      renderButton();
    }

    readPalette();
    mountButton();
    resize();
    sync();

    var resizeTimer = 0;
    window.addEventListener("resize", function () {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(resize, 120);
    });
    document.addEventListener("visibilitychange", sync);
    document.addEventListener("s4c:langchange", renderButton);
    reducedMotion.addEventListener("change", function () {
      renderButton();
      sync();
    });
    new MutationObserver(function () {
      readPalette();
      draw();
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
