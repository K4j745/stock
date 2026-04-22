# Stock Terminal Dashboard

A terminal-style dashboard for GitHub Pages displaying ML trading signals, interactive price charts, and a virtual portfolio simulator.

## Enable GitHub Pages

1. Go to your repository **Settings**
2. Navigate to **Pages** (left sidebar)
3. Under **Source**, select **Deploy from a branch**
4. Set branch to `main` (or your default branch) and folder to `/docs`
5. Click **Save**

The dashboard will be available at `https://<username>.github.io/<repo-name>/`

## Data Updates

Data is auto-updated by the GitHub Actions workflow (`.github/workflows/dashboard.yml`):
- **Schedule**: Mon-Fri at 17:00 UTC (after US market close)
- **Manual**: Go to Actions tab → "Update Dashboard Data" → "Run workflow"

To generate data locally:
```bash
pip install -r dashboard/requirements.txt
python dashboard/generate.py
```

## Pages

- **Home** (`index.html`): Signal overview table with RSI, MACD, and composite signals
- **Charts** (`charts/`): ML pipeline analysis plots (SHAP, feature importance, etc.)
- **Prices** (`prices/`): Interactive candlestick charts with technical indicators
- **Portfolio** (`portfolio/`): Virtual portfolio simulator with $10,000 starting cash
