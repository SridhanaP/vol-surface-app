# Volatility Surface Explorer

Live implied volatility surfaces from option prices, built with Python,
Streamlit, Plotly and SciPy. For education only, not financial advice.

- `app.py` - the Streamlit website
- `core/surface.py` - data download, implied vol, surface math
- `core/charts.py` - Plotly charts
- `collector.py` - daily snapshot job (run by GitHub Actions)
- `watchlist.txt` - tickers the collector saves every trading day
- `data/` - collected snapshots (written automatically)

Run locally:

    pip install -r requirements.txt
    streamlit run app.py
