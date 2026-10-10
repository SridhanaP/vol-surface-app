"""Daily options snapshot collector.

Saves, for every ticker in watchlist.txt:
  data/raw/YYYY-MM-DD.csv.gz   every usable IV point (for rebuilding surfaces later)
  data/summary.csv             one row per ticker per day (ATM IV, skew, spot...)

Run:  python collector.py          (skips non-trading days)
      python collector.py --force  (run anyway)
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from core.surface import NY, atm_by_expiry, atm_iv_at, fetch_chain_points, get_spot, skew_at

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SUMMARY = DATA / "summary.csv"


def read_watchlist():
    lines = (ROOT / "watchlist.txt").read_text().splitlines()
    return [l.split("#")[0].strip().upper() for l in lines if l.split("#")[0].strip()]


def snapshot(ticker):
    pts, meta = fetch_chain_points(ticker, min_days=1, max_days=400)
    quality = "bid_ask"
    if len(pts) < 10:  # bid/ask missing: fall back to last trades
        pts, meta = fetch_chain_points(ticker, min_days=1, max_days=400, use_last=True)
        quality = "last_price"
    if len(pts) < 10:
        raise ValueError(f"only {len(pts)} usable quotes")
    atm = atm_by_expiry(pts)
    summary = dict(
        ticker=ticker, spot=meta["spot"], rate=meta["rate"],
        atm_iv_7d=atm_iv_at(atm, 7), atm_iv_30d=atm_iv_at(atm, 30),
        atm_iv_90d=atm_iv_at(atm, 90), skew_30d=skew_at(pts, 30),
        n_points=len(pts), n_expiries=pts.expiry.nunique(), quality=quality,
        fetched=meta["fetched"])
    return pts.assign(ticker=ticker), summary


def main():
    force = "--force" in sys.argv
    today = pd.Timestamp.now(tz=NY).date()

    _, last_trade = get_spot(yf.Ticker("SPY"))
    if last_trade != today and not force:
        print(f"{today} is not a trading day (last SPY trade {last_trade}). Skipping.")
        return

    tickers = read_watchlist()
    print(f"Collecting {len(tickers)} tickers for {today}")
    all_pts, summaries = [], []
    for t in tickers:
        try:
            pts, summ = snapshot(t)
            all_pts.append(pts)
            summaries.append(summ)
            print(f"  {t:6s} ok   {summ['n_points']:4d} pts  "
                  f"{summ['n_expiries']:2d} expiries  ({summ['quality']})")
        except Exception as e:
            print(f"  {t:6s} FAIL {e}")
        time.sleep(1.0)  # be polite to Yahoo

    if not summaries:
        print("Nothing collected.")
        sys.exit(1)

    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{today}.csv.gz"
    pd.concat(all_pts).round(6).to_csv(raw_path, index=False, compression="gzip")

    new = pd.DataFrame(summaries)
    new.insert(0, "date", str(today))
    if SUMMARY.exists():
        old = pd.read_csv(SUMMARY)
        old = old[~((old.date == str(today)) & old.ticker.isin(new.ticker))]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["date", "ticker"]).to_csv(SUMMARY, index=False)
    print(f"Saved {raw_path.name} and updated summary.csv "
          f"({len(summaries)}/{len(tickers)} tickers)")


if __name__ == "__main__":
    main()
