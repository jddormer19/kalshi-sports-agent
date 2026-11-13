"""Minimal read-only client for Kalshi's public market-data API."""
import time
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"


class Kalshi:
    def __init__(self, rps: float = 6.0):
        self.s = requests.Session()
        self.max_rps, self.rps = rps, rps
        self._last = 0.0
        self._cutoff = None

    def get(self, path: str, **params):
        for attempt in range(6):
            wait = 1.0 / self.rps - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(BASE + path, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:  # adaptive: back off the rate, brief pause
                self.rps = max(1.0, self.rps * 0.7)
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            self.rps = min(self.max_rps, self.rps * 1.02)
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {path} failed after retries")

    def paginate(self, path: str, key: str, max_items=None, **params):
        cursor, n = None, 0
        while True:
            d = self.get(path, cursor=cursor, **params) if cursor else self.get(path, **params)
            if not d:
                return
            for item in d.get(key, []):
                yield item
                n += 1
                if max_items and n >= max_items:
                    return
            cursor = d.get("cursor")
            if not cursor or not d.get(key):
                return

    def cutoff(self) -> float:
        if self._cutoff is None:
            import pandas as pd
            d = self.get("/historical/cutoff") or {}
            self._cutoff = pd.Timestamp(d.get("market_settled_ts", "1970-01-01T00:00:00Z")).timestamp()
        return self._cutoff

    def candles(self, series: str, ticker: str, start_ts: int, end_ts: int, period: int = 1440):
        """Candles (default daily); tries live endpoint first, then the historical archive."""
        q = dict(start_ts=start_ts, end_ts=end_ts, period_interval=period)
        live, hist = f"/series/{series}/markets/{ticker}/candlesticks", f"/historical/markets/{ticker}/candlesticks"
        # markets settled before the archive cutoff live in /historical; try the likely one first
        for path in ((hist, live) if end_ts < self.cutoff() else (live, hist)):
            d = self.get(path, **q)
            if d and d.get("candlesticks"):
                return d["candlesticks"]
        return []
