# Data Sources

All data used in this project is **public or synthetically generated**.
No proprietary S&P Global or Crisil client data is used at any point.

---

## 1. Synthetic SVB Crisis Sample Data  ← committed to repo

| Field | Value |
|-------|-------|
| **Files** | `data/svb_news_sample.csv`, `data/svb_tweets_sample.csv`, `data/svb_prices_sample.csv` |
| **Source** | Hand-crafted synthetic data (this repo) |
| **License** | MIT (same as repo) |
| **Window** | 2023-03-08 → 2023-03-17 (SVB Bank collapse) |
| **Tickers covered** | JPM, BAC, C, GS, WFC |
| **Why SVB window** | Clear, date-stamped financial events affecting all five target tickers with unambiguous sentiment direction. Allows ground-truth validation. |
| **Generation** | See `scripts/make_sample_data.py` — regenerate via `make sample` |

### News sample (svb_news_sample.csv)
- 20 hand-crafted headlines covering: SVB stock sale (Mar 8), FDIC seizure (Mar 10), BTFP announcement (Mar 12), recovery (Mar 13-17).
- Columns: `id, publish_date, title, body, source, url`
- All sources are fictional/synthetic (Reuters/Bloomberg labels used for realism but content is fictional).

### Tweets sample (svb_tweets_sample.csv)
- 25 synthetic tweets from fictional accounts.
- Columns: `tweet_id, writer, post_date, body, ticker`
- Cashtags (`$JPM`, `$BAC`, etc.) embedded for entity linker testing.

### Prices sample (svb_prices_sample.csv)
- Synthetic daily OHLCV for JPM, BAC, C, GS, WFC (2023-03-01 to 2023-03-17).
- Approximate real price levels used for realism; not for financial decision-making.
- Regenerate with real data: `python scripts/make_sample_data.py` (uses yfinance).

---

## 2. All the News (Kaggle)  ← NOT committed, download manually

| Field | Value |
|-------|-------|
| **Kaggle URL** | https://www.kaggle.com/datasets/snapcrack/all-the-news |
| **License** | CC0 Public Domain |
| **Size** | ~200 MB compressed (3 CSV parts) |
| **Download** | `python scripts/download_data.py --news` |
| **Local path** | `data/all_the_news.csv` (after download + concatenation) |
| **Columns** | id, title, publication, author, date, year, month, url, content |
| **Usage** | `NewsSource` reads this file; `make_sample_data.py` samples it to the SVB window |
| **Assumption** | Articles are primarily US news; financial keywords filter narrows to banking content |

---

## 3. Stock Tweets (Kaggle)  ← NOT committed, download manually

| Field | Value |
|-------|-------|
| **Kaggle URL** | https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction |
| **License** | CC BY-SA 4.0 |
| **Size** | ~50 MB |
| **Download** | `python scripts/download_data.py --tweets` |
| **Local path** | `data/stock_tweets.csv` |
| **Columns** | tweet_id, writer, post_date, body, ticker_symbol |
| **Usage** | `SocialSource` reads this file; filtered to SVB window by `make_sample_data.py` |

---

## 4. Yahoo Finance / yfinance  ← fetched at runtime, not committed

| Field | Value |
|-------|-------|
| **Library** | [yfinance](https://github.com/ranaroussi/yfinance) (Apache-2.0) |
| **Usage** | Historical OHLCV for VaR computation in `src/modules/portfolio_risk.py` |
| **Tickers** | JPM, BAC, C, GS, WFC + config defaults (AAPL, MSFT, XOM) |
| **License** | Yahoo Finance ToS — research / educational use |
| **Assumption** | Prices fetched at runtime; synthetic fallback used if network unavailable |

---

## 5. FinBERT (ProsusAI/finbert)  ← downloaded at runtime, not committed

| Field | Value |
|-------|-------|
| **HuggingFace URL** | https://huggingface.co/ProsusAI/finbert |
| **License** | Apache-2.0 |
| **Usage** | Sentiment classification in `src/engine/nlp.py` |
| **Weights** | Downloaded on first inference, cached by HuggingFace; never committed |

---

## 6. GDELT v2 DOC API  ← optional live feed

| Field | Value |
|-------|-------|
| **API URL** | https://api.gdeltproject.org/api/v2/doc/doc |
| **License** | Public domain — no authentication required |
| **Usage** | Optional live news in `src/engine/ingest/news.py::GDELTSource` |
| **Caching** | Results cached to `data/gdelt_cache.jsonl` to avoid re-fetching |

---

## Download Quick Reference

```bash
# Download full Kaggle datasets (requires kaggle CLI + auth)
python scripts/download_data.py --all

# Generate SVB sample from full datasets (or regenerate prices via yfinance)
make sample

# Run the replay demo using sample data
make replay
```
