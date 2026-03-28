"""Fixed universe of large-cap tickers (~50) for Finnhub rate limits and demo scope."""

TICKERS = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "AMD",
    "INTC",
    "CRM",
    "ORCL",
    "ADBE",
    # Finance
    "JPM",
    "BAC",
    "GS",
    "MS",
    "WFC",
    "C",
    "AXP",
    "BLK",
    "SCHW",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "MRK",
    "ABBV",
    "LLY",
    # Consumer
    "AMZN",
    "WMT",
    "COST",
    "TGT",
    "MCD",
    "NKE",
    "SBUX",
    # Energy
    "XOM",
    "CVX",
    "COP",
    # Industrials
    "BA",
    "CAT",
    "GE",
    "HON",
    "MMM",
    # Telecom / Media
    "DIS",
    "NFLX",
    "CMCSA",
    "T",
    "VZ",
    # Other
    "TSLA",
    "BRK-B",
    "V",
    "MA",
    "PG",
    "KO",
    "PEP",
]

TICKER_SET = frozenset(TICKERS)
