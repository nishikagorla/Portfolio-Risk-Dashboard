#defaults and constants.
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
DEFAULT_WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]

DEFAULT_BENCHMARK = "SPY"

DEFAULT_START = "2020-01-01"

TRADING_DAYS = 252

# starting value for rf, can change to 3 yr t-bill once implemented
RISK_FREE_RATE = 0.0

VAR_CONFIDENCE = 0.95
MC_SIMULATIONS = 10000
MC_DF = 5  # for student t-distribution, lower = fatter tails

FRONTIER_POINTS = 50  # number of points traced along the efficient frontier