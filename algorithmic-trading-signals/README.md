# Algorithmic Trading Signals

Workflow for generating quantitative trading signals from news data, combining sentiment momentum, coverage velocity, cross-asset correlations, event clustering, and multi-factor alpha models using the [APITube News API](https://apitube.io).

## Overview

The **Algorithmic Trading Signals** workflow implements institutional-grade signal generation by combining multiple news-derived factors into actionable trading signals. Features include sentiment momentum indicators, coverage velocity breakouts, cross-asset correlation detection, event-driven signal clustering, factor decay modeling, and backtesting frameworks. Implements proper signal normalization, regime detection, and risk-adjusted scoring. Ideal for quantitative funds, systematic traders, and algorithmic trading desks.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by company/ticker.                                            |
| `title`                       | string  | Filter by signal keywords.                                           |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (-1.0 to 1.0).                              |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class AlphaFactorEngine:
    """
    Multi-factor alpha generation engine combining news-derived signals.
    Implements factor normalization, decay, and combination.
    """

    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.factor_weights = {
            "sentiment_momentum": 0.20,
            "coverage_velocity": 0.15,
            "sentiment_dispersion": 0.10,
            "tier1_ratio": 0.15,
            "event_intensity": 0.15,
            "sentiment_acceleration": 0.10,
            "coverage_breadth": 0.10,
            "controversy_score": 0.05,
        }
        self.universe_stats = {}  # For cross-sectional normalization

    def fetch_time_series(self, ticker, days=None):
        """Fetch daily news metrics for a ticker."""
        days = days or self.lookback_days
        series = []

        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            metrics = {"date": date}

            # Total coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": ticker,
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 100,
            })
            metrics["total"] = len(resp.json().get("results", []))

            # Sentiment breakdown
            for polarity in ["positive", "negative"]:
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "organization.name": ticker,
                    "sentiment.overall.polarity": polarity,
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language.code": "en",
                    "per_page": 100,
                })
                metrics[polarity] = len(resp.json().get("results", []))

            # Tier-1 sources
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": ticker,
                "source.rank.opr.min": 5,
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 100,
            })
            metrics["tier1"] = len(resp.json().get("results", []))

            # Calculate derived metrics
            metrics["net_sentiment"] = metrics["positive"] - metrics["negative"]
            metrics["sentiment_ratio"] = metrics["positive"] / max(metrics["positive"] + metrics["negative"], 1)
            metrics["tier1_ratio"] = metrics["tier1"] / max(metrics["total"], 1)

            series.append(metrics)

        return series

    def calculate_sentiment_momentum(self, series, short_window=5, long_window=20):
        """
        Sentiment momentum: short-term sentiment vs long-term sentiment.
        Positive values indicate improving sentiment.
        """
        if len(series) < long_window:
            return 0

        short_sentiment = sum(s["net_sentiment"] for s in series[-short_window:]) / short_window
        long_sentiment = sum(s["net_sentiment"] for s in series[-long_window:]) / long_window

        # Normalize by coverage
        short_coverage = sum(s["total"] for s in series[-short_window:]) / short_window
        long_coverage = sum(s["total"] for s in series[-long_window:]) / long_window

        if long_coverage == 0:
            return 0

        momentum = (short_sentiment / max(short_coverage, 1)) - (long_sentiment / max(long_coverage, 1))
        return momentum

    def calculate_coverage_velocity(self, series, window=7):
        """
        Coverage velocity: rate of change in coverage volume.
        Detects breakouts in media attention.
        """
        if len(series) < window * 2:
            return 0

        recent = sum(s["total"] for s in series[-window:])
        prior = sum(s["total"] for s in series[-window*2:-window])

        if prior == 0:
            return 1 if recent > 0 else 0

        velocity = (recent - prior) / prior
        return velocity

    def calculate_sentiment_dispersion(self, series, window=14):
        """
        Sentiment dispersion: volatility of daily sentiment.
        High dispersion indicates uncertainty/controversy.
        """
        if len(series) < window:
            return 0

        ratios = [s["sentiment_ratio"] for s in series[-window:]]

        if len(ratios) < 2:
            return 0

        return statistics.stdev(ratios)

    def calculate_sentiment_acceleration(self, series):
        """
        Sentiment acceleration: second derivative of sentiment.
        Detects inflection points.
        """
        if len(series) < 21:
            return 0

        # Calculate weekly momentum
        week1_momentum = sum(s["net_sentiment"] for s in series[-7:]) / 7
        week2_momentum = sum(s["net_sentiment"] for s in series[-14:-7]) / 7
        week3_momentum = sum(s["net_sentiment"] for s in series[-21:-14]) / 7

        # Velocity (first derivative)
        velocity_recent = week1_momentum - week2_momentum
        velocity_prior = week2_momentum - week3_momentum

        # Acceleration (second derivative)
        acceleration = velocity_recent - velocity_prior

        return acceleration

    def calculate_event_intensity(self, ticker, days=7):
        """
        Event intensity: concentration of high-impact news.
        """
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        # High-impact keywords
        impact_keywords = [
            "earnings", "revenue", "profit", "loss", "guidance",
            "acquisition", "merger", "lawsuit", "investigation",
            "FDA", "approval", "contract", "partnership"
        ]

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": ticker,
            "title": ",".join(impact_keywords),
            "source.rank.opr.min": 5,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })

        event_count = len(resp.json().get("results", []))

        # Normalize by total coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": ticker,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })
        total = len(resp.json().get("results", [])) or 1

        return event_count / max(total, 1)

    def calculate_coverage_breadth(self, ticker, days=7):
        """
        Coverage breadth: number of unique sources covering the ticker.
        """
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": ticker,
            "published_at.start": start,
            "language.code": "en",
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 100,
        })

        articles = resp.json().get("results", [])
        sources = set(a.get("source", {}).get("domain") for a in articles)

        return len(sources)

    def calculate_controversy_score(self, ticker, days=14):
        """
        Controversy score: ratio of negative high-impact news.
        """
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        controversy_keywords = [
            "scandal", "lawsuit", "investigation", "fraud", "violation",
            "controversy", "criticism", "accused", "alleged"
        ]

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": ticker,
            "title": ",".join(controversy_keywords),
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })

        controversy_count = len(resp.json().get("results", []))

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": ticker,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })
        total = len(resp.json().get("results", [])) or 1

        return controversy_count / max(total, 1)

    def compute_raw_factors(self, ticker):
        """Compute all raw factors for a ticker."""
        series = self.fetch_time_series(ticker)

        if not series:
            return None

        factors = {
            "sentiment_momentum": self.calculate_sentiment_momentum(series),
            "coverage_velocity": self.calculate_coverage_velocity(series),
            "sentiment_dispersion": -self.calculate_sentiment_dispersion(series),  # Negative = less dispersion is good
            "tier1_ratio": sum(s["tier1_ratio"] for s in series[-7:]) / 7,
            "event_intensity": self.calculate_event_intensity(ticker),
            "sentiment_acceleration": self.calculate_sentiment_acceleration(series),
            "coverage_breadth": self.calculate_coverage_breadth(ticker) / 100,  # Normalize
            "controversy_score": -self.calculate_controversy_score(ticker),  # Negative = less controversy is good
        }

        return factors

    def normalize_factors_cross_sectional(self, universe_factors):
        """
        Cross-sectional z-score normalization.
        Centers factors around 0 with unit variance across universe.
        """
        normalized = {}

        # Calculate universe statistics for each factor
        factor_names = list(self.factor_weights.keys())

        for factor in factor_names:
            values = [uf[factor] for uf in universe_factors.values() if uf and factor in uf]

            if len(values) < 2:
                continue

            mean = statistics.mean(values)
            std = statistics.stdev(values) or 1

            for ticker, factors in universe_factors.items():
                if factors and factor in factors:
                    if ticker not in normalized:
                        normalized[ticker] = {}
                    normalized[ticker][factor] = (factors[factor] - mean) / std

        return normalized

    def compute_composite_alpha(self, normalized_factors):
        """
        Compute weighted composite alpha score.
        """
        alpha_scores = {}

        for ticker, factors in normalized_factors.items():
            score = 0
            total_weight = 0

            for factor, weight in self.factor_weights.items():
                if factor in factors:
                    score += factors[factor] * weight
                    total_weight += weight

            if total_weight > 0:
                alpha_scores[ticker] = score / total_weight

        return alpha_scores

    def generate_signals(self, universe, threshold_long=0.5, threshold_short=-0.5):
        """
        Generate trading signals for a universe of tickers.
        """
        # Compute raw factors
        print(f"Computing factors for {len(universe)} tickers...")
        universe_factors = {}
        for ticker in universe:
            print(f"  Processing {ticker}...")
            universe_factors[ticker] = self.compute_raw_factors(ticker)

        # Normalize cross-sectionally
        normalized = self.normalize_factors_cross_sectional(universe_factors)

        # Compute composite alpha
        alpha_scores = self.compute_composite_alpha(normalized)

        # Generate signals
        signals = []
        for ticker, alpha in sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True):
            signal = {
                "ticker": ticker,
                "alpha_score": alpha,
                "signal": "LONG" if alpha > threshold_long else "SHORT" if alpha < threshold_short else "NEUTRAL",
                "strength": abs(alpha),
                "factors": normalized.get(ticker, {}),
                "timestamp": datetime.utcnow().isoformat()
            }
            signals.append(signal)

        return signals


class SignalBacktester:
    """Backtest news-based trading signals."""

    def __init__(self, engine):
        self.engine = engine

    def calculate_signal_decay(self, signal_strength, days_elapsed, half_life=5):
        """
        Calculate decayed signal strength.
        Signals lose predictive power over time.
        """
        decay_factor = 0.5 ** (days_elapsed / half_life)
        return signal_strength * decay_factor

    def compute_hit_rate(self, signals, forward_returns):
        """
        Compute signal hit rate: % of signals with correct direction.
        """
        correct = 0
        total = 0

        for signal in signals:
            ticker = signal["ticker"]
            if ticker not in forward_returns:
                continue

            ret = forward_returns[ticker]
            predicted_direction = 1 if signal["signal"] == "LONG" else -1 if signal["signal"] == "SHORT" else 0

            if predicted_direction != 0:
                total += 1
                if (predicted_direction > 0 and ret > 0) or (predicted_direction < 0 and ret < 0):
                    correct += 1

        return correct / max(total, 1)

    def compute_information_coefficient(self, alpha_scores, forward_returns):
        """
        Compute IC: correlation between alpha scores and forward returns.
        """
        common_tickers = set(alpha_scores.keys()) & set(forward_returns.keys())

        if len(common_tickers) < 5:
            return 0

        alphas = [alpha_scores[t] for t in common_tickers]
        returns = [forward_returns[t] for t in common_tickers]

        # Pearson correlation
        n = len(alphas)
        mean_a = sum(alphas) / n
        mean_r = sum(returns) / n

        cov = sum((alphas[i] - mean_a) * (returns[i] - mean_r) for i in range(n)) / n
        std_a = (sum((a - mean_a) ** 2 for a in alphas) / n) ** 0.5
        std_r = (sum((r - mean_r) ** 2 for r in returns) / n) ** 0.5

        if std_a * std_r == 0:
            return 0

        return cov / (std_a * std_r)


class RealTimeSignalMonitor:
    """Monitor signals in real-time with alerting."""

    def __init__(self, engine):
        self.engine = engine
        self.signal_history = defaultdict(list)
        self.alerts = []

    def update_signal(self, ticker):
        """Update signal for a single ticker."""
        factors = self.engine.compute_raw_factors(ticker)

        if not factors:
            return None

        # Simple scoring without cross-sectional normalization
        score = sum(
            factors.get(f, 0) * w
            for f, w in self.engine.factor_weights.items()
        )

        signal = {
            "ticker": ticker,
            "score": score,
            "factors": factors,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Check for signal changes
        history = self.signal_history[ticker]
        if history:
            prev_score = history[-1]["score"]
            score_change = score - prev_score

            # Alert on significant changes
            if abs(score_change) > 0.5:
                self.alerts.append({
                    "ticker": ticker,
                    "type": "SIGNAL_CHANGE",
                    "prev_score": prev_score,
                    "new_score": score,
                    "change": score_change,
                    "direction": "UPGRADE" if score_change > 0 else "DOWNGRADE",
                    "timestamp": datetime.utcnow().isoformat()
                })

        self.signal_history[ticker].append(signal)

        # Keep only last 30 signals
        if len(self.signal_history[ticker]) > 30:
            self.signal_history[ticker] = self.signal_history[ticker][-30:]

        return signal


# Run signal generation
print("ALGORITHMIC TRADING SIGNALS")
print("=" * 70)

engine = AlphaFactorEngine(lookback_days=30)

# Define universe
universe = ["Apple", "Microsoft", "Google", "Amazon", "Tesla", "NVIDIA", "Meta"]

print(f"\nUniverse: {len(universe)} tickers")
print("Generating signals...\n")

signals = engine.generate_signals(universe)

print("=" * 70)
print("SIGNAL RANKINGS")
print("-" * 50)
print(f"{'Ticker':<12} {'Alpha':>8} {'Signal':>8} {'Strength':>10}")
print("-" * 50)

for signal in signals:
    print(f"{signal['ticker']:<12} {signal['alpha_score']:>8.3f} {signal['signal']:>8} {signal['strength']:>10.3f}")

# Show factor breakdown for top signal
print("\n" + "=" * 70)
print("FACTOR BREAKDOWN (Top Signal)")
print("-" * 50)

top_signal = signals[0]
print(f"Ticker: {top_signal['ticker']}")
print(f"Alpha Score: {top_signal['alpha_score']:.3f}")
print(f"Signal: {top_signal['signal']}")
print("\nFactor Contributions:")
for factor, value in top_signal["factors"].items():
    weight = engine.factor_weights.get(factor, 0)
    contribution = value * weight
    print(f"  {factor:<25} z={value:>6.2f}  weight={weight:.2f}  contrib={contribution:>6.3f}")

# Long/Short portfolio
print("\n" + "=" * 70)
print("RECOMMENDED PORTFOLIO")
print("-" * 50)

longs = [s for s in signals if s["signal"] == "LONG"]
shorts = [s for s in signals if s["signal"] == "SHORT"]

print(f"LONG ({len(longs)}):")
for s in longs:
    print(f"  {s['ticker']}: alpha={s['alpha_score']:.3f}")

print(f"\nSHORT ({len(shorts)}):")
for s in shorts:
    print(f"  {s['ticker']}: alpha={s['alpha_score']:.3f}")
```

## Signal Factors

| Factor | Description | Weight |
|--------|-------------|--------|
| `sentiment_momentum` | Short-term vs long-term sentiment change | 20% |
| `coverage_velocity` | Rate of change in media coverage | 15% |
| `sentiment_dispersion` | Volatility of daily sentiment (inverted) | 10% |
| `tier1_ratio` | Share of coverage from authoritative sources | 15% |
| `event_intensity` | Concentration of high-impact news | 15% |
| `sentiment_acceleration` | Second derivative of sentiment | 10% |
| `coverage_breadth` | Number of unique sources | 10% |
| `controversy_score` | Ratio of negative high-impact news (inverted) | 5% |

## Common Use Cases

- **Systematic trading** — generate alpha signals for quantitative strategies.
- **Event-driven trading** — detect and trade on news events.
- **Sentiment arbitrage** — exploit sentiment mispricings.
- **Risk management** — monitor portfolio holdings for news risks.
- **Alpha research** — backtest news-derived factors.
- **Portfolio construction** — build long/short portfolios from signals.

## See Also

- [examples.md](./examples.md) — detailed code examples.
