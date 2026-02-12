# Economic Sentiment Indicators

Workflow for constructing news-based economic indicators, nowcasting economic activity, building leading indicators from media sentiment, tracking sector rotation signals, and detecting economic regime changes using the [APITube News API](https://apitube.io).

## Overview

The **Economic Sentiment Indicators** workflow creates quantitative economic indicators from news data by aggregating sentiment across economic themes, weighting by source authority and recency, applying seasonal adjustments, and correlating with official economic data. Features include consumer confidence proxies, business sentiment indices, sector momentum indicators, recession probability models, and inflation expectation tracking. Ideal for economists, macro traders, policy analysts, and research institutions.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `topic.id`                    | string  | Filter by economic topic.                                            |
| `title`                       | string  | Filter by economic keywords.                                         |
| `sentiment.overall.polarity`  | string  | Filter by sentiment.                                                 |
| `source.rank.opr.min`         | number  | Minimum source authority.                                            |
| `source.country`              | string  | Filter by country.                                                   |
| `published_at.start`          | string  | Start date.                                                          |
| `published_at.end`            | string  | End date.                                                            |
| `language`                    | string  | Filter by language code.                                             |
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


class EconomicTheme:
    """Represents an economic theme for indicator construction."""

    def __init__(self, name, positive_keywords, negative_keywords, weight=1.0):
        self.name = name
        self.positive_keywords = positive_keywords
        self.negative_keywords = negative_keywords
        self.weight = weight
        self.daily_readings = []

    def add_reading(self, date, positive_count, negative_count, total_count):
        """Add a daily reading."""
        if total_count == 0:
            sentiment = 0
        else:
            sentiment = (positive_count - negative_count) / total_count

        self.daily_readings.append({
            "date": date,
            "positive": positive_count,
            "negative": negative_count,
            "total": total_count,
            "sentiment": sentiment
        })


class EconomicSentimentIndex:
    """
    Constructs economic sentiment indicators from news data.
    Similar to University of Michigan Consumer Sentiment or PMI.
    """

    # Economic themes for indicator construction
    THEMES = {
        "consumer_confidence": {
            "positive": ["consumer spending", "retail sales strong", "consumer confidence", "shopping", "buying"],
            "negative": ["consumer slowdown", "retail weakness", "consumer pessimism", "spending cuts"],
            "weight": 0.20
        },
        "employment": {
            "positive": ["hiring", "job growth", "employment gains", "labor market strong", "wages rising"],
            "negative": ["layoffs", "unemployment", "job cuts", "hiring freeze", "workforce reduction"],
            "weight": 0.20
        },
        "business_investment": {
            "positive": ["capital expenditure", "business investment", "expansion plans", "capacity increase"],
            "negative": ["investment cuts", "capex reduction", "business contraction", "pullback"],
            "weight": 0.15
        },
        "manufacturing": {
            "positive": ["manufacturing growth", "factory orders", "industrial production up", "PMI expansion"],
            "negative": ["manufacturing decline", "factory slowdown", "industrial weakness", "PMI contraction"],
            "weight": 0.15
        },
        "housing": {
            "positive": ["housing starts", "home sales strong", "mortgage applications up", "housing recovery"],
            "negative": ["housing slowdown", "home sales fall", "mortgage decline", "housing weakness"],
            "weight": 0.10
        },
        "inflation": {
            "positive": ["inflation easing", "price stability", "disinflation", "inflation cooling"],
            "negative": ["inflation rising", "price increases", "cost pressures", "inflation concerns"],
            "weight": 0.10
        },
        "financial_conditions": {
            "positive": ["credit available", "lending growth", "financial stability", "market rally"],
            "negative": ["credit tightening", "lending decline", "financial stress", "market selloff"],
            "weight": 0.10
        }
    }

    SECTOR_THEMES = {
        "technology": ["tech sector", "software", "semiconductor", "cloud computing", "AI"],
        "healthcare": ["healthcare", "pharmaceutical", "biotech", "medical", "hospital"],
        "financials": ["banking", "financial services", "insurance", "investment", "fintech"],
        "energy": ["oil", "gas", "renewable energy", "utilities", "energy sector"],
        "consumer": ["retail", "consumer goods", "e-commerce", "restaurants", "travel"],
        "industrials": ["manufacturing", "aerospace", "defense", "construction", "transportation"],
        "materials": ["mining", "chemicals", "metals", "commodities", "raw materials"],
        "real_estate": ["real estate", "property", "REIT", "commercial real estate", "residential"]
    }

    def __init__(self, country="us"):
        self.country = country
        self.themes = {}
        self.sector_data = {}
        self.composite_readings = []

    def fetch_theme_data(self, theme_name, theme_config, days=90):
        """Fetch daily data for a theme."""
        theme = EconomicTheme(
            theme_name,
            theme_config["positive"],
            theme_config["negative"],
            theme_config["weight"]
        )

        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            # Positive mentions
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(theme_config["positive"]),
                "source.country": self.country,
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            positive = resp.json().get("total_results", 0)

            # Negative mentions
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(theme_config["negative"]),
                "source.country": self.country,
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            negative = resp.json().get("total_results", 0)

            total = positive + negative
            theme.add_reading(date, positive, negative, total)

        self.themes[theme_name] = theme
        return theme

    def build_composite_index(self, days=90):
        """Build composite economic sentiment index."""
        print(f"Building composite index for {self.country.upper()}...")

        # Fetch all themes
        for name, config in self.THEMES.items():
            print(f"  Fetching {name}...")
            self.fetch_theme_data(name, config, days)

        # Compute daily composite
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days-i)).strftime("%Y-%m-%d")

            weighted_sum = 0
            total_weight = 0

            for theme in self.themes.values():
                if i < len(theme.daily_readings):
                    reading = theme.daily_readings[i]
                    weighted_sum += reading["sentiment"] * theme.weight
                    total_weight += theme.weight

            if total_weight > 0:
                composite = weighted_sum / total_weight
            else:
                composite = 0

            self.composite_readings.append({
                "date": date,
                "composite": composite,
                "components": {
                    name: theme.daily_readings[i]["sentiment"] if i < len(theme.daily_readings) else 0
                    for name, theme in self.themes.items()
                }
            })

        return self.composite_readings

    def normalize_to_index(self, base_value=100, base_date=None):
        """Normalize composite to index with base value."""
        if not self.composite_readings:
            return []

        # Find base reading
        base_sentiment = 0
        if base_date:
            for reading in self.composite_readings:
                if reading["date"] == base_date:
                    base_sentiment = reading["composite"]
                    break
        else:
            # Use average as base
            base_sentiment = statistics.mean([r["composite"] for r in self.composite_readings])

        # Normalize: convert sentiment (-1 to 1) to index (base_value ± range)
        normalized = []
        for reading in self.composite_readings:
            # Map sentiment to index
            index_value = base_value + (reading["composite"] - base_sentiment) * base_value

            normalized.append({
                "date": reading["date"],
                "index": index_value,
                "raw_sentiment": reading["composite"]
            })

        return normalized

    def calculate_momentum(self, window=14):
        """Calculate momentum (rate of change) of the index."""
        if len(self.composite_readings) < window:
            return []

        momentum = []
        for i in range(window, len(self.composite_readings)):
            current = self.composite_readings[i]["composite"]
            prior = self.composite_readings[i - window]["composite"]

            if prior != 0:
                mom = (current - prior) / abs(prior) * 100
            else:
                mom = 0

            momentum.append({
                "date": self.composite_readings[i]["date"],
                "momentum": mom,
                "direction": "improving" if mom > 0 else "deteriorating"
            })

        return momentum

    def detect_regime(self):
        """Detect economic regime (expansion, contraction, transition)."""
        if len(self.composite_readings) < 30:
            return "unknown"

        recent = [r["composite"] for r in self.composite_readings[-14:]]
        prior = [r["composite"] for r in self.composite_readings[-30:-14]]

        recent_avg = statistics.mean(recent)
        prior_avg = statistics.mean(prior)
        recent_trend = recent[-1] - recent[0]

        if recent_avg > 0.1 and recent_trend > 0:
            return "expansion"
        elif recent_avg < -0.1 and recent_trend < 0:
            return "contraction"
        elif prior_avg > 0 > recent_avg:
            return "transition_down"
        elif prior_avg < 0 < recent_avg:
            return "transition_up"
        else:
            return "stable"

    def calculate_recession_probability(self):
        """Estimate recession probability from sentiment indicators."""
        if len(self.composite_readings) < 60:
            return 0.5

        # Factors for recession model
        factors = {}

        # 1. Current sentiment level
        current = self.composite_readings[-1]["composite"]
        factors["sentiment_level"] = 1 / (1 + math.exp(current * 5))  # Sigmoid

        # 2. Sentiment trend (declining = higher probability)
        recent_30 = [r["composite"] for r in self.composite_readings[-30:]]
        trend = (recent_30[-1] - recent_30[0]) / 30
        factors["sentiment_trend"] = 1 / (1 + math.exp(trend * 100))

        # 3. Employment theme (key leading indicator)
        if "employment" in self.themes:
            emp_readings = self.themes["employment"].daily_readings[-14:]
            emp_avg = statistics.mean([r["sentiment"] for r in emp_readings])
            factors["employment"] = 1 / (1 + math.exp(emp_avg * 5))
        else:
            factors["employment"] = 0.5

        # 4. Consumer confidence
        if "consumer_confidence" in self.themes:
            cons_readings = self.themes["consumer_confidence"].daily_readings[-14:]
            cons_avg = statistics.mean([r["sentiment"] for r in cons_readings])
            factors["consumer"] = 1 / (1 + math.exp(cons_avg * 5))
        else:
            factors["consumer"] = 0.5

        # Weighted combination
        weights = {
            "sentiment_level": 0.25,
            "sentiment_trend": 0.30,
            "employment": 0.25,
            "consumer": 0.20
        }

        probability = sum(factors[k] * weights[k] for k in factors)

        return {
            "probability": probability,
            "factors": factors,
            "assessment": "elevated" if probability > 0.5 else "low" if probability < 0.3 else "moderate"
        }

    def build_sector_indicators(self, days=30):
        """Build sector-level sentiment indicators."""
        print("\nBuilding sector indicators...")

        for sector, keywords in self.SECTOR_THEMES.items():
            print(f"  Processing {sector}...")
            sector_data = []

            for i in range(days, 0, -1):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

                # Positive sentiment
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "title": ",".join(keywords),
                    "sentiment.overall.polarity": "positive",
                    "source.country": self.country,
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                positive = resp.json().get("total_results", 0)

                # Negative sentiment
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "title": ",".join(keywords),
                    "sentiment.overall.polarity": "negative",
                    "source.country": self.country,
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                negative = resp.json().get("total_results", 0)

                total = positive + negative
                sentiment = (positive - negative) / total if total > 0 else 0

                sector_data.append({
                    "date": date,
                    "sentiment": sentiment,
                    "volume": total
                })

            self.sector_data[sector] = sector_data

        return self.sector_data

    def get_sector_rotation_signals(self):
        """Generate sector rotation signals based on momentum."""
        if not self.sector_data:
            return []

        signals = []

        for sector, data in self.sector_data.items():
            if len(data) < 14:
                continue

            recent = statistics.mean([d["sentiment"] for d in data[-7:]])
            prior = statistics.mean([d["sentiment"] for d in data[-14:-7]])

            momentum = recent - prior
            trend = "improving" if momentum > 0.05 else "deteriorating" if momentum < -0.05 else "stable"

            signals.append({
                "sector": sector,
                "current_sentiment": recent,
                "momentum": momentum,
                "trend": trend,
                "signal": "OVERWEIGHT" if momentum > 0.1 else "UNDERWEIGHT" if momentum < -0.1 else "NEUTRAL"
            })

        return sorted(signals, key=lambda x: x["momentum"], reverse=True)


class InflationExpectations:
    """Track inflation expectations from news sentiment."""

    INFLATION_KEYWORDS = {
        "rising": ["inflation rising", "prices increase", "cost surge", "inflation concerns", "price hikes"],
        "falling": ["inflation easing", "prices fall", "deflation", "price cuts", "inflation cooling"],
        "stable": ["price stability", "inflation contained", "stable prices"]
    }

    def __init__(self):
        self.daily_readings = []

    def fetch_data(self, days=60, country="us"):
        """Fetch inflation expectation data."""
        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            readings = {}
            for direction, keywords in self.INFLATION_KEYWORDS.items():
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "title": ",".join(keywords),
                    "source.country": country,
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                readings[direction] = resp.json().get("total_results", 0)

            total = sum(readings.values())
            if total > 0:
                expectation = (readings["rising"] - readings["falling"]) / total
            else:
                expectation = 0

            self.daily_readings.append({
                "date": date,
                "expectation": expectation,
                "rising_mentions": readings["rising"],
                "falling_mentions": readings["falling"],
                "stable_mentions": readings["stable"]
            })

        return self.daily_readings

    def get_current_expectation(self):
        """Get current inflation expectation assessment."""
        if len(self.daily_readings) < 7:
            return "unknown"

        recent = statistics.mean([r["expectation"] for r in self.daily_readings[-7:]])

        if recent > 0.3:
            return "high_inflation_expected"
        elif recent > 0.1:
            return "moderate_inflation_expected"
        elif recent < -0.1:
            return "disinflation_expected"
        else:
            return "stable_prices_expected"


# Build economic indicators
print("ECONOMIC SENTIMENT INDICATORS")
print("=" * 70)

index = EconomicSentimentIndex(country="us")

# Build composite index
index.build_composite_index(days=60)

# Normalize to index
normalized = index.normalize_to_index(base_value=100)

print("\n" + "=" * 70)
print("COMPOSITE ECONOMIC SENTIMENT INDEX")
print("-" * 50)
print(f"{'Date':<12} {'Index':>8} {'Change':>8}")
print("-" * 50)

for reading in normalized[-10:]:
    print(f"{reading['date']:<12} {reading['index']:>8.1f}")

# Current reading
latest = normalized[-1]
print(f"\nLatest Reading: {latest['index']:.1f}")
print(f"Raw Sentiment: {latest['raw_sentiment']:.3f}")

# Momentum
print("\n" + "=" * 70)
print("MOMENTUM ANALYSIS")
print("-" * 50)
momentum = index.calculate_momentum(window=14)
if momentum:
    latest_mom = momentum[-1]
    print(f"14-day Momentum: {latest_mom['momentum']:+.1f}%")
    print(f"Direction: {latest_mom['direction'].upper()}")

# Regime detection
print("\n" + "=" * 70)
print("ECONOMIC REGIME")
print("-" * 50)
regime = index.detect_regime()
print(f"Current Regime: {regime.upper()}")

# Recession probability
print("\n" + "=" * 70)
print("RECESSION PROBABILITY MODEL")
print("-" * 50)
recession = index.calculate_recession_probability()
print(f"Probability: {recession['probability']:.1%}")
print(f"Assessment: {recession['assessment'].upper()}")
print("\nFactor Contributions:")
for factor, value in recession["factors"].items():
    print(f"  {factor}: {value:.2f}")

# Sector rotation
print("\n" + "=" * 70)
print("SECTOR ROTATION SIGNALS")
print("-" * 50)
index.build_sector_indicators(days=30)
signals = index.get_sector_rotation_signals()

print(f"{'Sector':<15} {'Sentiment':>10} {'Momentum':>10} {'Signal':>12}")
print("-" * 50)
for s in signals:
    print(f"{s['sector']:<15} {s['current_sentiment']:>10.3f} {s['momentum']:>+10.3f} {s['signal']:>12}")

# Inflation expectations
print("\n" + "=" * 70)
print("INFLATION EXPECTATIONS")
print("-" * 50)
inflation = InflationExpectations()
inflation.fetch_data(days=30)
expectation = inflation.get_current_expectation()
print(f"Current Assessment: {expectation.upper()}")
```

## Indicator Components

### Composite Economic Sentiment Index
| Theme | Description | Weight |
|-------|-------------|--------|
| Consumer Confidence | Retail spending, shopping sentiment | 20% |
| Employment | Hiring, layoffs, labor market | 20% |
| Business Investment | CapEx, expansion, investment | 15% |
| Manufacturing | Factory orders, industrial production | 15% |
| Housing | Home sales, construction, mortgages | 10% |
| Inflation | Price pressures, cost trends | 10% |
| Financial Conditions | Credit, lending, market stress | 10% |

### Sector Indicators
Technology, Healthcare, Financials, Energy, Consumer, Industrials, Materials, Real Estate

## Common Use Cases

- **Economic forecasting** — build leading indicators of economic activity.
- **Macro trading** — generate signals for macro strategies.
- **Policy analysis** — track sentiment around policy changes.
- **Sector allocation** — inform sector rotation decisions.
- **Risk assessment** — monitor recession indicators.
- **Research** — supplement traditional economic data.

## See Also

- [examples.md](./examples.md) — additional code examples.
