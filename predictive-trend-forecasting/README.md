# Predictive Trend Forecasting

Workflow for forecasting news coverage trends, topic evolution, and content dynamics using time series models and machine learning with the [APITube News API](https://apitube.io).

## Overview

The **Predictive Trend Forecasting** workflow implements comprehensive forecasting capabilities for news-driven metrics. Features include time series decomposition (trend, seasonality, residuals), ARIMA-style forecasting, exponential smoothing, breakout detection with confidence intervals, multi-topic portfolio optimization, and cyclical pattern analysis using spectral methods. Combines statistical forecasting with real-time news signals. Ideal for content planning, editorial calendars, trend-driven trading, marketing timing, and any application requiring forward-looking news intelligence.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `topic.id`                    | string  | Filter by topic ID.                                                  |
| `organization.name`           | string  | Filter by organization name.                                         |
| `category.id`                 | string  | Filter by category ID.                                               |
| `title`                       | string  | Filter by keywords in title.                                         |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### Python

```python
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


@dataclass
class ForecastResult:
    """Represents a forecast output."""
    timestamp: datetime
    point_forecast: float
    lower_bound: float
    upper_bound: float
    confidence: float


class TrendForecaster:
    """
    Time series forecasting for news trends.
    Implements decomposition, smoothing, and prediction.
    """

    def __init__(self, topic: str):
        self.topic = topic
        self.history = []
        self.model_params = {}

    def fetch_historical_data(self, days: int = 30) -> List[Dict]:
        """Fetch historical daily data for the topic."""
        data = []

        for d in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=d-1)).strftime("%Y-%m-%d")

            params = {
                "api_key": API_KEY,
                "topic.id": self.topic,
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code.eq": "en",
                "per_page": 100,
            }

            response = requests.get(BASE_URL, params=params)
            articles = response.json().get("results", [])

            # Calculate daily metrics
            volume = len(articles)

            sentiments = [
                a.get("sentiment", {}).get("overall", {}).get("score", 0)
                for a in articles
            ]
            avg_sentiment = np.mean(sentiments) if sentiments else 0

            high_authority = sum(
                1 for a in articles
                if a.get("source", {}).get("rankings", {}).get("opr", 0) >= 5
            )

            data.append({
                "date": date,
                "volume": volume,
                "sentiment": avg_sentiment,
                "high_authority_ratio": high_authority / max(volume, 1),
                "day_of_week": datetime.strptime(date, "%Y-%m-%d").weekday()
            })

        self.history = data
        return data

    def decompose_time_series(self, values: List[float], period: int = 7) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.
        Uses simple moving average decomposition.
        """
        n = len(values)
        if n < period * 2:
            return {"trend": values, "seasonal": [0] * n, "residual": [0] * n}

        values = np.array(values)

        # Trend: centered moving average
        trend = np.convolve(values, np.ones(period) / period, mode='same')

        # Seasonal: average by position in period
        detrended = values - trend
        seasonal = np.zeros(n)
        for i in range(period):
            indices = list(range(i, n, period))
            seasonal_value = np.mean(detrended[indices])
            for idx in indices:
                seasonal[idx] = seasonal_value

        # Residual
        residual = values - trend - seasonal

        return {
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist(),
            "period": period
        }

    def exponential_smoothing(self, values: List[float], alpha: float = 0.3,
                              beta: float = 0.1) -> Tuple[List[float], Dict]:
        """
        Double exponential smoothing (Holt's method).
        Captures level and trend.
        """
        n = len(values)
        if n < 3:
            return values, {"alpha": alpha, "beta": beta}

        values = np.array(values)

        # Initialize
        level = values[0]
        trend = values[1] - values[0]

        smoothed = [level]
        levels = [level]
        trends = [trend]

        for i in range(1, n):
            last_level = level
            level = alpha * values[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend

            smoothed.append(level)
            levels.append(level)
            trends.append(trend)

        self.model_params = {
            "alpha": alpha,
            "beta": beta,
            "final_level": level,
            "final_trend": trend
        }

        return smoothed, self.model_params

    def forecast(self, horizon: int = 7, confidence: float = 0.9) -> List[ForecastResult]:
        """
        Generate forecasts with prediction intervals.
        """
        if not self.history:
            self.fetch_historical_data()

        volumes = [d["volume"] for d in self.history]

        # Apply smoothing
        smoothed, params = self.exponential_smoothing(volumes)

        # Calculate residual variance for prediction intervals
        residuals = np.array(volumes) - np.array(smoothed)
        residual_std = np.std(residuals)

        # Generate forecasts
        level = params["final_level"]
        trend = params["final_trend"]

        forecasts = []
        z_score = 1.645 if confidence == 0.9 else 1.96  # 90% or 95% CI

        for h in range(1, horizon + 1):
            point = level + h * trend

            # Prediction interval widens with horizon
            interval_width = z_score * residual_std * np.sqrt(1 + h * 0.1)

            forecast_time = datetime.utcnow() + timedelta(days=h)

            forecasts.append(ForecastResult(
                timestamp=forecast_time,
                point_forecast=max(0, point),
                lower_bound=max(0, point - interval_width),
                upper_bound=point + interval_width,
                confidence=confidence
            ))

        return forecasts

    def detect_trend_change(self, window: int = 7, threshold: float = 2.0) -> Optional[Dict]:
        """
        Detect significant trend changes using CUSUM-like method.
        """
        if not self.history or len(self.history) < window * 2:
            return None

        volumes = [d["volume"] for d in self.history]

        # Calculate rolling statistics
        recent = np.mean(volumes[-window:])
        historical = np.mean(volumes[:-window])
        historical_std = np.std(volumes[:-window])

        if historical_std == 0:
            return None

        z_score = (recent - historical) / historical_std

        if abs(z_score) < threshold:
            return None

        return {
            "detected": True,
            "direction": "increasing" if z_score > 0 else "decreasing",
            "z_score": round(z_score, 2),
            "recent_avg": round(recent, 1),
            "historical_avg": round(historical, 1),
            "change_percent": round((recent - historical) / historical * 100, 1)
        }

    def seasonal_adjustment(self) -> List[Dict]:
        """
        Calculate seasonal factors (day-of-week effects).
        """
        if not self.history:
            self.fetch_historical_data()

        # Group by day of week
        day_volumes = {i: [] for i in range(7)}

        for d in self.history:
            day_volumes[d["day_of_week"]].append(d["volume"])

        overall_mean = np.mean([d["volume"] for d in self.history])

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]

        factors = []
        for i in range(7):
            if day_volumes[i]:
                day_mean = np.mean(day_volumes[i])
                factor = day_mean / overall_mean if overall_mean > 0 else 1.0
            else:
                factor = 1.0

            factors.append({
                "day": day_names[i],
                "factor": round(factor, 3),
                "interpretation": "above_average" if factor > 1.1 else "below_average" if factor < 0.9 else "average"
            })

        return factors

    def full_analysis(self, forecast_days: int = 7) -> Dict:
        """Run complete trend analysis and forecasting."""
        print(f"Analyzing trend for topic: {self.topic}")

        # Fetch data
        print("  Fetching historical data...")
        self.fetch_historical_data(days=30)
        print(f"  Collected {len(self.history)} days of data")

        volumes = [d["volume"] for d in self.history]

        # Decomposition
        print("  Decomposing time series...")
        decomposition = self.decompose_time_series(volumes)

        # Detect trend changes
        print("  Detecting trend changes...")
        trend_change = self.detect_trend_change()

        # Seasonal factors
        print("  Calculating seasonal factors...")
        seasonal_factors = self.seasonal_adjustment()

        # Generate forecast
        print(f"  Generating {forecast_days}-day forecast...")
        forecasts = self.forecast(horizon=forecast_days)

        # Summary statistics
        recent_trend = np.polyfit(range(7), volumes[-7:], 1)[0]
        trend_direction = "increasing" if recent_trend > 0.5 else "decreasing" if recent_trend < -0.5 else "stable"

        return {
            "topic": self.topic,
            "analysis_date": datetime.utcnow().isoformat(),
            "data_points": len(self.history),
            "summary": {
                "current_volume": volumes[-1],
                "avg_volume_7d": round(np.mean(volumes[-7:]), 1),
                "avg_volume_30d": round(np.mean(volumes), 1),
                "trend_direction": trend_direction,
                "volatility": round(np.std(volumes) / np.mean(volumes), 3) if np.mean(volumes) > 0 else 0
            },
            "trend_change": trend_change,
            "seasonal_factors": seasonal_factors,
            "decomposition": {
                "trend": decomposition["trend"][-7:],
                "seasonal_amplitude": round(max(decomposition["seasonal"]) - min(decomposition["seasonal"]), 2)
            },
            "forecast": [
                {
                    "date": f.timestamp.strftime("%Y-%m-%d"),
                    "point": round(f.point_forecast, 1),
                    "lower": round(f.lower_bound, 1),
                    "upper": round(f.upper_bound, 1)
                }
                for f in forecasts
            ],
            "model_params": self.model_params,
            "recommendations": self._generate_recommendations(trend_change, seasonal_factors, trend_direction)
        }

    def _generate_recommendations(self, trend_change: Optional[Dict],
                                   seasonal: List[Dict], direction: str) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        # Trend-based recommendations
        if direction == "increasing":
            recs.append("Coverage is trending up - consider increasing content production")
        elif direction == "decreasing":
            recs.append("Coverage is declining - may indicate topic saturation")

        # Trend change alerts
        if trend_change:
            recs.append(f"Significant trend change detected: {trend_change['direction']} "
                       f"({trend_change['change_percent']:+.1f}%)")

        # Seasonal recommendations
        peak_day = max(seasonal, key=lambda x: x["factor"])
        trough_day = min(seasonal, key=lambda x: x["factor"])

        recs.append(f"Best day for publishing: {peak_day['day']} "
                   f"(factor: {peak_day['factor']:.2f})")
        recs.append(f"Lowest coverage day: {trough_day['day']} "
                   f"(factor: {trough_day['factor']:.2f})")

        return recs


# Run forecasting
print("PREDICTIVE TREND FORECASTING")
print("=" * 70)

forecaster = TrendForecaster(topic="artificial_intelligence")

results = forecaster.full_analysis(forecast_days=7)

print("\n" + "=" * 70)
print("ANALYSIS RESULTS")
print("-" * 50)
print(f"Topic: {results['topic']}")
print(f"\nSUMMARY:")
print(f"  Current Volume: {results['summary']['current_volume']}")
print(f"  7-day Average: {results['summary']['avg_volume_7d']}")
print(f"  30-day Average: {results['summary']['avg_volume_30d']}")
print(f"  Trend: {results['summary']['trend_direction']}")
print(f"  Volatility: {results['summary']['volatility']:.3f}")

if results['trend_change']:
    print(f"\nTREND CHANGE DETECTED:")
    print(f"  Direction: {results['trend_change']['direction']}")
    print(f"  Change: {results['trend_change']['change_percent']:+.1f}%")

print("\nSEASONAL FACTORS:")
for sf in results['seasonal_factors']:
    bar = "+" * int(sf['factor'] * 5) if sf['factor'] >= 1 else "-" * int((1 - sf['factor']) * 10)
    print(f"  {sf['day']:<10} {sf['factor']:.3f} {bar}")

print("\n7-DAY FORECAST:")
print("-" * 50)
for f in results['forecast']:
    print(f"  {f['date']}: {f['point']:.0f} [{f['lower']:.0f} - {f['upper']:.0f}]")

print("\nRECOMMENDATIONS:")
for rec in results['recommendations']:
    print(f"  - {rec}")
```

## Forecasting Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| **Exponential Smoothing** | Double smoothing (Holt's method) | alpha (level), beta (trend) |
| **Decomposition** | Trend + Seasonal + Residual | period (default: 7 days) |
| **Trend Detection** | CUSUM-based change detection | threshold (z-score) |
| **Seasonal Adjustment** | Day-of-week factors | window size |

## Output Components

| Output | Description |
|--------|-------------|
| **Point Forecast** | Expected value for each future period |
| **Prediction Interval** | Lower/upper bounds at specified confidence |
| **Trend Direction** | Increasing, decreasing, or stable |
| **Seasonal Factors** | Day-of-week multipliers |
| **Trend Change Alert** | Detection of significant shifts |

## Common Use Cases

- **Content planning** — optimize publication timing based on coverage patterns.
- **Editorial calendars** — align content production with forecasted trends.
- **Marketing timing** — schedule campaigns during high-coverage periods.
- **Trend trading** — generate signals from coverage momentum.
- **Resource allocation** — plan staffing around expected coverage volume.
- **Competitive monitoring** — forecast competitor coverage trajectories.

## See Also

- [examples.md](./examples.md) — detailed code examples including multi-topic portfolio forecasting, breakout detection, and cyclical pattern analysis.
