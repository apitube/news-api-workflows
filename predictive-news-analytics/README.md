# Predictive News Analytics

Workflow for forecasting news coverage trends, detecting emerging patterns before they peak, building early warning systems, and identifying anomalies using statistical analysis with the [APITube News API](https://apitube.io).

## Overview

The **Predictive News Analytics** workflow applies statistical forecasting techniques to news data, enabling prediction of coverage spikes, detection of emerging trends before mainstream attention, identification of anomalous patterns, and construction of early warning indicators. Combines moving averages, z-score analysis, momentum indicators, and pattern matching to surface predictive signals. Ideal for trading desks, strategic planning, risk management, and competitive intelligence.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by organization for trend analysis.                           |
| `person.name`                 | string  | Filter by person for trend analysis.                                 |
| `topic.id`                    | string  | Filter by topic for category trends.                                 |
| `title`                       | string  | Filter by keywords for signal detection.                             |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Get historical coverage for trend analysis
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Tesla&published_at.start=2024-01-01&published_at.end=2024-01-31&language.code=en&per_page=100" | jq '.results | length'

# Monitor emerging topics
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=breakthrough,revolutionary,first-ever&source.rank.opr.min=6&language.code=en&per_page=20"

# Detect sudden coverage spikes
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=OpenAI&published_at.start=$(date -v-1d +%Y-%m-%d)&language.code=en&per_page=100" | jq '.results | length'
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class PredictiveAnalyzer:
    """Statistical forecasting for news coverage."""

    def __init__(self, entity, lookback_days=90):
        self.entity = entity
        self.lookback_days = lookback_days
        self.daily_counts = []
        self.sentiment_series = []

    def build_historical_series(self):
        """Build time series of daily coverage."""
        for i in range(self.lookback_days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            # Total coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": self.entity,
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 100,
            })
            count = len(resp.json().get("results", []))

            # Sentiment
            neg_resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": self.entity,
                "sentiment.overall.polarity": "negative",
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 100,
            })
            neg_count = len(neg_resp.json().get("results", []))

            self.daily_counts.append({"date": date, "count": count})
            sentiment_ratio = neg_count / max(count, 1)
            self.sentiment_series.append({"date": date, "negativity": sentiment_ratio})

    def calculate_moving_averages(self, window_short=7, window_long=21):
        """Calculate short and long moving averages."""
        counts = [d["count"] for d in self.daily_counts]

        ma_short = []
        ma_long = []

        for i in range(len(counts)):
            if i >= window_short - 1:
                ma_short.append(statistics.mean(counts[i-window_short+1:i+1]))
            else:
                ma_short.append(None)

            if i >= window_long - 1:
                ma_long.append(statistics.mean(counts[i-window_long+1:i+1]))
            else:
                ma_long.append(None)

        return ma_short, ma_long

    def detect_anomalies(self, threshold=2.0):
        """Detect anomalies using z-score."""
        counts = [d["count"] for d in self.daily_counts]

        if len(counts) < 30:
            return []

        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) or 1

        anomalies = []
        for i, point in enumerate(self.daily_counts):
            z_score = (point["count"] - mean) / stdev
            if abs(z_score) > threshold:
                anomalies.append({
                    "date": point["date"],
                    "count": point["count"],
                    "z_score": z_score,
                    "type": "spike" if z_score > 0 else "drop"
                })

        return anomalies

    def calculate_momentum(self, period=14):
        """Calculate momentum indicator (rate of change)."""
        counts = [d["count"] for d in self.daily_counts]
        momentum = []

        for i in range(len(counts)):
            if i >= period:
                prev = counts[i - period] or 1
                roc = ((counts[i] - prev) / prev) * 100
                momentum.append({"date": self.daily_counts[i]["date"], "momentum": roc})
            else:
                momentum.append({"date": self.daily_counts[i]["date"], "momentum": None})

        return momentum

    def forecast_next_period(self, days=7):
        """Simple linear regression forecast."""
        counts = [d["count"] for d in self.daily_counts[-30:]]
        n = len(counts)

        # Linear regression
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(counts)

        numerator = sum((i - x_mean) * (counts[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator else 0
        intercept = y_mean - slope * x_mean

        forecasts = []
        for i in range(1, days + 1):
            future_date = (datetime.utcnow() + timedelta(days=i)).strftime("%Y-%m-%d")
            predicted = intercept + slope * (n + i - 1)
            forecasts.append({
                "date": future_date,
                "predicted_count": max(0, int(predicted)),
                "confidence": "medium"
            })

        return forecasts, slope

    def detect_trend_reversal(self):
        """Detect potential trend reversals."""
        ma_short, ma_long = self.calculate_moving_averages()

        signals = []
        for i in range(1, len(ma_short)):
            if ma_short[i] is None or ma_long[i] is None:
                continue
            if ma_short[i-1] is None or ma_long[i-1] is None:
                continue

            # Golden cross (bullish)
            if ma_short[i-1] <= ma_long[i-1] and ma_short[i] > ma_long[i]:
                signals.append({
                    "date": self.daily_counts[i]["date"],
                    "signal": "golden_cross",
                    "interpretation": "Coverage likely to increase"
                })

            # Death cross (bearish)
            if ma_short[i-1] >= ma_long[i-1] and ma_short[i] < ma_long[i]:
                signals.append({
                    "date": self.daily_counts[i]["date"],
                    "signal": "death_cross",
                    "interpretation": "Coverage likely to decrease"
                })

        return signals

    def generate_early_warning(self):
        """Generate early warning indicators."""
        warnings = []

        # Check momentum
        momentum = self.calculate_momentum()
        recent_momentum = [m["momentum"] for m in momentum[-7:] if m["momentum"] is not None]
        if recent_momentum and statistics.mean(recent_momentum) > 50:
            warnings.append({
                "type": "momentum_surge",
                "severity": "high",
                "message": f"Coverage momentum +{statistics.mean(recent_momentum):.0f}% over 14 days"
            })

        # Check sentiment deterioration
        recent_sentiment = [s["negativity"] for s in self.sentiment_series[-7:]]
        prior_sentiment = [s["negativity"] for s in self.sentiment_series[-14:-7]]

        if recent_sentiment and prior_sentiment:
            if statistics.mean(recent_sentiment) > statistics.mean(prior_sentiment) * 1.5:
                warnings.append({
                    "type": "sentiment_deterioration",
                    "severity": "high",
                    "message": "Negative sentiment increasing significantly"
                })

        # Check for anomaly clustering
        anomalies = self.detect_anomalies()
        recent_anomalies = [a for a in anomalies if a["date"] >= (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")]
        if len(recent_anomalies) >= 2:
            warnings.append({
                "type": "anomaly_cluster",
                "severity": "medium",
                "message": f"{len(recent_anomalies)} anomalies detected in past 7 days"
            })

        return warnings

# Run analysis
analyzer = PredictiveAnalyzer("Tesla", lookback_days=60)
print("Building historical time series...")
analyzer.build_historical_series()

print("\nPREDICTIVE NEWS ANALYTICS")
print("=" * 70)
print(f"Entity: {analyzer.entity}")
print(f"Analysis Period: {analyzer.lookback_days} days\n")

# Anomalies
anomalies = analyzer.detect_anomalies()
print(f"ANOMALIES DETECTED: {len(anomalies)}")
for a in anomalies[-5:]:
    print(f"  {a['date']}: {a['count']} articles (z={a['z_score']:+.2f}, {a['type']})")

# Trend reversals
signals = analyzer.detect_trend_reversal()
print(f"\nTREND SIGNALS: {len(signals)}")
for s in signals[-3:]:
    print(f"  {s['date']}: {s['signal']} - {s['interpretation']}")

# Forecast
forecasts, trend = analyzer.forecast_next_period(7)
print(f"\n7-DAY FORECAST (trend: {trend:+.2f}/day):")
for f in forecasts:
    print(f"  {f['date']}: ~{f['predicted_count']} articles")

# Early warnings
warnings = analyzer.generate_early_warning()
print(f"\nEARLY WARNINGS: {len(warnings)}")
for w in warnings:
    print(f"  [{w['severity'].upper()}] {w['type']}: {w['message']}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class PredictiveAnalyzer {
  constructor(entity, lookbackDays = 60) {
    this.entity = entity;
    this.lookbackDays = lookbackDays;
    this.dailyCounts = [];
  }

  async buildHistoricalSeries() {
    for (let i = this.lookbackDays; i > 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
      const nextDate = new Date(Date.now() - (i - 1) * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": this.entity,
        "published_at.start": date,
        "published_at.end": nextDate,
        "language.code": "en",
        per_page: "100",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      this.dailyCounts.push({ date, count: (data.results || []).length });
    }
  }

  calculateMovingAverage(window) {
    const counts = this.dailyCounts.map((d) => d.count);
    const ma = [];

    for (let i = 0; i < counts.length; i++) {
      if (i >= window - 1) {
        const slice = counts.slice(i - window + 1, i + 1);
        ma.push(slice.reduce((a, b) => a + b, 0) / window);
      } else {
        ma.push(null);
      }
    }
    return ma;
  }

  detectAnomalies(threshold = 2.0) {
    const counts = this.dailyCounts.map((d) => d.count);
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
    const variance = counts.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / counts.length;
    const stdev = Math.sqrt(variance) || 1;

    const anomalies = [];
    this.dailyCounts.forEach((point, i) => {
      const zScore = (point.count - mean) / stdev;
      if (Math.abs(zScore) > threshold) {
        anomalies.push({
          date: point.date,
          count: point.count,
          zScore,
          type: zScore > 0 ? "spike" : "drop",
        });
      }
    });

    return anomalies;
  }

  forecastNextPeriod(days = 7) {
    const counts = this.dailyCounts.slice(-30).map((d) => d.count);
    const n = counts.length;

    const xMean = (n - 1) / 2;
    const yMean = counts.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      numerator += (i - xMean) * (counts[i] - yMean);
      denominator += Math.pow(i - xMean, 2);
    }

    const slope = denominator ? numerator / denominator : 0;
    const intercept = yMean - slope * xMean;

    const forecasts = [];
    for (let i = 1; i <= days; i++) {
      const futureDate = new Date(Date.now() + i * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
      const predicted = intercept + slope * (n + i - 1);
      forecasts.push({
        date: futureDate,
        predictedCount: Math.max(0, Math.round(predicted)),
      });
    }

    return { forecasts, trend: slope };
  }
}

async function runAnalysis() {
  const analyzer = new PredictiveAnalyzer("Tesla", 30);

  console.log("Building historical series...");
  await analyzer.buildHistoricalSeries();

  console.log("\nPREDICTIVE NEWS ANALYTICS");
  console.log("=".repeat(50));

  const anomalies = analyzer.detectAnomalies();
  console.log(`\nAnomalies: ${anomalies.length}`);
  anomalies.slice(-3).forEach((a) => {
    console.log(`  ${a.date}: ${a.count} (z=${a.zScore.toFixed(2)}, ${a.type})`);
  });

  const { forecasts, trend } = analyzer.forecastNextPeriod(7);
  console.log(`\nForecast (trend: ${trend.toFixed(2)}/day):`);
  forecasts.forEach((f) => {
    console.log(`  ${f.date}: ~${f.predictedCount} articles`);
  });
}

runAnalysis();
```

### PHP

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class PredictiveAnalyzer
{
    private string $apiKey;
    private string $baseUrl;
    private string $entity;
    private int $lookbackDays;
    private array $dailyCounts = [];

    public function __construct(string $entity, int $lookbackDays = 60)
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
        $this->entity = $entity;
        $this->lookbackDays = $lookbackDays;
    }

    public function buildHistoricalSeries(): void
    {
        for ($i = $this->lookbackDays; $i > 0; $i--) {
            $date = (new DateTime("-{$i} days"))->format("Y-m-d");
            $nextDate = (new DateTime("-" . ($i - 1) . " days"))->format("Y-m-d");

            $query = http_build_query([
                "api_key" => $this->apiKey,
                "organization.name" => $this->entity,
                "published_at.start" => $date,
                "published_at.end" => $nextDate,
                "language.code" => "en",
                "per_page" => 100,
            ]);

            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $this->dailyCounts[] = ["date" => $date, "count" => count($data["results"] ?? [])];
        }
    }

    public function detectAnomalies(float $threshold = 2.0): array
    {
        $counts = array_column($this->dailyCounts, "count");
        $mean = array_sum($counts) / count($counts);
        $variance = array_sum(array_map(fn($x) => pow($x - $mean, 2), $counts)) / count($counts);
        $stdev = sqrt($variance) ?: 1;

        $anomalies = [];
        foreach ($this->dailyCounts as $point) {
            $zScore = ($point["count"] - $mean) / $stdev;
            if (abs($zScore) > $threshold) {
                $anomalies[] = [
                    "date" => $point["date"],
                    "count" => $point["count"],
                    "z_score" => $zScore,
                    "type" => $zScore > 0 ? "spike" : "drop",
                ];
            }
        }

        return $anomalies;
    }

    public function forecastNextPeriod(int $days = 7): array
    {
        $counts = array_column(array_slice($this->dailyCounts, -30), "count");
        $n = count($counts);

        $xMean = ($n - 1) / 2;
        $yMean = array_sum($counts) / $n;

        $numerator = 0;
        $denominator = 0;

        for ($i = 0; $i < $n; $i++) {
            $numerator += ($i - $xMean) * ($counts[$i] - $yMean);
            $denominator += pow($i - $xMean, 2);
        }

        $slope = $denominator ? $numerator / $denominator : 0;
        $intercept = $yMean - $slope * $xMean;

        $forecasts = [];
        for ($i = 1; $i <= $days; $i++) {
            $futureDate = (new DateTime("+{$i} days"))->format("Y-m-d");
            $predicted = $intercept + $slope * ($n + $i - 1);
            $forecasts[] = [
                "date" => $futureDate,
                "predicted_count" => max(0, (int)$predicted),
            ];
        }

        return ["forecasts" => $forecasts, "trend" => $slope];
    }
}

$analyzer = new PredictiveAnalyzer("Tesla", 30);
echo "Building historical series...\n";
$analyzer->buildHistoricalSeries();

echo "\nPREDICTIVE NEWS ANALYTICS\n";
echo str_repeat("=", 50) . "\n";

$anomalies = $analyzer->detectAnomalies();
echo "\nAnomalies: " . count($anomalies) . "\n";
foreach (array_slice($anomalies, -3) as $a) {
    printf("  %s: %d (z=%.2f, %s)\n", $a["date"], $a["count"], $a["z_score"], $a["type"]);
}

$forecast = $analyzer->forecastNextPeriod(7);
printf("\nForecast (trend: %.2f/day):\n", $forecast["trend"]);
foreach ($forecast["forecasts"] as $f) {
    printf("  %s: ~%d articles\n", $f["date"], $f["predicted_count"]);
}
```

## Common Use Cases

- **Trading signal generation** — predict coverage spikes before they affect stock prices.
- **Campaign timing optimization** — identify low-coverage periods for announcements.
- **Risk early warning** — detect sentiment deterioration before crises emerge.
- **Competitive timing** — forecast competitor coverage patterns.
- **Resource planning** — predict PR team workload based on coverage forecasts.
- **Trend surfing** — identify emerging topics before mainstream attention.
- **Anomaly investigation** — automatically flag unusual coverage for review.
- **Seasonal pattern detection** — understand cyclical coverage patterns.

## See Also

- [examples.md](./examples.md) — detailed code examples for predictive news analytics.
