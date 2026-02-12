# Predictive News Analytics — Examples

Advanced code examples for statistical forecasting, anomaly detection, trend prediction, and early warning systems.

---

## Python — Advanced Time Series Forecasting System

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class AdvancedTimeSeriesForecaster:
    """
    Comprehensive time series forecasting with multiple algorithms,
    confidence intervals, and ensemble predictions.
    """

    def __init__(self):
        self.series_cache = {}

    def fetch_entity_series(self, entity, days=90):
        """Fetch daily coverage time series for an entity."""
        if entity in self.series_cache:
            return self.series_cache[entity]

        series = []
        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity,
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            count = resp.json().get("total_results", 0)
            series.append({"date": date, "count": count, "day_of_week": datetime.fromisoformat(date).weekday()})

        self.series_cache[entity] = series
        return series

    def exponential_smoothing(self, series, alpha=0.3):
        """Simple exponential smoothing."""
        counts = [d["count"] for d in series]
        smoothed = [counts[0]]

        for i in range(1, len(counts)):
            smoothed.append(alpha * counts[i] + (1 - alpha) * smoothed[i-1])

        return smoothed

    def double_exponential_smoothing(self, series, alpha=0.3, beta=0.1):
        """Holt's linear trend method."""
        counts = [d["count"] for d in series]
        n = len(counts)

        level = [counts[0]]
        trend = [counts[1] - counts[0] if n > 1 else 0]

        for i in range(1, n):
            level.append(alpha * counts[i] + (1 - alpha) * (level[i-1] + trend[i-1]))
            trend.append(beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1])

        return level, trend

    def seasonal_decomposition(self, series, period=7):
        """Decompose series into trend, seasonal, and residual components."""
        counts = [d["count"] for d in series]
        n = len(counts)

        # Calculate trend using centered moving average
        trend = [None] * n
        half = period // 2

        for i in range(half, n - half):
            window = counts[i-half:i+half+1]
            trend[i] = statistics.mean(window)

        # Calculate seasonal component
        seasonal = [0] * period
        for i in range(period):
            seasonal_values = []
            for j in range(i, n, period):
                if trend[j] is not None and trend[j] > 0:
                    seasonal_values.append(counts[j] / trend[j])
            if seasonal_values:
                seasonal[i] = statistics.mean(seasonal_values)

        # Normalize seasonal factors
        seasonal_mean = statistics.mean(seasonal) if seasonal else 1
        seasonal = [s / seasonal_mean if seasonal_mean else 1 for s in seasonal]

        # Calculate residual
        residual = []
        for i in range(n):
            if trend[i] is not None and seasonal[i % period] > 0:
                residual.append(counts[i] / (trend[i] * seasonal[i % period]))
            else:
                residual.append(None)

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "period": period
        }

    def forecast_with_confidence(self, series, horizon=14, confidence=0.95):
        """Generate forecasts with confidence intervals."""
        level, trend = self.double_exponential_smoothing(series)

        # Calculate residuals for confidence interval
        counts = [d["count"] for d in series]
        fitted = [level[i] + trend[i] for i in range(len(level))]
        residuals = [counts[i] - fitted[i] for i in range(len(counts))]
        residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0

        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        forecasts = []
        last_level = level[-1]
        last_trend = trend[-1]

        for h in range(1, horizon + 1):
            point_forecast = last_level + h * last_trend
            margin = z * residual_std * (h ** 0.5)  # Widening confidence interval

            forecasts.append({
                "date": (datetime.utcnow() + timedelta(days=h)).strftime("%Y-%m-%d"),
                "forecast": max(0, point_forecast),
                "lower": max(0, point_forecast - margin),
                "upper": point_forecast + margin,
                "confidence": confidence
            })

        return forecasts

    def detect_changepoints(self, series, min_segment=7):
        """Detect structural breaks in the time series."""
        counts = [d["count"] for d in series]
        n = len(counts)
        changepoints = []

        if n < min_segment * 2:
            return changepoints

        # Sliding window comparison
        for i in range(min_segment, n - min_segment):
            before = counts[i-min_segment:i]
            after = counts[i:i+min_segment]

            before_mean = statistics.mean(before)
            after_mean = statistics.mean(after)
            before_std = statistics.stdev(before) if len(before) > 1 else 1
            after_std = statistics.stdev(after) if len(after) > 1 else 1

            # T-test approximation
            pooled_std = ((before_std**2 + after_std**2) / 2) ** 0.5
            t_stat = abs(after_mean - before_mean) / (pooled_std * (2/min_segment)**0.5) if pooled_std else 0

            if t_stat > 2.5:  # Significant change
                changepoints.append({
                    "date": series[i]["date"],
                    "index": i,
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "change_pct": ((after_mean - before_mean) / before_mean * 100) if before_mean else 0,
                    "significance": t_stat
                })

        # Remove nearby duplicates
        filtered = []
        for cp in changepoints:
            if not filtered or cp["index"] - filtered[-1]["index"] >= min_segment:
                filtered.append(cp)

        return filtered

    def ensemble_forecast(self, entity, horizon=7):
        """Combine multiple forecasting methods."""
        series = self.fetch_entity_series(entity, days=90)

        # Method 1: Simple moving average
        counts = [d["count"] for d in series]
        ma_forecast = statistics.mean(counts[-7:])

        # Method 2: Exponential smoothing
        smoothed = self.exponential_smoothing(series, alpha=0.3)
        es_forecast = smoothed[-1]

        # Method 3: Linear trend
        level, trend = self.double_exponential_smoothing(series)
        lt_forecasts = [level[-1] + i * trend[-1] for i in range(1, horizon + 1)]

        # Method 4: Seasonal naive
        decomp = self.seasonal_decomposition(series)
        seasonal = decomp["seasonal"]
        base = statistics.mean(counts[-7:])

        # Ensemble (weighted average)
        weights = {"ma": 0.2, "es": 0.3, "lt": 0.3, "seasonal": 0.2}

        ensemble = []
        for h in range(horizon):
            day_of_week = (datetime.utcnow() + timedelta(days=h+1)).weekday()
            seasonal_factor = seasonal[day_of_week] if day_of_week < len(seasonal) else 1

            combined = (
                weights["ma"] * ma_forecast +
                weights["es"] * es_forecast +
                weights["lt"] * lt_forecasts[h] +
                weights["seasonal"] * (base * seasonal_factor)
            )

            ensemble.append({
                "date": (datetime.utcnow() + timedelta(days=h+1)).strftime("%Y-%m-%d"),
                "ensemble_forecast": max(0, combined),
                "ma_component": ma_forecast,
                "es_component": es_forecast,
                "lt_component": lt_forecasts[h],
                "seasonal_component": base * seasonal_factor
            })

        return ensemble


class MultiEntityCorrelationAnalyzer:
    """Analyze correlations between entities for predictive signals."""

    def __init__(self, forecaster):
        self.forecaster = forecaster

    def calculate_correlation(self, series1, series2):
        """Calculate Pearson correlation coefficient."""
        counts1 = [d["count"] for d in series1]
        counts2 = [d["count"] for d in series2]

        n = min(len(counts1), len(counts2))
        counts1 = counts1[-n:]
        counts2 = counts2[-n:]

        mean1 = statistics.mean(counts1)
        mean2 = statistics.mean(counts2)

        numerator = sum((counts1[i] - mean1) * (counts2[i] - mean2) for i in range(n))
        denom1 = sum((x - mean1)**2 for x in counts1) ** 0.5
        denom2 = sum((x - mean2)**2 for x in counts2) ** 0.5

        if denom1 * denom2 == 0:
            return 0

        return numerator / (denom1 * denom2)

    def find_leading_indicators(self, target_entity, candidate_entities, max_lag=7):
        """Find entities whose coverage leads the target entity."""
        target_series = self.forecaster.fetch_entity_series(target_entity, days=60)

        leading_indicators = []

        for candidate in candidate_entities:
            candidate_series = self.forecaster.fetch_entity_series(candidate, days=60)

            # Test different lags
            for lag in range(1, max_lag + 1):
                # Shift candidate series forward (it leads target)
                shifted_candidate = candidate_series[:-lag]
                shifted_target = target_series[lag:]

                if len(shifted_candidate) < 14:
                    continue

                corr = self.calculate_correlation(shifted_candidate, shifted_target)

                if abs(corr) > 0.5:
                    leading_indicators.append({
                        "entity": candidate,
                        "lag_days": lag,
                        "correlation": corr,
                        "relationship": "positive_lead" if corr > 0 else "negative_lead"
                    })

        # Sort by absolute correlation
        leading_indicators.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return leading_indicators

    def build_correlation_matrix(self, entities):
        """Build correlation matrix between entities."""
        series_data = {e: self.forecaster.fetch_entity_series(e, days=60) for e in entities}

        matrix = {}
        for e1 in entities:
            matrix[e1] = {}
            for e2 in entities:
                if e1 == e2:
                    matrix[e1][e2] = 1.0
                else:
                    matrix[e1][e2] = self.calculate_correlation(series_data[e1], series_data[e2])

        return matrix


class EarlyWarningSystem:
    """Multi-signal early warning system."""

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.alert_thresholds = {
            "volume_spike_zscore": 2.5,
            "sentiment_deterioration_pct": 50,
            "momentum_threshold": 100,
            "volatility_spike_ratio": 2.0
        }

    def calculate_volatility(self, series, window=7):
        """Calculate rolling volatility."""
        counts = [d["count"] for d in series]
        volatilities = []

        for i in range(window, len(counts)):
            window_data = counts[i-window:i]
            if len(window_data) > 1:
                volatilities.append(statistics.stdev(window_data))
            else:
                volatilities.append(0)

        return volatilities

    def check_volume_anomaly(self, entity):
        """Check for unusual volume patterns."""
        series = self.forecaster.fetch_entity_series(entity, days=60)
        counts = [d["count"] for d in series]

        recent = statistics.mean(counts[-3:])
        historical_mean = statistics.mean(counts[:-3])
        historical_std = statistics.stdev(counts[:-3]) if len(counts) > 3 else 1

        z_score = (recent - historical_mean) / historical_std if historical_std else 0

        return {
            "metric": "volume_anomaly",
            "current": recent,
            "baseline": historical_mean,
            "z_score": z_score,
            "alert": abs(z_score) > self.alert_thresholds["volume_spike_zscore"]
        }

    def check_volatility_regime(self, entity):
        """Detect volatility regime changes."""
        series = self.forecaster.fetch_entity_series(entity, days=60)
        volatilities = self.calculate_volatility(series)

        if len(volatilities) < 14:
            return {"metric": "volatility_regime", "alert": False}

        recent_vol = statistics.mean(volatilities[-7:])
        prior_vol = statistics.mean(volatilities[-14:-7])

        ratio = recent_vol / prior_vol if prior_vol else 1

        return {
            "metric": "volatility_regime",
            "current_volatility": recent_vol,
            "prior_volatility": prior_vol,
            "ratio": ratio,
            "alert": ratio > self.alert_thresholds["volatility_spike_ratio"]
        }

    def check_trend_acceleration(self, entity):
        """Detect accelerating or decelerating trends."""
        series = self.forecaster.fetch_entity_series(entity, days=30)
        counts = [d["count"] for d in series]

        if len(counts) < 21:
            return {"metric": "trend_acceleration", "alert": False}

        # Calculate rate of change
        roc_recent = (statistics.mean(counts[-7:]) - statistics.mean(counts[-14:-7])) / max(statistics.mean(counts[-14:-7]), 1)
        roc_prior = (statistics.mean(counts[-14:-7]) - statistics.mean(counts[-21:-14])) / max(statistics.mean(counts[-21:-14]), 1)

        acceleration = roc_recent - roc_prior

        return {
            "metric": "trend_acceleration",
            "recent_roc": roc_recent * 100,
            "prior_roc": roc_prior * 100,
            "acceleration": acceleration * 100,
            "alert": abs(acceleration) > 0.5  # 50% acceleration/deceleration
        }

    def generate_comprehensive_alert(self, entity):
        """Generate comprehensive early warning report."""
        alerts = []

        volume = self.check_volume_anomaly(entity)
        if volume["alert"]:
            alerts.append({
                "type": "VOLUME_ANOMALY",
                "severity": "HIGH" if abs(volume["z_score"]) > 3 else "MEDIUM",
                "message": f"Volume z-score: {volume['z_score']:.2f}",
                "data": volume
            })

        volatility = self.check_volatility_regime(entity)
        if volatility["alert"]:
            alerts.append({
                "type": "VOLATILITY_REGIME_CHANGE",
                "severity": "HIGH" if volatility["ratio"] > 3 else "MEDIUM",
                "message": f"Volatility increased {volatility['ratio']:.1f}x",
                "data": volatility
            })

        acceleration = self.check_trend_acceleration(entity)
        if acceleration["alert"]:
            alerts.append({
                "type": "TREND_ACCELERATION",
                "severity": "MEDIUM",
                "message": f"Trend acceleration: {acceleration['acceleration']:.1f}%",
                "data": acceleration
            })

        return {
            "entity": entity,
            "timestamp": datetime.utcnow().isoformat(),
            "alerts": alerts,
            "alert_count": len(alerts),
            "max_severity": max([a["severity"] for a in alerts], default="NONE")
        }


# Run comprehensive analysis
print("ADVANCED PREDICTIVE NEWS ANALYTICS")
print("=" * 70)

forecaster = AdvancedTimeSeriesForecaster()
entity = "Tesla"

print(f"\nBuilding time series for {entity}...")
series = forecaster.fetch_entity_series(entity, days=60)

# Seasonal decomposition
print("\n1. SEASONAL DECOMPOSITION")
print("-" * 40)
decomp = forecaster.seasonal_decomposition(series)
print("Day-of-week seasonality factors:")
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for i, factor in enumerate(decomp["seasonal"]):
    print(f"  {days[i]}: {factor:.2f}x")

# Changepoint detection
print("\n2. CHANGEPOINT DETECTION")
print("-" * 40)
changepoints = forecaster.detect_changepoints(series)
print(f"Detected {len(changepoints)} structural breaks:")
for cp in changepoints[:3]:
    print(f"  {cp['date']}: {cp['change_pct']:+.1f}% change (significance: {cp['significance']:.2f})")

# Ensemble forecast
print("\n3. ENSEMBLE FORECAST (7 days)")
print("-" * 40)
ensemble = forecaster.ensemble_forecast(entity, horizon=7)
for f in ensemble:
    print(f"  {f['date']}: {f['ensemble_forecast']:.0f} articles")

# Confidence intervals
print("\n4. FORECAST WITH 95% CONFIDENCE INTERVALS")
print("-" * 40)
conf_forecast = forecaster.forecast_with_confidence(series, horizon=7)
for f in conf_forecast:
    print(f"  {f['date']}: {f['forecast']:.0f} [{f['lower']:.0f} - {f['upper']:.0f}]")

# Early warning
print("\n5. EARLY WARNING SIGNALS")
print("-" * 40)
ews = EarlyWarningSystem(forecaster)
alert_report = ews.generate_comprehensive_alert(entity)
print(f"Alert count: {alert_report['alert_count']}")
print(f"Max severity: {alert_report['max_severity']}")
for alert in alert_report["alerts"]:
    print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

# Correlation analysis
print("\n6. LEADING INDICATOR ANALYSIS")
print("-" * 40)
correlator = MultiEntityCorrelationAnalyzer(forecaster)
candidates = ["NVIDIA", "Apple", "Microsoft"]
print(f"Checking if {candidates} lead {entity}...")
indicators = correlator.find_leading_indicators(entity, candidates, max_lag=5)
for ind in indicators[:3]:
    print(f"  {ind['entity']}: {ind['lag_days']}-day lead, r={ind['correlation']:.2f}")
```

---

## JavaScript — Real-Time Anomaly Detection Engine

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class AnomalyDetectionEngine {
  constructor(config = {}) {
    this.windowSize = config.windowSize || 30;
    this.sensitivityMultiplier = config.sensitivityMultiplier || 2.0;
    this.minSamples = config.minSamples || 14;
    this.entityBaselines = new Map();
  }

  async fetchDailyCount(entity, date) {
    const nextDate = new Date(new Date(date).getTime() + 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": entity,
      "published_at.start": date,
      "published_at.end": nextDate,
      language: "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    return data.total_results || 0;
  }

  async buildBaseline(entity) {
    const series = [];

    for (let i = this.windowSize; i > 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];
      const count = await this.fetchDailyCount(entity, date);
      series.push({ date, count });
    }

    const counts = series.map((d) => d.count);
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
    const variance =
      counts.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / counts.length;
    const std = Math.sqrt(variance);

    // Calculate percentiles
    const sorted = [...counts].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(counts.length * 0.25)];
    const p50 = sorted[Math.floor(counts.length * 0.5)];
    const p75 = sorted[Math.floor(counts.length * 0.75)];
    const p95 = sorted[Math.floor(counts.length * 0.95)];

    // Day-of-week patterns
    const dowPatterns = Array(7).fill(0);
    const dowCounts = Array(7).fill(0);

    series.forEach((d) => {
      const dow = new Date(d.date).getDay();
      dowPatterns[dow] += d.count;
      dowCounts[dow]++;
    });

    const dowFactors = dowPatterns.map((sum, i) =>
      dowCounts[i] ? sum / dowCounts[i] / mean : 1
    );

    const baseline = {
      entity,
      series,
      stats: { mean, std, p25, p50, p75, p95 },
      dowFactors,
      lastUpdated: new Date().toISOString(),
    };

    this.entityBaselines.set(entity, baseline);
    return baseline;
  }

  async detectAnomaly(entity, date = null) {
    if (!this.entityBaselines.has(entity)) {
      await this.buildBaseline(entity);
    }

    const baseline = this.entityBaselines.get(entity);
    const targetDate =
      date || new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().split("T")[0];
    const count = await this.fetchDailyCount(entity, targetDate);

    const dow = new Date(targetDate).getDay();
    const expectedMean = baseline.stats.mean * baseline.dowFactors[dow];
    const adjustedStd = baseline.stats.std * baseline.dowFactors[dow];

    const zScore = adjustedStd > 0 ? (count - expectedMean) / adjustedStd : 0;

    const anomaly = {
      entity,
      date: targetDate,
      observed: count,
      expected: expectedMean,
      zScore,
      percentile: this.calculatePercentile(count, baseline.series),
      isAnomaly: Math.abs(zScore) > this.sensitivityMultiplier,
      anomalyType: null,
      severity: null,
    };

    if (anomaly.isAnomaly) {
      anomaly.anomalyType = zScore > 0 ? "SPIKE" : "DROP";
      anomaly.severity =
        Math.abs(zScore) > 3.5
          ? "CRITICAL"
          : Math.abs(zScore) > 2.5
          ? "HIGH"
          : "MEDIUM";
    }

    return anomaly;
  }

  calculatePercentile(value, series) {
    const counts = series.map((d) => d.count);
    const below = counts.filter((c) => c < value).length;
    return (below / counts.length) * 100;
  }

  async detectMultiDayPattern(entity, days = 7) {
    const anomalies = [];

    for (let i = days; i > 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];
      const anomaly = await this.detectAnomaly(entity, date);
      anomalies.push(anomaly);
    }

    // Detect patterns
    const patterns = {
      consecutiveSpikes: 0,
      consecutiveDrops: 0,
      oscillation: false,
      trend: null,
    };

    let spikeCount = 0;
    let dropCount = 0;
    let signChanges = 0;
    let prevSign = null;

    anomalies.forEach((a) => {
      if (a.zScore > 1) spikeCount++;
      if (a.zScore < -1) dropCount++;

      const sign = a.zScore > 0 ? 1 : -1;
      if (prevSign !== null && sign !== prevSign) signChanges++;
      prevSign = sign;
    });

    patterns.consecutiveSpikes = spikeCount;
    patterns.consecutiveDrops = dropCount;
    patterns.oscillation = signChanges >= days - 2;

    // Trend detection
    const zScores = anomalies.map((a) => a.zScore);
    const firstHalf = zScores.slice(0, Math.floor(days / 2));
    const secondHalf = zScores.slice(Math.floor(days / 2));

    const firstMean = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondMean = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    if (secondMean - firstMean > 0.5) patterns.trend = "INCREASING";
    else if (firstMean - secondMean > 0.5) patterns.trend = "DECREASING";
    else patterns.trend = "STABLE";

    return { anomalies, patterns };
  }
}

class PredictiveAlertManager {
  constructor(engine) {
    this.engine = engine;
    this.alertHistory = [];
    this.subscribers = [];
  }

  subscribe(callback) {
    this.subscribers.push(callback);
  }

  async monitorEntity(entity, intervalMs = 3600000) {
    console.log(`Starting monitoring for ${entity}`);

    const check = async () => {
      const result = await this.engine.detectMultiDayPattern(entity, 3);
      const latestAnomaly = result.anomalies[result.anomalies.length - 1];

      if (latestAnomaly.isAnomaly) {
        const alert = {
          timestamp: new Date().toISOString(),
          entity,
          anomaly: latestAnomaly,
          pattern: result.patterns,
          message: this.generateAlertMessage(latestAnomaly, result.patterns),
        };

        this.alertHistory.push(alert);
        this.subscribers.forEach((cb) => cb(alert));

        return alert;
      }

      return null;
    };

    // Initial check
    const initialAlert = await check();

    // Set up interval
    setInterval(check, intervalMs);

    return initialAlert;
  }

  generateAlertMessage(anomaly, patterns) {
    let message = `${anomaly.entity}: ${anomaly.anomalyType} detected `;
    message += `(${anomaly.observed} vs expected ${anomaly.expected.toFixed(0)}, `;
    message += `z=${anomaly.zScore.toFixed(2)}, severity=${anomaly.severity})`;

    if (patterns.trend !== "STABLE") {
      message += `. Trend: ${patterns.trend}`;
    }

    if (patterns.oscillation) {
      message += `. Warning: High volatility detected`;
    }

    return message;
  }
}

// Run anomaly detection
async function runAnomalyDetection() {
  console.log("REAL-TIME ANOMALY DETECTION ENGINE");
  console.log("=".repeat(60));

  const engine = new AnomalyDetectionEngine({
    windowSize: 30,
    sensitivityMultiplier: 2.0,
  });

  const entity = "Tesla";

  console.log(`\nBuilding baseline for ${entity}...`);
  const baseline = await engine.buildBaseline(entity);

  console.log("\nBASELINE STATISTICS:");
  console.log(`  Mean: ${baseline.stats.mean.toFixed(1)} articles/day`);
  console.log(`  Std Dev: ${baseline.stats.std.toFixed(1)}`);
  console.log(`  Median: ${baseline.stats.p50}`);
  console.log(`  95th percentile: ${baseline.stats.p95}`);

  console.log("\nDAY-OF-WEEK FACTORS:");
  const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  baseline.dowFactors.forEach((f, i) => {
    console.log(`  ${days[i]}: ${f.toFixed(2)}x`);
  });

  console.log("\nMULTI-DAY PATTERN ANALYSIS (last 7 days):");
  const pattern = await engine.detectMultiDayPattern(entity, 7);

  pattern.anomalies.forEach((a) => {
    const marker = a.isAnomaly ? `[${a.severity}]` : "";
    console.log(
      `  ${a.date}: ${a.observed} (z=${a.zScore.toFixed(2)}) ${marker}`
    );
  });

  console.log("\nPATTERN SUMMARY:");
  console.log(`  Trend: ${pattern.patterns.trend}`);
  console.log(`  Spikes: ${pattern.patterns.consecutiveSpikes}`);
  console.log(`  Drops: ${pattern.patterns.consecutiveDrops}`);
  console.log(`  High volatility: ${pattern.patterns.oscillation}`);

  // Set up alert manager
  console.log("\nSetting up alert monitoring...");
  const alertManager = new PredictiveAlertManager(engine);

  alertManager.subscribe((alert) => {
    console.log(`\n🚨 ALERT: ${alert.message}`);
  });

  // Single check (in production, this would run continuously)
  const alert = await alertManager.monitorEntity(entity);
  if (alert) {
    console.log(`Initial alert triggered: ${alert.message}`);
  } else {
    console.log("No immediate anomalies detected");
  }
}

runAnomalyDetection();
```

---

## PHP — Comprehensive Forecasting Service

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class ForecastingService
{
    private string $apiKey;
    private string $baseUrl;
    private array $seriesCache = [];

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function fetchTimeSeries(string $entity, int $days = 60): array
    {
        if (isset($this->seriesCache[$entity])) {
            return $this->seriesCache[$entity];
        }

        $series = [];

        for ($i = $days; $i > 0; $i--) {
            $date = (new DateTime("-{$i} days"))->format("Y-m-d");
            $nextDate = (new DateTime("-" . ($i - 1) . " days"))->format("Y-m-d");

            $query = http_build_query([
                "api_key" => $this->apiKey,
                "entity.name" => $entity,
                "published_at.start" => $date,
                "published_at.end" => $nextDate,
                "language" => "en",
                "per_page" => 1,
            ]);

            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $series[] = [
                "date" => $date,
                "count" => $data["total_results"] ?? 0,
                "dow" => (int)(new DateTime($date))->format("w"),
            ];
        }

        $this->seriesCache[$entity] = $series;
        return $series;
    }

    public function calculateStatistics(array $series): array
    {
        $counts = array_column($series, "count");
        $n = count($counts);

        $mean = array_sum($counts) / $n;
        $variance = array_sum(array_map(fn($x) => pow($x - $mean, 2), $counts)) / $n;
        $std = sqrt($variance);

        sort($counts);

        return [
            "mean" => $mean,
            "std" => $std,
            "min" => min($counts),
            "max" => max($counts),
            "median" => $counts[(int)($n / 2)],
            "p25" => $counts[(int)($n * 0.25)],
            "p75" => $counts[(int)($n * 0.75)],
            "p95" => $counts[(int)($n * 0.95)],
        ];
    }

    public function exponentialSmoothing(array $series, float $alpha = 0.3): array
    {
        $counts = array_column($series, "count");
        $smoothed = [$counts[0]];

        for ($i = 1; $i < count($counts); $i++) {
            $smoothed[] = $alpha * $counts[$i] + (1 - $alpha) * $smoothed[$i - 1];
        }

        return $smoothed;
    }

    public function holtLinearTrend(array $series, float $alpha = 0.3, float $beta = 0.1): array
    {
        $counts = array_column($series, "count");
        $n = count($counts);

        $level = [$counts[0]];
        $trend = [$n > 1 ? $counts[1] - $counts[0] : 0];

        for ($i = 1; $i < $n; $i++) {
            $level[] = $alpha * $counts[$i] + (1 - $alpha) * ($level[$i - 1] + $trend[$i - 1]);
            $trend[] = $beta * ($level[$i] - $level[$i - 1]) + (1 - $beta) * $trend[$i - 1];
        }

        return ["level" => $level, "trend" => $trend];
    }

    public function forecast(string $entity, int $horizon = 7, float $confidence = 0.95): array
    {
        $series = $this->fetchTimeSeries($entity);
        $holt = $this->holtLinearTrend($series);

        $counts = array_column($series, "count");
        $fitted = [];
        for ($i = 0; $i < count($holt["level"]); $i++) {
            $fitted[] = $holt["level"][$i] + $holt["trend"][$i];
        }

        $residuals = [];
        for ($i = 0; $i < count($counts); $i++) {
            $residuals[] = $counts[$i] - $fitted[$i];
        }

        $residualStd = sqrt(
            array_sum(array_map(fn($x) => pow($x, 2), $residuals)) / count($residuals)
        );

        $zScores = [0.90 => 1.645, 0.95 => 1.96, 0.99 => 2.576];
        $z = $zScores[$confidence] ?? 1.96;

        $lastLevel = end($holt["level"]);
        $lastTrend = end($holt["trend"]);

        $forecasts = [];
        for ($h = 1; $h <= $horizon; $h++) {
            $pointForecast = $lastLevel + $h * $lastTrend;
            $margin = $z * $residualStd * sqrt($h);

            $forecasts[] = [
                "date" => (new DateTime("+{$h} days"))->format("Y-m-d"),
                "forecast" => max(0, $pointForecast),
                "lower" => max(0, $pointForecast - $margin),
                "upper" => $pointForecast + $margin,
                "confidence" => $confidence,
            ];
        }

        return $forecasts;
    }

    public function detectAnomalies(string $entity, float $threshold = 2.0): array
    {
        $series = $this->fetchTimeSeries($entity);
        $stats = $this->calculateStatistics($series);

        $anomalies = [];

        foreach ($series as $point) {
            $zScore = $stats["std"] > 0
                ? ($point["count"] - $stats["mean"]) / $stats["std"]
                : 0;

            if (abs($zScore) > $threshold) {
                $anomalies[] = [
                    "date" => $point["date"],
                    "count" => $point["count"],
                    "z_score" => $zScore,
                    "type" => $zScore > 0 ? "SPIKE" : "DROP",
                    "severity" => abs($zScore) > 3 ? "HIGH" : "MEDIUM",
                ];
            }
        }

        return $anomalies;
    }

    public function detectChangepoints(string $entity, int $minSegment = 7): array
    {
        $series = $this->fetchTimeSeries($entity);
        $counts = array_column($series, "count");
        $n = count($counts);

        $changepoints = [];

        for ($i = $minSegment; $i < $n - $minSegment; $i++) {
            $before = array_slice($counts, $i - $minSegment, $minSegment);
            $after = array_slice($counts, $i, $minSegment);

            $beforeMean = array_sum($before) / count($before);
            $afterMean = array_sum($after) / count($after);

            $beforeVar = array_sum(array_map(fn($x) => pow($x - $beforeMean, 2), $before)) / count($before);
            $afterVar = array_sum(array_map(fn($x) => pow($x - $afterMean, 2), $after)) / count($after);

            $pooledStd = sqrt(($beforeVar + $afterVar) / 2);
            $tStat = $pooledStd > 0
                ? abs($afterMean - $beforeMean) / ($pooledStd * sqrt(2 / $minSegment))
                : 0;

            if ($tStat > 2.5) {
                $changepoints[] = [
                    "date" => $series[$i]["date"],
                    "before_mean" => $beforeMean,
                    "after_mean" => $afterMean,
                    "change_pct" => $beforeMean > 0
                        ? (($afterMean - $beforeMean) / $beforeMean) * 100
                        : 0,
                    "significance" => $tStat,
                ];
            }
        }

        // Filter nearby changepoints
        $filtered = [];
        foreach ($changepoints as $cp) {
            if (empty($filtered) ||
                strtotime($cp["date"]) - strtotime(end($filtered)["date"]) >= $minSegment * 86400) {
                $filtered[] = $cp;
            }
        }

        return $filtered;
    }

    public function generateReport(string $entity): array
    {
        $series = $this->fetchTimeSeries($entity);
        $stats = $this->calculateStatistics($series);
        $forecasts = $this->forecast($entity, 7);
        $anomalies = $this->detectAnomalies($entity);
        $changepoints = $this->detectChangepoints($entity);

        // Early warning signals
        $warnings = [];

        $recentCounts = array_slice(array_column($series, "count"), -7);
        $recentMean = array_sum($recentCounts) / count($recentCounts);

        if ($recentMean > $stats["mean"] * 1.5) {
            $warnings[] = [
                "type" => "ELEVATED_VOLUME",
                "message" => sprintf(
                    "Recent volume %.0f%% above baseline",
                    (($recentMean / $stats["mean"]) - 1) * 100
                ),
            ];
        }

        $recentAnomalies = array_filter($anomalies, function ($a) {
            return strtotime($a["date"]) >= strtotime("-7 days");
        });

        if (count($recentAnomalies) >= 2) {
            $warnings[] = [
                "type" => "ANOMALY_CLUSTER",
                "message" => count($recentAnomalies) . " anomalies in past 7 days",
            ];
        }

        return [
            "entity" => $entity,
            "generated_at" => (new DateTime())->format("c"),
            "statistics" => $stats,
            "forecasts" => $forecasts,
            "anomalies" => $anomalies,
            "changepoints" => $changepoints,
            "warnings" => $warnings,
        ];
    }
}

// Run forecasting
$service = new ForecastingService();
$entity = "Tesla";

echo "COMPREHENSIVE FORECASTING SERVICE\n";
echo str_repeat("=", 60) . "\n";

$report = $service->generateReport($entity);

echo "\nENTITY: {$report['entity']}\n";
echo "Generated: {$report['generated_at']}\n";

echo "\nSTATISTICS:\n";
printf("  Mean: %.1f articles/day\n", $report["statistics"]["mean"]);
printf("  Std Dev: %.1f\n", $report["statistics"]["std"]);
printf("  Median: %d\n", $report["statistics"]["median"]);
printf("  Range: %d - %d\n", $report["statistics"]["min"], $report["statistics"]["max"]);

echo "\n7-DAY FORECAST (95% CI):\n";
foreach ($report["forecasts"] as $f) {
    printf(
        "  %s: %.0f [%.0f - %.0f]\n",
        $f["date"],
        $f["forecast"],
        $f["lower"],
        $f["upper"]
    );
}

echo "\nANOMALIES DETECTED: " . count($report["anomalies"]) . "\n";
foreach (array_slice($report["anomalies"], -5) as $a) {
    printf(
        "  %s: %d (z=%.2f, %s, %s)\n",
        $a["date"],
        $a["count"],
        $a["z_score"],
        $a["type"],
        $a["severity"]
    );
}

echo "\nCHANGEPOINTS: " . count($report["changepoints"]) . "\n";
foreach ($report["changepoints"] as $cp) {
    printf(
        "  %s: %.1f → %.1f (%+.1f%%)\n",
        $cp["date"],
        $cp["before_mean"],
        $cp["after_mean"],
        $cp["change_pct"]
    );
}

echo "\nWARNINGS: " . count($report["warnings"]) . "\n";
foreach ($report["warnings"] as $w) {
    echo "  [{$w['type']}] {$w['message']}\n";
}
```

---

## See Also

- [README.md](./README.md) — Predictive News Analytics workflow overview and quick start.
