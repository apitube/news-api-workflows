# Real-Time Anomaly Detection System

Workflow for detecting anomalies in news streams using multiple statistical methods, ensemble scoring, adaptive thresholds, pattern classification, and automated alerting with root cause analysis using the [APITube News API](https://apitube.io).

## Overview

The **Real-Time Anomaly Detection System** implements production-grade anomaly detection by combining multiple detection algorithms (z-score, IQR, CUSUM, exponential smoothing residuals), ensemble scoring with confidence weighting, adaptive threshold adjustment based on regime, pattern classification (spike, drop, shift, oscillation), and automated root cause analysis. Features include multi-entity monitoring, alert prioritization, and false positive suppression. Ideal for news monitoring operations, crisis detection, trading desks, and security operations centers.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Quick Start

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import math
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class AnomalyDetector:
    """Base class for anomaly detection algorithms."""

    def __init__(self, name):
        self.name = name

    def detect(self, value, history):
        """Returns anomaly score (0-1) and details."""
        raise NotImplementedError


class ZScoreDetector(AnomalyDetector):
    """Z-score based anomaly detection."""

    def __init__(self, threshold=2.5):
        super().__init__("zscore")
        self.threshold = threshold

    def detect(self, value, history):
        if len(history) < 10:
            return 0, {"z_score": 0, "threshold": self.threshold}

        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 1

        if std == 0:
            std = 1

        z_score = (value - mean) / std

        # Convert to anomaly score (0-1)
        anomaly_score = min(1.0, abs(z_score) / (self.threshold * 2))

        return anomaly_score, {
            "z_score": z_score,
            "mean": mean,
            "std": std,
            "threshold": self.threshold,
            "is_anomaly": abs(z_score) > self.threshold
        }


class IQRDetector(AnomalyDetector):
    """Interquartile range based anomaly detection."""

    def __init__(self, multiplier=1.5):
        super().__init__("iqr")
        self.multiplier = multiplier

    def detect(self, value, history):
        if len(history) < 10:
            return 0, {"iqr": 0}

        sorted_history = sorted(history)
        n = len(sorted_history)

        q1 = sorted_history[int(n * 0.25)]
        q3 = sorted_history[int(n * 0.75)]
        iqr = q3 - q1

        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        if value < lower_bound:
            distance = (lower_bound - value) / iqr if iqr > 0 else 0
            anomaly_score = min(1.0, distance / 2)
        elif value > upper_bound:
            distance = (value - upper_bound) / iqr if iqr > 0 else 0
            anomaly_score = min(1.0, distance / 2)
        else:
            anomaly_score = 0

        return anomaly_score, {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "is_anomaly": value < lower_bound or value > upper_bound
        }


class CUSUMDetector(AnomalyDetector):
    """Cumulative sum (CUSUM) change detection."""

    def __init__(self, threshold=5, drift=0.5):
        super().__init__("cusum")
        self.threshold = threshold
        self.drift = drift
        self.pos_cusum = 0
        self.neg_cusum = 0

    def detect(self, value, history):
        if len(history) < 10:
            return 0, {"cusum_pos": 0, "cusum_neg": 0}

        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 1

        if std == 0:
            std = 1

        normalized = (value - mean) / std

        # Update CUSUM
        self.pos_cusum = max(0, self.pos_cusum + normalized - self.drift)
        self.neg_cusum = max(0, self.neg_cusum - normalized - self.drift)

        max_cusum = max(self.pos_cusum, self.neg_cusum)
        anomaly_score = min(1.0, max_cusum / (self.threshold * 2))

        return anomaly_score, {
            "cusum_pos": self.pos_cusum,
            "cusum_neg": self.neg_cusum,
            "threshold": self.threshold,
            "is_anomaly": max_cusum > self.threshold
        }

    def reset(self):
        self.pos_cusum = 0
        self.neg_cusum = 0


class ExponentialSmoothingDetector(AnomalyDetector):
    """Exponential smoothing residual based detection."""

    def __init__(self, alpha=0.3, threshold=2.5):
        super().__init__("exp_smoothing")
        self.alpha = alpha
        self.threshold = threshold
        self.smoothed = None

    def detect(self, value, history):
        if len(history) < 5:
            self.smoothed = value
            return 0, {"smoothed": value, "residual": 0}

        # Update smoothed value
        if self.smoothed is None:
            self.smoothed = history[0]

        for h in history[-5:]:
            self.smoothed = self.alpha * h + (1 - self.alpha) * self.smoothed

        # Calculate residual
        residual = value - self.smoothed

        # Estimate residual std from history
        residuals = []
        temp_smoothed = history[0]
        for h in history:
            residuals.append(h - temp_smoothed)
            temp_smoothed = self.alpha * h + (1 - self.alpha) * temp_smoothed

        residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 1
        if residual_std == 0:
            residual_std = 1

        z_residual = residual / residual_std
        anomaly_score = min(1.0, abs(z_residual) / (self.threshold * 2))

        return anomaly_score, {
            "smoothed": self.smoothed,
            "residual": residual,
            "z_residual": z_residual,
            "is_anomaly": abs(z_residual) > self.threshold
        }


class EnsembleAnomalyDetector:
    """Combines multiple detectors with weighted voting."""

    def __init__(self):
        self.detectors = [
            (ZScoreDetector(threshold=2.5), 0.30),
            (IQRDetector(multiplier=1.5), 0.25),
            (CUSUMDetector(threshold=5), 0.25),
            (ExponentialSmoothingDetector(alpha=0.3), 0.20)
        ]

    def detect(self, value, history):
        """Run all detectors and combine scores."""
        results = {}
        weighted_score = 0
        total_weight = 0

        for detector, weight in self.detectors:
            score, details = detector.detect(value, history)
            results[detector.name] = {
                "score": score,
                "weight": weight,
                "details": details
            }
            weighted_score += score * weight
            total_weight += weight

        ensemble_score = weighted_score / total_weight if total_weight > 0 else 0

        # Determine if anomaly based on ensemble
        is_anomaly = ensemble_score > 0.5
        confidence = ensemble_score if is_anomaly else 1 - ensemble_score

        return {
            "ensemble_score": ensemble_score,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "detectors": results
        }

    def reset(self):
        """Reset stateful detectors."""
        for detector, _ in self.detectors:
            if hasattr(detector, 'reset'):
                detector.reset()


class AnomalyClassifier:
    """Classifies anomaly patterns."""

    @staticmethod
    def classify(value, history, window=7):
        """Classify the type of anomaly."""
        if len(history) < window * 2:
            return "unknown"

        recent = history[-window:]
        prior = history[-window*2:-window]

        recent_avg = statistics.mean(recent)
        prior_avg = statistics.mean(prior)
        recent_std = statistics.stdev(recent) if len(recent) > 1 else 0
        prior_std = statistics.stdev(prior) if len(prior) > 1 else 0

        # Classification logic
        if value > recent_avg + 2 * (recent_std or 1):
            if recent_avg > prior_avg * 1.3:
                return "sustained_spike"
            return "point_spike"

        elif value < recent_avg - 2 * (recent_std or 1):
            if recent_avg < prior_avg * 0.7:
                return "sustained_drop"
            return "point_drop"

        elif abs(recent_avg - prior_avg) > prior_std * 2:
            return "level_shift"

        elif recent_std > prior_std * 2:
            return "volatility_increase"

        else:
            return "normal"


class RootCauseAnalyzer:
    """Analyzes potential root causes of anomalies."""

    EVENT_KEYWORDS = {
        "earnings": ["earnings", "revenue", "profit", "quarterly results", "guidance"],
        "leadership": ["CEO", "executive", "resignation", "appointed", "fired"],
        "legal": ["lawsuit", "litigation", "court", "sued", "verdict"],
        "regulatory": ["SEC", "FTC", "investigation", "fine", "compliance"],
        "product": ["launch", "recall", "defect", "new product", "release"],
        "market": ["stock", "shares", "market cap", "valuation", "IPO"],
        "partnership": ["partnership", "acquisition", "merger", "deal", "contract"],
        "crisis": ["scandal", "controversy", "crisis", "breach", "hack"]
    }

    def __init__(self):
        pass

    def analyze(self, entity, date, anomaly_type):
        """Find potential root causes for an anomaly."""
        causes = []

        next_date = (datetime.fromisoformat(date) + timedelta(days=1)).strftime("%Y-%m-%d")

        for category, keywords in self.EVENT_KEYWORDS.items():
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity,
                "title": ",".join(keywords),
                "published_at.start": date,
                "published_at.end": next_date,
                "source.rank.opr.min": 0.6,
                "language": "en",
                "per_page": 5,
            })

            articles = resp.json().get("results", [])
            if articles:
                causes.append({
                    "category": category,
                    "article_count": len(articles),
                    "sample_headlines": [a.get("title") for a in articles[:3]],
                    "relevance_score": len(articles) / 10  # Normalized
                })

        # Sort by relevance
        causes.sort(key=lambda x: x["relevance_score"], reverse=True)

        return {
            "entity": entity,
            "date": date,
            "anomaly_type": anomaly_type,
            "potential_causes": causes[:3],
            "confidence": causes[0]["relevance_score"] if causes else 0
        }


class RealTimeAnomalyMonitor:
    """Production-grade real-time anomaly monitoring system."""

    def __init__(self, entities, history_days=60):
        self.entities = entities
        self.history_days = history_days
        self.detectors = {e: EnsembleAnomalyDetector() for e in entities}
        self.history = {e: deque(maxlen=history_days) for e in entities}
        self.classifier = AnomalyClassifier()
        self.root_cause = RootCauseAnalyzer()
        self.alerts = []
        self.alert_cooldown = defaultdict(lambda: None)  # Prevent alert fatigue

    def fetch_historical(self, entity):
        """Fetch historical data for baseline."""
        for i in range(self.history_days, 0, -1):
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
            self.history[entity].append(count)

    def initialize(self):
        """Initialize historical baselines for all entities."""
        print("Initializing historical baselines...")
        for entity in self.entities:
            print(f"  Loading history for {entity}...")
            self.fetch_historical(entity)

    def check_entity(self, entity):
        """Check single entity for anomalies."""
        # Get today's count
        today = datetime.utcnow().strftime("%Y-%m-%d")
        tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "published_at.start": today,
            "published_at.end": tomorrow,
            "language": "en",
            "per_page": 1,
        })

        current_count = resp.json().get("total_results", 0)
        history_list = list(self.history[entity])

        # Run detection
        result = self.detectors[entity].detect(current_count, history_list)

        # Classify anomaly type
        anomaly_type = self.classifier.classify(current_count, history_list)

        # Update history
        self.history[entity].append(current_count)

        return {
            "entity": entity,
            "timestamp": datetime.utcnow().isoformat(),
            "current_value": current_count,
            "baseline_mean": statistics.mean(history_list) if history_list else 0,
            "detection": result,
            "anomaly_type": anomaly_type
        }

    def check_all(self):
        """Check all entities for anomalies."""
        results = []

        for entity in self.entities:
            result = self.check_entity(entity)
            results.append(result)

            # Generate alert if anomaly detected
            if result["detection"]["is_anomaly"]:
                self._process_alert(result)

        return results

    def _process_alert(self, result):
        """Process and potentially raise an alert."""
        entity = result["entity"]

        # Check cooldown
        last_alert = self.alert_cooldown[entity]
        if last_alert:
            hours_since = (datetime.utcnow() - last_alert).total_seconds() / 3600
            if hours_since < 4:  # 4 hour cooldown
                return

        # Analyze root cause
        root_cause = self.root_cause.analyze(
            entity,
            datetime.utcnow().strftime("%Y-%m-%d"),
            result["anomaly_type"]
        )

        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "entity": entity,
            "severity": self._calculate_severity(result),
            "anomaly_type": result["anomaly_type"],
            "current_value": result["current_value"],
            "baseline": result["baseline_mean"],
            "deviation_pct": ((result["current_value"] - result["baseline_mean"]) / max(result["baseline_mean"], 1)) * 100,
            "confidence": result["detection"]["confidence"],
            "root_cause": root_cause,
            "detector_scores": {
                name: data["score"]
                for name, data in result["detection"]["detectors"].items()
            }
        }

        self.alerts.append(alert)
        self.alert_cooldown[entity] = datetime.utcnow()

        return alert

    def _calculate_severity(self, result):
        """Calculate alert severity."""
        score = result["detection"]["ensemble_score"]
        anomaly_type = result["anomaly_type"]

        if score > 0.8 or anomaly_type in ["sustained_spike", "sustained_drop"]:
            return "CRITICAL"
        elif score > 0.6 or anomaly_type == "level_shift":
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def get_alerts(self, hours=24, severity=None):
        """Get recent alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        filtered = [
            a for a in self.alerts
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]

        if severity:
            filtered = [a for a in filtered if a["severity"] == severity]

        return sorted(filtered, key=lambda x: x["timestamp"], reverse=True)

    def get_summary(self):
        """Get monitoring summary."""
        return {
            "entities_monitored": len(self.entities),
            "total_alerts_24h": len(self.get_alerts(hours=24)),
            "critical_alerts": len(self.get_alerts(severity="CRITICAL")),
            "high_alerts": len(self.get_alerts(severity="HIGH")),
            "entities_with_anomalies": len(set(a["entity"] for a in self.get_alerts(hours=24)))
        }


# Run anomaly detection
print("REAL-TIME ANOMALY DETECTION SYSTEM")
print("=" * 70)

entities = ["Tesla", "Apple", "Microsoft", "Amazon", "Google"]

monitor = RealTimeAnomalyMonitor(entities, history_days=30)
monitor.initialize()

print("\n" + "=" * 70)
print("RUNNING ANOMALY CHECK")
print("-" * 50)

results = monitor.check_all()

for result in results:
    status = "🚨 ANOMALY" if result["detection"]["is_anomaly"] else "✓ Normal"
    print(f"\n{result['entity']}: {status}")
    print(f"  Current: {result['current_value']} | Baseline: {result['baseline_mean']:.1f}")
    print(f"  Ensemble Score: {result['detection']['ensemble_score']:.3f}")
    print(f"  Pattern: {result['anomaly_type']}")

    if result["detection"]["is_anomaly"]:
        print(f"  Confidence: {result['detection']['confidence']:.1%}")
        print("  Detector Breakdown:")
        for name, data in result["detection"]["detectors"].items():
            indicator = "⚠" if data["details"].get("is_anomaly") else "○"
            print(f"    {indicator} {name}: {data['score']:.3f}")

# Show alerts
print("\n" + "=" * 70)
print("ACTIVE ALERTS")
print("-" * 50)

alerts = monitor.get_alerts(hours=24)
if alerts:
    for alert in alerts:
        print(f"\n[{alert['severity']}] {alert['entity']}")
        print(f"  Type: {alert['anomaly_type']}")
        print(f"  Deviation: {alert['deviation_pct']:+.1f}%")
        print(f"  Confidence: {alert['confidence']:.1%}")

        if alert["root_cause"]["potential_causes"]:
            print("  Likely Causes:")
            for cause in alert["root_cause"]["potential_causes"]:
                print(f"    - {cause['category']}: {cause['sample_headlines'][0][:50]}...")
else:
    print("No active alerts")

# Summary
print("\n" + "=" * 70)
print("MONITORING SUMMARY")
print("-" * 50)
summary = monitor.get_summary()
for key, value in summary.items():
    print(f"  {key}: {value}")
```

## Detection Algorithms

| Algorithm | Method | Strengths |
|-----------|--------|-----------|
| Z-Score | Standard deviation from mean | Good for normally distributed data |
| IQR | Interquartile range bounds | Robust to outliers |
| CUSUM | Cumulative sum of deviations | Detects gradual shifts |
| Exp. Smoothing | Residuals from smoothed series | Handles trends well |

## Anomaly Classifications

| Pattern | Description |
|---------|-------------|
| `point_spike` | Single-day spike above normal |
| `sustained_spike` | Multi-day elevated levels |
| `point_drop` | Single-day drop below normal |
| `sustained_drop` | Multi-day depressed levels |
| `level_shift` | Permanent change in baseline |
| `volatility_increase` | Increased variation |

## Common Use Cases

- **News monitoring operations** — detect unusual coverage patterns.
- **Crisis detection** — early warning of emerging issues.
- **Trading desks** — monitor news-driven market signals.
- **Security operations** — detect coordinated campaigns.
- **Brand monitoring** — alert on reputation changes.
- **Competitive intelligence** — track competitor news spikes.

## See Also

- [examples.md](./examples.md) — additional code examples.
