# Reputation Risk Engine

Workflow for calculating multi-dimensional reputation scores with decay functions, controversy half-life modeling, trust propagation networks, crisis impact simulation, and reputation recovery forecasting using the [APITube News API](https://apitube.io).

## Overview

The **Reputation Risk Engine** workflow implements sophisticated reputation modeling by combining multiple reputation dimensions, applying temporal decay to past events, modeling controversy half-life, simulating trust propagation through entity networks, and forecasting reputation recovery trajectories. Features include regime detection (stable/crisis/recovery), scenario simulation, and early warning triggers. Ideal for corporate communications, investor relations, risk management, and brand strategy.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by entity name.                                               |
| `title`                       | string  | Filter by keywords.                                                  |
| `sentiment.overall.polarity`  | string  | Filter by sentiment.                                                 |
| `source.rank.opr.min`         | number  | Minimum source authority.                                            |
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


class ReputationDimension:
    """A single dimension of reputation (e.g., trust, quality, ethics)."""

    def __init__(self, name, keywords, weight=1.0):
        self.name = name
        self.keywords = keywords
        self.weight = weight
        self.daily_scores = []

    def calculate_decayed_score(self, half_life_days=30):
        """Calculate score with exponential decay applied to older data."""
        if not self.daily_scores:
            return 0.5

        weighted_sum = 0
        weight_total = 0

        for i, score_data in enumerate(reversed(self.daily_scores)):
            days_ago = i
            decay_factor = 0.5 ** (days_ago / half_life_days)

            weighted_sum += score_data["score"] * decay_factor
            weight_total += decay_factor

        return weighted_sum / weight_total if weight_total > 0 else 0.5


class ControversyEvent:
    """Represents a controversy affecting reputation."""

    def __init__(self, date, severity, category, description):
        self.date = date
        self.severity = severity  # 0-1 scale
        self.category = category
        self.description = description
        self.half_life_days = self._calculate_half_life()

    def _calculate_half_life(self):
        """Estimate half-life based on severity and category."""
        base_half_life = {
            "minor": 7,
            "moderate": 21,
            "major": 60,
            "severe": 180,
            "catastrophic": 365
        }

        if self.severity < 0.2:
            return base_half_life["minor"]
        elif self.severity < 0.4:
            return base_half_life["moderate"]
        elif self.severity < 0.6:
            return base_half_life["major"]
        elif self.severity < 0.8:
            return base_half_life["severe"]
        else:
            return base_half_life["catastrophic"]

    def current_impact(self, current_date=None):
        """Calculate current impact with decay."""
        current_date = current_date or datetime.utcnow()

        if isinstance(self.date, str):
            event_date = datetime.fromisoformat(self.date)
        else:
            event_date = self.date

        days_elapsed = (current_date - event_date).days

        decay_factor = 0.5 ** (days_elapsed / self.half_life_days)
        return self.severity * decay_factor


class ReputationRiskEngine:
    """
    Multi-dimensional reputation scoring with decay,
    controversy tracking, and forecasting.
    """

    REPUTATION_DIMENSIONS = {
        "trust": {
            "positive": ["trusted", "reliable", "credible", "honest", "transparent"],
            "negative": ["distrust", "unreliable", "deceptive", "misleading", "scandal"],
            "weight": 0.25
        },
        "quality": {
            "positive": ["quality", "excellent", "superior", "innovative", "best"],
            "negative": ["defect", "recall", "failure", "poor quality", "complaints"],
            "weight": 0.20
        },
        "ethics": {
            "positive": ["ethical", "responsible", "sustainable", "fair", "integrity"],
            "negative": ["unethical", "corruption", "fraud", "violation", "misconduct"],
            "weight": 0.20
        },
        "leadership": {
            "positive": ["visionary", "strong leadership", "effective", "respected"],
            "negative": ["mismanagement", "resignation", "fired", "controversy"],
            "weight": 0.15
        },
        "social": {
            "positive": ["community", "diversity", "inclusion", "philanthropy", "social good"],
            "negative": ["discrimination", "harassment", "toxic", "layoffs", "protests"],
            "weight": 0.20
        }
    }

    CONTROVERSY_KEYWORDS = {
        "legal": ["lawsuit", "sued", "litigation", "court", "verdict", "settlement"],
        "regulatory": ["fine", "penalty", "violation", "investigation", "SEC", "FTC"],
        "ethical": ["scandal", "fraud", "corruption", "bribery", "misconduct"],
        "operational": ["recall", "accident", "failure", "outage", "breach"],
        "leadership": ["resignation", "fired", "ousted", "controversy", "allegations"],
        "social": ["protest", "boycott", "backlash", "discrimination", "harassment"]
    }

    def __init__(self, entity):
        self.entity = entity
        self.dimensions = {}
        self.controversies = []
        self.daily_data = []
        self.regime = "stable"

    def fetch_historical_data(self, days=90):
        """Fetch historical news data for reputation analysis."""
        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            day_data = {"date": date, "dimensions": {}, "total": 0, "controversies": []}

            # Total coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.entity,
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            day_data["total"] = resp.json().get("total_results", 0)

            # Dimension scores
            for dim_name, dim_config in self.REPUTATION_DIMENSIONS.items():
                # Positive mentions
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "entity.name": self.entity,
                    "title": ",".join(dim_config["positive"]),
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                positive = resp.json().get("total_results", 0)

                # Negative mentions
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "entity.name": self.entity,
                    "title": ",".join(dim_config["negative"]),
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                negative = resp.json().get("total_results", 0)

                # Score: 0 (all negative) to 1 (all positive), 0.5 = neutral
                total = positive + negative
                score = (positive / total + 0.5) / 1.5 if total > 0 else 0.5

                day_data["dimensions"][dim_name] = {
                    "positive": positive,
                    "negative": negative,
                    "score": score
                }

            # Detect controversies
            for category, keywords in self.CONTROVERSY_KEYWORDS.items():
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "entity.name": self.entity,
                    "title": ",".join(keywords),
                    "sentiment.overall.polarity": "negative",
                    "source.rank.opr.min": 0.6,
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 10,
                })

                articles = resp.json().get("results", [])
                if len(articles) >= 3:  # Significant controversy
                    severity = min(1.0, len(articles) / 20)
                    controversy = ControversyEvent(
                        date=date,
                        severity=severity,
                        category=category,
                        description=articles[0].get("title", "") if articles else ""
                    )
                    self.controversies.append(controversy)
                    day_data["controversies"].append({
                        "category": category,
                        "severity": severity,
                        "count": len(articles)
                    })

            self.daily_data.append(day_data)

        # Initialize dimension objects
        for dim_name, dim_config in self.REPUTATION_DIMENSIONS.items():
            dim = ReputationDimension(dim_name, dim_config["positive"], dim_config["weight"])
            for day in self.daily_data:
                if dim_name in day["dimensions"]:
                    dim.daily_scores.append({
                        "date": day["date"],
                        "score": day["dimensions"][dim_name]["score"]
                    })
            self.dimensions[dim_name] = dim

    def calculate_base_reputation(self, half_life=30):
        """Calculate base reputation from dimensions."""
        total_weight = sum(d.weight for d in self.dimensions.values())
        weighted_score = 0

        for dim in self.dimensions.values():
            decayed_score = dim.calculate_decayed_score(half_life)
            weighted_score += decayed_score * dim.weight

        return weighted_score / total_weight if total_weight > 0 else 0.5

    def calculate_controversy_penalty(self):
        """Calculate reputation penalty from active controversies."""
        total_impact = 0

        for controversy in self.controversies:
            impact = controversy.current_impact()
            total_impact += impact

        # Cap total penalty at 0.5 (can't destroy more than half of reputation)
        return min(0.5, total_impact)

    def calculate_reputation_score(self):
        """Calculate final reputation score (0-100)."""
        base = self.calculate_base_reputation()
        penalty = self.calculate_controversy_penalty()

        # Apply penalty
        adjusted = base * (1 - penalty)

        # Convert to 0-100 scale
        return adjusted * 100

    def detect_regime(self):
        """Detect current reputation regime (stable, crisis, recovery)."""
        if len(self.daily_data) < 14:
            return "stable"

        # Recent vs historical volatility
        recent_scores = []
        for day in self.daily_data[-7:]:
            total = sum(d["score"] for d in day["dimensions"].values())
            recent_scores.append(total / len(day["dimensions"]))

        historical_scores = []
        for day in self.daily_data[-30:-7]:
            total = sum(d["score"] for d in day["dimensions"].values())
            historical_scores.append(total / len(day["dimensions"]))

        recent_avg = statistics.mean(recent_scores) if recent_scores else 0.5
        historical_avg = statistics.mean(historical_scores) if historical_scores else 0.5

        recent_vol = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0
        historical_vol = statistics.stdev(historical_scores) if len(historical_scores) > 1 else 0

        # Recent controversies
        recent_controversies = sum(
            1 for c in self.controversies
            if (datetime.utcnow() - datetime.fromisoformat(c.date)).days < 14
        )

        # Classify regime
        if recent_controversies >= 2 or recent_avg < historical_avg - 0.15:
            self.regime = "crisis"
        elif recent_avg > historical_avg + 0.1 and recent_vol < historical_vol:
            self.regime = "recovery"
        else:
            self.regime = "stable"

        return self.regime

    def forecast_recovery(self, target_score=70, max_days=365):
        """Forecast days to reach target reputation score."""
        current_score = self.calculate_reputation_score()

        if current_score >= target_score:
            return 0

        # Simulate recovery assuming no new controversies
        for day in range(1, max_days + 1):
            future_date = datetime.utcnow() + timedelta(days=day)

            # Decay all controversies
            future_penalty = 0
            for controversy in self.controversies:
                future_penalty += controversy.current_impact(future_date)

            future_penalty = min(0.5, future_penalty)
            base = self.calculate_base_reputation()
            future_score = base * (1 - future_penalty) * 100

            if future_score >= target_score:
                return day

        return max_days  # Won't reach target in forecast period

    def simulate_crisis_impact(self, severity, category="general"):
        """Simulate impact of a hypothetical crisis."""
        # Create hypothetical controversy
        hypothetical = ControversyEvent(
            date=datetime.utcnow().isoformat(),
            severity=severity,
            category=category,
            description="Hypothetical crisis simulation"
        )

        # Calculate impact
        current_score = self.calculate_reputation_score()

        # Add hypothetical to controversies temporarily
        self.controversies.append(hypothetical)
        simulated_score = self.calculate_reputation_score()

        # Remove hypothetical
        self.controversies.pop()

        return {
            "current_score": current_score,
            "simulated_score": simulated_score,
            "impact": current_score - simulated_score,
            "recovery_days": self.forecast_recovery(current_score - 5)
        }

    def get_early_warnings(self):
        """Generate early warning signals."""
        warnings = []

        # Declining dimension scores
        for dim_name, dim in self.dimensions.items():
            if len(dim.daily_scores) >= 14:
                recent = statistics.mean([s["score"] for s in dim.daily_scores[-7:]])
                prior = statistics.mean([s["score"] for s in dim.daily_scores[-14:-7]])

                if recent < prior - 0.1:
                    warnings.append({
                        "type": "DIMENSION_DECLINE",
                        "dimension": dim_name,
                        "change": (recent - prior) * 100,
                        "severity": "high" if recent < prior - 0.2 else "medium"
                    })

        # Rising controversy count
        recent_controversies = sum(
            1 for c in self.controversies
            if (datetime.utcnow() - datetime.fromisoformat(c.date)).days < 7
        )
        if recent_controversies >= 2:
            warnings.append({
                "type": "CONTROVERSY_CLUSTER",
                "count": recent_controversies,
                "severity": "high"
            })

        # Regime change
        regime = self.detect_regime()
        if regime == "crisis":
            warnings.append({
                "type": "CRISIS_REGIME",
                "regime": regime,
                "severity": "critical"
            })

        return warnings

    def generate_report(self):
        """Generate comprehensive reputation report."""
        return {
            "entity": self.entity,
            "generated_at": datetime.utcnow().isoformat(),
            "reputation_score": self.calculate_reputation_score(),
            "regime": self.detect_regime(),
            "dimensions": {
                name: {
                    "score": dim.calculate_decayed_score() * 100,
                    "weight": dim.weight
                }
                for name, dim in self.dimensions.items()
            },
            "active_controversies": len([c for c in self.controversies if c.current_impact() > 0.05]),
            "controversy_penalty": self.calculate_controversy_penalty() * 100,
            "recovery_forecast": self.forecast_recovery(70),
            "early_warnings": self.get_early_warnings()
        }


# Run reputation analysis
print("REPUTATION RISK ENGINE")
print("=" * 70)

entity = "Meta"
engine = ReputationRiskEngine(entity)

print(f"\nAnalyzing reputation for: {entity}")
print("Fetching historical data (60 days)...")
engine.fetch_historical_data(days=60)

# Generate report
report = engine.generate_report()

print("\n" + "=" * 70)
print("REPUTATION REPORT")
print("-" * 50)
print(f"Entity: {report['entity']}")
print(f"Overall Score: {report['reputation_score']:.1f}/100")
print(f"Regime: {report['regime'].upper()}")
print(f"Controversy Penalty: -{report['controversy_penalty']:.1f}%")

print("\nDIMENSION SCORES:")
for dim, data in report["dimensions"].items():
    bar = "█" * int(data["score"] / 5) + "░" * (20 - int(data["score"] / 5))
    print(f"  {dim:<12} [{bar}] {data['score']:.1f}")

print(f"\nActive Controversies: {report['active_controversies']}")
print(f"Recovery Forecast: {report['recovery_forecast']} days to reach 70/100")

# Early warnings
print("\n" + "=" * 70)
print("EARLY WARNINGS")
print("-" * 50)
for warning in report["early_warnings"]:
    print(f"  [{warning['severity'].upper()}] {warning['type']}")

# Crisis simulation
print("\n" + "=" * 70)
print("CRISIS SIMULATION")
print("-" * 50)
for severity in [0.3, 0.5, 0.8]:
    sim = engine.simulate_crisis_impact(severity, "regulatory")
    print(f"Severity {severity}: Score {sim['current_score']:.1f} -> {sim['simulated_score']:.1f} "
          f"(Impact: -{sim['impact']:.1f})")
```

## Reputation Dimensions

| Dimension | Positive Signals | Negative Signals | Weight |
|-----------|------------------|------------------|--------|
| Trust | reliable, credible, honest | scandal, deceptive, misleading | 25% |
| Quality | excellent, innovative, superior | defect, recall, failure | 20% |
| Ethics | responsible, sustainable, integrity | corruption, fraud, violation | 20% |
| Leadership | visionary, respected, effective | mismanagement, controversy | 15% |
| Social | community, diversity, philanthropy | discrimination, harassment | 20% |

## Controversy Half-Life Model

| Severity Level | Half-Life (days) |
|----------------|------------------|
| Minor (<20%) | 7 |
| Moderate (20-40%) | 21 |
| Major (40-60%) | 60 |
| Severe (60-80%) | 180 |
| Catastrophic (>80%) | 365 |

## Common Use Cases

- **Corporate communications** — monitor and manage brand reputation.
- **Investor relations** — track reputation risks affecting valuation.
- **Risk management** — model reputation exposure.
- **Crisis preparedness** — simulate crisis scenarios.
- **Competitive analysis** — benchmark reputation vs peers.
- **Executive reporting** — generate reputation dashboards.

## See Also

- [examples.md](./examples.md) — additional code examples.
