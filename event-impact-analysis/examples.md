# Event Impact Analysis — Code Examples

Advanced examples for building event impact measurement and analysis systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Comprehensive Event Impact Analyzer

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class EventImpactAnalyzer:
    """Comprehensive event impact analysis system."""

    def __init__(self, entity, event_date, event_keywords=None):
        self.entity = entity
        self.event_date = datetime.fromisoformat(event_date)
        self.event_keywords = event_keywords or []
        self.periods = {}
        self.timeline = []

    def fetch_period_metrics(self, start_date, end_date, period_name):
        """Fetch comprehensive metrics for a time period."""

        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        days = (end_date - start_date).days or 1

        metrics = {
            "period": period_name,
            "start": start,
            "end": end,
            "days": days,
        }

        # Total coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.entity,
            "entity.type": "organization",
            "published_at.start": start,
            "published_at.end": end,
            "language": "en",
            "per_page": 1,
        })
        metrics["total"] = resp.json().get("total_results", 0)
        metrics["daily_avg"] = metrics["total"] / days

        # Sentiment breakdown
        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.entity,
                "sentiment.overall.polarity": polarity,
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 1,
            })
            metrics[polarity] = resp.json().get("total_results", 0)

        # Calculate ratios
        total = metrics["total"] or 1
        metrics["positive_ratio"] = metrics["positive"] / total
        metrics["negative_ratio"] = metrics["negative"] / total
        metrics["net_sentiment"] = (metrics["positive"] - metrics["negative"]) / total

        # Tier-1 coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.entity,
            "source.rank.opr.min": 0.7,
            "published_at.start": start,
            "published_at.end": end,
            "per_page": 1,
        })
        metrics["tier1"] = resp.json().get("total_results", 0)
        metrics["tier1_ratio"] = metrics["tier1"] / total

        # Event-specific coverage
        if self.event_keywords:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.entity,
                "title": ",".join(self.event_keywords),
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 1,
            })
            metrics["event_specific"] = resp.json().get("total_results", 0)
            metrics["event_dominance"] = metrics["event_specific"] / total

        self.periods[period_name] = metrics
        return metrics

    def build_daily_timeline(self, days_before=7, days_after=21):
        """Build day-by-day timeline of coverage."""

        timeline = []
        start = self.event_date - timedelta(days=days_before)

        for day_offset in range(days_before + days_after + 1):
            current_day = start + timedelta(days=day_offset)
            next_day = current_day + timedelta(days=1)

            day_start = current_day.strftime("%Y-%m-%d")
            day_end = next_day.strftime("%Y-%m-%d")

            # Get daily metrics
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.entity,
                "entity.type": "organization",
                "published_at.start": day_start,
                "published_at.end": day_end,
                "language": "en",
                "per_page": 1,
            })
            total = resp.json().get("total_results", 0)

            # Get sentiment
            negative = 0
            if total > 0:
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "entity.name": self.entity,
                    "sentiment.overall.polarity": "negative",
                    "published_at.start": day_start,
                    "published_at.end": day_end,
                    "language": "en",
                    "per_page": 1,
                })
                negative = resp.json().get("total_results", 0)

            day_data = {
                "date": day_start,
                "day_offset": day_offset - days_before,
                "is_event_day": day_offset == days_before,
                "total": total,
                "negative": negative,
                "negative_ratio": negative / max(total, 1),
            }

            timeline.append(day_data)

        self.timeline = timeline
        return timeline

    def calculate_impact_metrics(self):
        """Calculate comprehensive impact metrics."""

        if "before" not in self.periods or "after" not in self.periods:
            raise ValueError("Must fetch before and after periods first")

        before = self.periods["before"]
        after = self.periods["after"]

        impact = {
            # Volume impact
            "volume_multiplier": after["daily_avg"] / max(before["daily_avg"], 0.1),
            "absolute_increase": after["total"] - before["total"],

            # Sentiment impact
            "negativity_shift": after["negative_ratio"] - before["negative_ratio"],
            "sentiment_change": after["net_sentiment"] - before["net_sentiment"],

            # Reach impact
            "tier1_shift": after["tier1_ratio"] - before["tier1_ratio"],
        }

        # Event dominance
        if "event_specific" in after:
            impact["event_dominance"] = after["event_dominance"]

        # Peak detection from timeline
        if self.timeline:
            max_day = max(self.timeline, key=lambda x: x["total"])
            impact["peak_day"] = max_day["date"]
            impact["peak_volume"] = max_day["total"]
            impact["days_to_peak"] = max_day["day_offset"]

            # Recovery detection (first day negative ratio returns to baseline)
            baseline_neg = before["negative_ratio"]
            for day in self.timeline:
                if day["day_offset"] > 0 and day["negative_ratio"] <= baseline_neg * 1.1:
                    impact["recovery_day"] = day["date"]
                    impact["days_to_recovery"] = day["day_offset"]
                    break

        return impact

    def generate_report(self, before_days=7, after_days=21):
        """Generate comprehensive event impact report."""

        # Fetch periods
        before_start = self.event_date - timedelta(days=before_days)
        after_end = self.event_date + timedelta(days=after_days)

        self.fetch_period_metrics(before_start, self.event_date, "before")
        self.fetch_period_metrics(self.event_date, after_end, "after")

        # Build timeline
        self.build_daily_timeline(before_days, after_days)

        # Calculate impact
        impact = self.calculate_impact_metrics()

        return {
            "entity": self.entity,
            "event_date": self.event_date.strftime("%Y-%m-%d"),
            "event_keywords": self.event_keywords,
            "periods": self.periods,
            "impact": impact,
            "timeline": self.timeline,
        }


# Analyze event
entity = "Boeing"
event_date = "2024-01-05"
keywords = ["door", "plug", "incident", "grounded", "737", "Alaska Airlines"]

analyzer = EventImpactAnalyzer(entity, event_date, keywords)
report = analyzer.generate_report(before_days=7, after_days=21)

print("=" * 70)
print("EVENT IMPACT ANALYSIS REPORT")
print("=" * 70)
print(f"\nEntity: {report['entity']}")
print(f"Event Date: {report['event_date']}")
print(f"Keywords: {', '.join(report['event_keywords'])}")

print("\n" + "-" * 50)
print("PERIOD COMPARISON")
print("-" * 50)

before = report["periods"]["before"]
after = report["periods"]["after"]

print(f"\n{'Metric':<20} {'Before':>12} {'After':>12} {'Change':>12}")
print("-" * 56)
print(f"{'Total Coverage':<20} {before['total']:>12,} {after['total']:>12,} "
      f"{after['total'] - before['total']:>+12,}")
print(f"{'Daily Average':<20} {before['daily_avg']:>12.1f} {after['daily_avg']:>12.1f} "
      f"{after['daily_avg'] - before['daily_avg']:>+12.1f}")
print(f"{'Negative Ratio':<20} {before['negative_ratio']:>11.1%} {after['negative_ratio']:>11.1%} "
      f"{after['negative_ratio'] - before['negative_ratio']:>+11.1%}")
print(f"{'Net Sentiment':<20} {before['net_sentiment']:>+11.3f} {after['net_sentiment']:>+11.3f} "
      f"{after['net_sentiment'] - before['net_sentiment']:>+11.3f}")

print("\n" + "-" * 50)
print("IMPACT METRICS")
print("-" * 50)

impact = report["impact"]
print(f"\n  Volume Multiplier: {impact['volume_multiplier']:.1f}x")
print(f"  Negativity Shift: {impact['negativity_shift']:+.1%}")
print(f"  Sentiment Change: {impact['sentiment_change']:+.3f}")

if "event_dominance" in impact:
    print(f"  Event Dominance: {impact['event_dominance']:.1%}")

if "peak_day" in impact:
    print(f"\n  Peak Day: {impact['peak_day']} (Day +{impact['days_to_peak']})")
    print(f"  Peak Volume: {impact['peak_volume']} articles")

if "recovery_day" in impact:
    print(f"  Recovery Day: {impact['recovery_day']} (Day +{impact['days_to_recovery']})")
else:
    print("  Recovery: Not yet detected")

print("\n" + "-" * 50)
print("DAILY TIMELINE (Volume)")
print("-" * 50)

for day in report["timeline"]:
    marker = " <<<EVENT" if day["is_event_day"] else ""
    bar = "█" * min(50, day["total"] // 10)
    print(f"  {day['date']} | {day['total']:>5} | {bar}{marker}")
```

### Cross-Entity Spillover Analysis

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def analyze_spillover(primary_entity, related_entities, event_date, before_days=7, after_days=14):
    """Analyze how an event affecting one entity spills over to related entities."""

    event = datetime.fromisoformat(event_date)
    before_start = (event - timedelta(days=before_days)).strftime("%Y-%m-%d")
    after_end = (event + timedelta(days=after_days)).strftime("%Y-%m-%d")

    results = {
        "primary": primary_entity,
        "event_date": event_date,
        "entities": {},
    }

    all_entities = [primary_entity] + related_entities

    for entity in all_entities:
        entity_data = {
            "before": {"total": 0, "negative": 0},
            "after": {"total": 0, "negative": 0},
        }

        # Before
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "entity.type": "organization",
            "published_at.start": before_start,
            "published_at.end": event_date,
            "language": "en",
            "per_page": 1,
        })
        entity_data["before"]["total"] = resp.json().get("total_results", 0)

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "sentiment.overall.polarity": "negative",
            "published_at.start": before_start,
            "published_at.end": event_date,
            "language": "en",
            "per_page": 1,
        })
        entity_data["before"]["negative"] = resp.json().get("total_results", 0)

        # After
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "entity.type": "organization",
            "published_at.start": event_date,
            "published_at.end": after_end,
            "language": "en",
            "per_page": 1,
        })
        entity_data["after"]["total"] = resp.json().get("total_results", 0)

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "sentiment.overall.polarity": "negative",
            "published_at.start": event_date,
            "published_at.end": after_end,
            "language": "en",
            "per_page": 1,
        })
        entity_data["after"]["negative"] = resp.json().get("total_results", 0)

        # Calculate changes
        before_total = entity_data["before"]["total"] or 1
        after_total = entity_data["after"]["total"] or 1

        entity_data["volume_change"] = (after_total - before_total) / before_total
        entity_data["neg_ratio_before"] = entity_data["before"]["negative"] / before_total
        entity_data["neg_ratio_after"] = entity_data["after"]["negative"] / after_total
        entity_data["neg_shift"] = entity_data["neg_ratio_after"] - entity_data["neg_ratio_before"]

        entity_data["is_primary"] = entity == primary_entity

        results["entities"][entity] = entity_data

    return results

# Analyze Boeing event spillover
primary = "Boeing"
related = ["Airbus", "Spirit AeroSystems", "Alaska Airlines", "FAA"]
event_date = "2024-01-05"

print("CROSS-ENTITY SPILLOVER ANALYSIS")
print("=" * 70)
print(f"Primary Entity: {primary}")
print(f"Event Date: {event_date}\n")

spillover = analyze_spillover(primary, related, event_date)

print(f"{'Entity':<20} {'Vol Change':>12} {'Neg Before':>12} {'Neg After':>12} {'Neg Shift':>12}")
print("-" * 70)

# Sort by volume change
sorted_entities = sorted(
    spillover["entities"].items(),
    key=lambda x: x[1]["volume_change"],
    reverse=True
)

for entity, data in sorted_entities:
    marker = " <<<PRIMARY" if data["is_primary"] else ""
    print(f"{entity:<20} {data['volume_change']:>+11.1%} {data['neg_ratio_before']:>11.1%} "
          f"{data['neg_ratio_after']:>11.1%} {data['neg_shift']:>+11.1%}{marker}")

print("\nSPILLOVER SUMMARY:")
print("-" * 50)

primary_data = spillover["entities"][primary]
print(f"  Primary Impact: {primary_data['volume_change']:+.1%} volume, {primary_data['neg_shift']:+.1%} negativity")

for entity, data in sorted_entities:
    if not data["is_primary"] and abs(data["volume_change"]) > 0.1:
        print(f"  {entity}: {data['volume_change']:+.1%} volume spillover")
```

---

## JavaScript

### Event Impact Dashboard

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function fetchPeriodMetrics(entity, startDate, endDate) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": entity,
    "entity.type": "organization",
    "published_at.start": startDate,
    "published_at.end": endDate,
    language: "en",
    per_page: "1",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  const total = data.total_results || 0;

  // Get negative
  params.set("sentiment.overall.polarity", "negative");
  const negResponse = await fetch(`${BASE_URL}?${params}`);
  const negData = await negResponse.json();
  const negative = negData.total_results || 0;

  return {
    total,
    negative,
    negativeRatio: negative / Math.max(total, 1),
  };
}

async function analyzeEventImpact(entity, eventDate, beforeDays = 7, afterDays = 14) {
  const event = new Date(eventDate);
  const beforeStart = new Date(event.getTime() - beforeDays * 24 * 60 * 60 * 1000);
  const afterEnd = new Date(event.getTime() + afterDays * 24 * 60 * 60 * 1000);

  const before = await fetchPeriodMetrics(
    entity,
    beforeStart.toISOString().split("T")[0],
    eventDate
  );

  const after = await fetchPeriodMetrics(
    entity,
    eventDate,
    afterEnd.toISOString().split("T")[0]
  );

  const beforeDaily = before.total / beforeDays;
  const afterDaily = after.total / afterDays;

  return {
    entity,
    eventDate,
    before,
    after,
    impact: {
      volumeMultiplier: afterDaily / Math.max(beforeDaily, 0.1),
      absoluteIncrease: after.total - before.total,
      negativityShift: after.negativeRatio - before.negativeRatio,
    },
  };
}

async function runAnalysis() {
  console.log("EVENT IMPACT DASHBOARD");
  console.log("=".repeat(50));

  const analysis = await analyzeEventImpact("Boeing", "2024-01-05", 7, 14);

  console.log(`\nEntity: ${analysis.entity}`);
  console.log(`Event: ${analysis.eventDate}`);

  console.log("\nBEFORE:");
  console.log(`  Total: ${analysis.before.total}`);
  console.log(`  Negative: ${(analysis.before.negativeRatio * 100).toFixed(1)}%`);

  console.log("\nAFTER:");
  console.log(`  Total: ${analysis.after.total}`);
  console.log(`  Negative: ${(analysis.after.negativeRatio * 100).toFixed(1)}%`);

  console.log("\nIMPACT:");
  console.log(`  Volume Multiplier: ${analysis.impact.volumeMultiplier.toFixed(1)}x`);
  console.log(`  Absolute Increase: ${analysis.impact.absoluteIncrease > 0 ? "+" : ""}${analysis.impact.absoluteIncrease}`);
  console.log(`  Negativity Shift: ${analysis.impact.negativityShift >= 0 ? "+" : ""}${(analysis.impact.negativityShift * 100).toFixed(1)}%`);
}

runAnalysis();
```

---

## PHP

### Event Impact Report Generator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function fetchPeriodMetrics(string $entity, string $start, string $end): array
{
    global $apiKey, $baseUrl;

    // Total
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $entity,
        "entity.type"        => "organization",
        "published_at.start" => $start,
        "published_at.end"   => $end,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $total = $data["total_results"] ?? 0;

    // Negative
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $entity,
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "published_at.end"           => $end,
        "language"                   => "en",
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $negative = $data["total_results"] ?? 0;

    return [
        "total"          => $total,
        "negative"       => $negative,
        "negative_ratio" => $negative / max($total, 1),
    ];
}

function analyzeEventImpact(string $entity, string $eventDate, int $beforeDays = 7, int $afterDays = 14): array
{
    $event = new DateTime($eventDate);
    $beforeStart = (clone $event)->modify("-{$beforeDays} days")->format("Y-m-d");
    $afterEnd = (clone $event)->modify("+{$afterDays} days")->format("Y-m-d");

    $before = fetchPeriodMetrics($entity, $beforeStart, $eventDate);
    $after = fetchPeriodMetrics($entity, $eventDate, $afterEnd);

    $beforeDaily = $before["total"] / $beforeDays;
    $afterDaily = $after["total"] / $afterDays;

    return [
        "entity"     => $entity,
        "event_date" => $eventDate,
        "before"     => $before,
        "after"      => $after,
        "impact"     => [
            "volume_multiplier"  => $afterDaily / max($beforeDaily, 0.1),
            "absolute_increase"  => $after["total"] - $before["total"],
            "negativity_shift"   => $after["negative_ratio"] - $before["negative_ratio"],
        ],
    ];
}

$analysis = analyzeEventImpact("Boeing", "2024-01-05", 7, 14);

echo "EVENT IMPACT REPORT\n";
echo str_repeat("=", 50) . "\n";
echo "Entity: {$analysis['entity']}\n";
echo "Event: {$analysis['event_date']}\n";

echo "\nBEFORE:\n";
echo "  Total: {$analysis['before']['total']}\n";
printf("  Negative: %.1f%%\n", $analysis["before"]["negative_ratio"] * 100);

echo "\nAFTER:\n";
echo "  Total: {$analysis['after']['total']}\n";
printf("  Negative: %.1f%%\n", $analysis["after"]["negative_ratio"] * 100);

echo "\nIMPACT:\n";
printf("  Volume Multiplier: %.1fx\n", $analysis["impact"]["volume_multiplier"]);
printf("  Absolute Increase: %+d\n", $analysis["impact"]["absolute_increase"]);
printf("  Negativity Shift: %+.1f%%\n", $analysis["impact"]["negativity_shift"] * 100);
```
