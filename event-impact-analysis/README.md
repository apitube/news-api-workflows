# Event Impact Analysis

Workflow for measuring the impact of specific events on media coverage, tracking narrative evolution, analyzing ripple effects across entities, and quantifying recovery patterns using the [APITube News API](https://apitube.io).

## Overview

The **Event Impact Analysis** workflow provides sophisticated before/after analysis of events by comparing coverage metrics across time periods, tracking how narratives spread and evolve, measuring cross-entity spillover effects, and analyzing sentiment recovery trajectories. Ideal for crisis communications, campaign measurement, product launch analysis, PR effectiveness assessment, and market event research.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by affected entity.                                           |
| `entity.type`                 | string  | Filter by type: `organization`, `person`, `product`.                |
| `title`                       | string  | Filter by event-related keywords.                                    |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `source.domain`               | string  | Filter by specific media outlets.                                    |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Get coverage before event
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Boeing&published_at.start=2024-01-01&published_at.end=2024-01-05&language=en&per_page=1" | jq '.total_results'

# Get coverage after event
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Boeing&published_at.start=2024-01-05&published_at.end=2024-01-15&language=en&per_page=1" | jq '.total_results'

# Track event-specific keywords
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Boeing&title=door,plug,incident,grounded&published_at.start=2024-01-05&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def analyze_event_impact(entity, event_date, event_keywords, before_days=7, after_days=14):
    """Analyze the impact of an event on media coverage."""

    event = datetime.fromisoformat(event_date)
    before_start = (event - timedelta(days=before_days)).strftime("%Y-%m-%d")
    before_end = event.strftime("%Y-%m-%d")
    after_start = event.strftime("%Y-%m-%d")
    after_end = (event + timedelta(days=after_days)).strftime("%Y-%m-%d")

    results = {
        "entity": entity,
        "event_date": event_date,
        "before": {},
        "after": {},
        "impact": {},
    }

    # Before period
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity,
        "entity.type": "organization",
        "published_at.start": before_start,
        "published_at.end": before_end,
        "language": "en",
        "per_page": 1,
    })
    results["before"]["total"] = resp.json().get("total_results", 0)

    for polarity in ["positive", "negative"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "sentiment.overall.polarity": polarity,
            "published_at.start": before_start,
            "published_at.end": before_end,
            "language": "en",
            "per_page": 1,
        })
        results["before"][polarity] = resp.json().get("total_results", 0)

    # After period
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity,
        "entity.type": "organization",
        "published_at.start": after_start,
        "published_at.end": after_end,
        "language": "en",
        "per_page": 1,
    })
    results["after"]["total"] = resp.json().get("total_results", 0)

    for polarity in ["positive", "negative"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity,
            "sentiment.overall.polarity": polarity,
            "published_at.start": after_start,
            "published_at.end": after_end,
            "language": "en",
            "per_page": 1,
        })
        results["after"][polarity] = resp.json().get("total_results", 0)

    # Event-specific coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity,
        "title": ",".join(event_keywords),
        "published_at.start": after_start,
        "published_at.end": after_end,
        "language": "en",
        "per_page": 1,
    })
    results["after"]["event_specific"] = resp.json().get("total_results", 0)

    # Calculate impact metrics
    before_daily = results["before"]["total"] / before_days
    after_daily = results["after"]["total"] / after_days

    results["impact"]["volume_multiplier"] = after_daily / max(before_daily, 0.1)

    before_neg_ratio = results["before"]["negative"] / max(results["before"]["total"], 1)
    after_neg_ratio = results["after"]["negative"] / max(results["after"]["total"], 1)

    results["impact"]["negativity_shift"] = after_neg_ratio - before_neg_ratio
    results["impact"]["event_dominance"] = results["after"]["event_specific"] / max(results["after"]["total"], 1)

    return results

# Analyze event
entity = "Boeing"
event_date = "2024-01-05"  # Door plug incident
keywords = ["door", "plug", "incident", "grounded", "Alaska Airlines"]

print("EVENT IMPACT ANALYSIS")
print("=" * 60)
print(f"Entity: {entity}")
print(f"Event Date: {event_date}\n")

analysis = analyze_event_impact(entity, event_date, keywords)

print("BEFORE EVENT (7 days prior):")
print(f"  Total Coverage: {analysis['before']['total']}")
print(f"  Positive: {analysis['before']['positive']} | Negative: {analysis['before']['negative']}")

print("\nAFTER EVENT (14 days post):")
print(f"  Total Coverage: {analysis['after']['total']}")
print(f"  Positive: {analysis['after']['positive']} | Negative: {analysis['after']['negative']}")
print(f"  Event-Specific: {analysis['after']['event_specific']}")

print("\nIMPACT METRICS:")
print(f"  Volume Multiplier: {analysis['impact']['volume_multiplier']:.1f}x")
print(f"  Negativity Shift: {analysis['impact']['negativity_shift']:+.1%}")
print(f"  Event Dominance: {analysis['impact']['event_dominance']:.1%}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function analyzeEventImpact(entity, eventDate, keywords, beforeDays = 7, afterDays = 14) {
  const event = new Date(eventDate);
  const beforeStart = new Date(event.getTime() - beforeDays * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
  const afterEnd = new Date(event.getTime() + afterDays * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const results = { before: {}, after: {}, impact: {} };

  // Before period - total
  let params = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": entity,
    "published_at.start": beforeStart,
    "published_at.end": eventDate,
    language: "en",
    per_page: "1",
  });

  let response = await fetch(`${BASE_URL}?${params}`);
  let data = await response.json();
  results.before.total = data.total_results || 0;

  // After period - total
  params = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": entity,
    "published_at.start": eventDate,
    "published_at.end": afterEnd,
    language: "en",
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${params}`);
  data = await response.json();
  results.after.total = data.total_results || 0;

  // Calculate multiplier
  const beforeDaily = results.before.total / beforeDays;
  const afterDaily = results.after.total / afterDays;
  results.impact.volumeMultiplier = afterDaily / Math.max(beforeDaily, 0.1);

  return results;
}

async function runAnalysis() {
  const analysis = await analyzeEventImpact("Boeing", "2024-01-05", ["door", "plug"], 7, 14);

  console.log("EVENT IMPACT ANALYSIS");
  console.log("=".repeat(50));
  console.log(`Before: ${analysis.before.total} articles`);
  console.log(`After: ${analysis.after.total} articles`);
  console.log(`Volume Multiplier: ${analysis.impact.volumeMultiplier.toFixed(1)}x`);
}

runAnalysis();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function analyzeEventImpact(string $entity, string $eventDate, int $beforeDays = 7, int $afterDays = 14): array
{
    global $apiKey, $baseUrl;

    $event = new DateTime($eventDate);
    $beforeStart = (clone $event)->modify("-{$beforeDays} days")->format("Y-m-d");
    $afterEnd = (clone $event)->modify("+{$afterDays} days")->format("Y-m-d");

    // Before
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $entity,
        "published_at.start" => $beforeStart,
        "published_at.end"   => $eventDate,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $beforeTotal = $data["total_results"] ?? 0;

    // After
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $entity,
        "published_at.start" => $eventDate,
        "published_at.end"   => $afterEnd,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $afterTotal = $data["total_results"] ?? 0;

    $beforeDaily = $beforeTotal / $beforeDays;
    $afterDaily = $afterTotal / $afterDays;

    return [
        "before_total"      => $beforeTotal,
        "after_total"       => $afterTotal,
        "volume_multiplier" => $afterDaily / max($beforeDaily, 0.1),
    ];
}

$analysis = analyzeEventImpact("Boeing", "2024-01-05", 7, 14);

echo "EVENT IMPACT ANALYSIS\n";
echo str_repeat("=", 50) . "\n";
echo "Before: {$analysis['before_total']} articles\n";
echo "After: {$analysis['after_total']} articles\n";
printf("Volume Multiplier: %.1fx\n", $analysis["volume_multiplier"]);
```

## Common Use Cases

- **Crisis impact measurement** — quantify how crises affect brand coverage.
- **Product launch analysis** — measure media response to announcements.
- **Campaign effectiveness** — compare pre/post campaign metrics.
- **Earnings impact** — analyze coverage changes around earnings.
- **M&A announcement effects** — track deal announcement ripple effects.
- **Executive change impact** — measure leadership transition coverage.
- **Regulatory event analysis** — assess policy announcement impacts.
- **Recovery tracking** — monitor sentiment normalization over time.

## See Also

- [examples.md](./examples.md) — detailed code examples for event impact analysis.
