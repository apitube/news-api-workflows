# Narrative Intelligence

Workflow for tracking how narratives emerge, evolve, and spread across media, detecting coordinated messaging campaigns, identifying narrative shifts, and analyzing framing patterns using the [APITube News API](https://apitube.io).

## Overview

The **Narrative Intelligence** workflow provides deep analysis of media narratives by tracking keyword evolution over time, detecting coordinated messaging patterns, identifying narrative frame shifts, measuring narrative penetration across source tiers, and analyzing counter-narrative dynamics. Combines temporal analysis, linguistic pattern detection, and source clustering to reveal narrative strategies. Ideal for communications teams, political analysts, brand managers, and disinformation researchers.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `title`                       | string  | Filter by narrative keywords.                                        |
| `entity.name`                 | string  | Filter by entity involved in narrative.                             |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `source.domain`               | string  | Filter by specific domains.                                          |
| `source.country`              | string  | Filter by source country.                                            |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Track narrative keywords over time
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=AI,job,replace,automate&language=en&per_page=30"

# Monitor competing narratives
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=climate,crisis,emergency&source.rank.opr.min=0.7&language=en&per_page=30"

# Detect narrative spread across sources
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=breakthrough,revolutionary&published_at.start=2024-01-01&language=en&per_page=50"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import re

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class NarrativeTracker:
    """Track and analyze media narrative evolution."""

    def __init__(self):
        self.narrative_data = defaultdict(list)
        self.source_patterns = defaultdict(lambda: defaultdict(int))

    def define_narrative(self, name, keywords, counter_keywords=None):
        """Define a narrative to track."""
        return {
            "name": name,
            "keywords": keywords,
            "counter_keywords": counter_keywords or [],
            "timeline": [],
            "sources": defaultdict(int),
            "sentiment_trend": []
        }

    def track_narrative(self, narrative, days=30):
        """Track narrative evolution over time."""
        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            # Main narrative
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(narrative["keywords"]),
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            main_count = resp.json().get("total_results", 0)

            # Counter narrative
            counter_count = 0
            if narrative["counter_keywords"]:
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "title": ",".join(narrative["counter_keywords"]),
                    "published_at.start": date,
                    "published_at.end": next_date,
                    "language": "en",
                    "per_page": 1,
                })
                counter_count = resp.json().get("total_results", 0)

            # Sentiment
            pos_resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(narrative["keywords"]),
                "sentiment.overall.polarity": "positive",
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            pos_count = pos_resp.json().get("total_results", 0)

            neg_resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(narrative["keywords"]),
                "sentiment.overall.polarity": "negative",
                "published_at.start": date,
                "published_at.end": next_date,
                "language": "en",
                "per_page": 1,
            })
            neg_count = neg_resp.json().get("total_results", 0)

            narrative["timeline"].append({
                "date": date,
                "main_count": main_count,
                "counter_count": counter_count,
                "positive": pos_count,
                "negative": neg_count,
                "sentiment_ratio": pos_count / max(pos_count + neg_count, 1)
            })

        return narrative

    def analyze_narrative_velocity(self, narrative):
        """Calculate how fast narrative is spreading."""
        timeline = narrative["timeline"]
        if len(timeline) < 7:
            return {"velocity": 0, "acceleration": 0}

        recent = sum(t["main_count"] for t in timeline[-7:])
        prior = sum(t["main_count"] for t in timeline[-14:-7])

        velocity = (recent - prior) / max(prior, 1) * 100

        # Acceleration (change in velocity)
        if len(timeline) >= 21:
            earlier = sum(t["main_count"] for t in timeline[-21:-14])
            prior_velocity = (prior - earlier) / max(earlier, 1) * 100
            acceleration = velocity - prior_velocity
        else:
            acceleration = 0

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "status": "accelerating" if acceleration > 10 else "decelerating" if acceleration < -10 else "steady"
        }

    def detect_narrative_shift(self, narrative):
        """Detect significant shifts in narrative sentiment or volume."""
        timeline = narrative["timeline"]
        shifts = []

        for i in range(7, len(timeline)):
            recent_avg = sum(t["main_count"] for t in timeline[i-7:i]) / 7
            prior_avg = sum(t["main_count"] for t in timeline[max(0, i-14):i-7]) / 7

            if prior_avg > 0:
                change = (recent_avg - prior_avg) / prior_avg

                if abs(change) > 0.5:  # 50% change
                    shifts.append({
                        "date": timeline[i]["date"],
                        "type": "surge" if change > 0 else "decline",
                        "magnitude": change * 100
                    })

            # Sentiment shift
            recent_sentiment = sum(t["sentiment_ratio"] for t in timeline[i-7:i]) / 7
            prior_sentiment = sum(t["sentiment_ratio"] for t in timeline[max(0, i-14):i-7]) / 7

            sentiment_change = recent_sentiment - prior_sentiment
            if abs(sentiment_change) > 0.15:
                shifts.append({
                    "date": timeline[i]["date"],
                    "type": "sentiment_shift",
                    "direction": "positive" if sentiment_change > 0 else "negative",
                    "magnitude": sentiment_change
                })

        return shifts

    def analyze_source_penetration(self, narrative, days=14):
        """Analyze how narrative penetrates different source tiers."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        tiers = {
            "tier1": {"min": 0.8, "count": 0},
            "tier2": {"min": 0.6, "max": 0.8, "count": 0},
            "tier3": {"min": 0.4, "max": 0.6, "count": 0},
            "tier4": {"min": 0.0, "max": 0.4, "count": 0},
        }

        for tier_name, tier in tiers.items():
            params = {
                "api_key": API_KEY,
                "title": ",".join(narrative["keywords"]),
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            }

            if "min" in tier:
                params["source.rank.opr.min"] = tier["min"]
            if "max" in tier:
                params["source.rank.opr.max"] = tier["max"]

            resp = requests.get(BASE_URL, params=params)
            tier["count"] = resp.json().get("total_results", 0)

        total = sum(t["count"] for t in tiers.values())

        return {
            "tiers": {k: {"count": v["count"], "share": v["count"] / max(total, 1)} for k, v in tiers.items()},
            "total": total,
            "elite_penetration": tiers["tier1"]["count"] / max(total, 1)
        }

# Run narrative tracking
print("NARRATIVE INTELLIGENCE ANALYSIS")
print("=" * 70)

tracker = NarrativeTracker()

# Define narratives
ai_disruption = tracker.define_narrative(
    "AI Job Disruption",
    keywords=["AI", "jobs", "replace", "automate", "unemployment"],
    counter_keywords=["AI", "jobs", "create", "augment", "opportunity"]
)

climate_narrative = tracker.define_narrative(
    "Climate Crisis",
    keywords=["climate", "crisis", "emergency", "catastrophe"],
    counter_keywords=["climate", "alarmism", "exaggeration", "natural"]
)

# Track narratives
print("\nTracking 'AI Job Disruption' narrative...")
ai_disruption = tracker.track_narrative(ai_disruption, days=30)

print("\nTracking 'Climate Crisis' narrative...")
climate_narrative = tracker.track_narrative(climate_narrative, days=30)

# Analysis
for narrative in [ai_disruption, climate_narrative]:
    print(f"\n{'='*60}")
    print(f"NARRATIVE: {narrative['name']}")
    print("-" * 40)

    # Timeline summary
    total_coverage = sum(t["main_count"] for t in narrative["timeline"])
    counter_coverage = sum(t["counter_count"] for t in narrative["timeline"])
    print(f"Total Coverage: {total_coverage:,} articles")
    print(f"Counter Narrative: {counter_coverage:,} articles")
    print(f"Dominance Ratio: {total_coverage / max(counter_coverage, 1):.1f}x")

    # Velocity
    velocity = tracker.analyze_narrative_velocity(narrative)
    print(f"\nVelocity: {velocity['velocity']:+.1f}% (week-over-week)")
    print(f"Acceleration: {velocity['acceleration']:+.1f}%")
    print(f"Status: {velocity['status'].upper()}")

    # Shifts
    shifts = tracker.detect_narrative_shift(narrative)
    print(f"\nNarrative Shifts Detected: {len(shifts)}")
    for shift in shifts[-3:]:
        print(f"  {shift['date']}: {shift['type']} ({shift.get('magnitude', 0):.1f})")

    # Source penetration
    penetration = tracker.analyze_source_penetration(narrative)
    print(f"\nSource Tier Penetration:")
    for tier, data in penetration["tiers"].items():
        print(f"  {tier}: {data['count']:,} ({data['share']:.1%})")
    print(f"Elite Media Share: {penetration['elite_penetration']:.1%}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class NarrativeAnalyzer {
  constructor() {
    this.narratives = new Map();
  }

  async trackNarrative(name, keywords, counterKeywords = [], days = 30) {
    const timeline = [];

    for (let i = days; i > 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];
      const nextDate = new Date(Date.now() - (i - 1) * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];

      // Main narrative
      const mainParams = new URLSearchParams({
        api_key: API_KEY,
        title: keywords.join(","),
        "published_at.start": date,
        "published_at.end": nextDate,
        language: "en",
        per_page: "1",
      });

      let response = await fetch(`${BASE_URL}?${mainParams}`);
      let data = await response.json();
      const mainCount = data.total_results || 0;

      // Counter narrative
      let counterCount = 0;
      if (counterKeywords.length > 0) {
        const counterParams = new URLSearchParams({
          api_key: API_KEY,
          title: counterKeywords.join(","),
          "published_at.start": date,
          "published_at.end": nextDate,
          language: "en",
          per_page: "1",
        });

        response = await fetch(`${BASE_URL}?${counterParams}`);
        data = await response.json();
        counterCount = data.total_results || 0;
      }

      timeline.push({ date, mainCount, counterCount });
    }

    const narrative = { name, keywords, counterKeywords, timeline };
    this.narratives.set(name, narrative);
    return narrative;
  }

  calculateVelocity(narrative) {
    const timeline = narrative.timeline;
    if (timeline.length < 14) return { velocity: 0, status: "insufficient_data" };

    const recent = timeline.slice(-7).reduce((sum, t) => sum + t.mainCount, 0);
    const prior = timeline.slice(-14, -7).reduce((sum, t) => sum + t.mainCount, 0);

    const velocity = prior > 0 ? ((recent - prior) / prior) * 100 : 0;

    return {
      velocity,
      recentTotal: recent,
      priorTotal: prior,
      status: velocity > 20 ? "surging" : velocity < -20 ? "declining" : "stable",
    };
  }

  detectShifts(narrative, threshold = 0.5) {
    const timeline = narrative.timeline;
    const shifts = [];

    for (let i = 7; i < timeline.length; i++) {
      const recentAvg =
        timeline.slice(i - 7, i).reduce((sum, t) => sum + t.mainCount, 0) / 7;
      const priorAvg =
        timeline.slice(Math.max(0, i - 14), i - 7).reduce((sum, t) => sum + t.mainCount, 0) / 7;

      if (priorAvg > 0) {
        const change = (recentAvg - priorAvg) / priorAvg;
        if (Math.abs(change) > threshold) {
          shifts.push({
            date: timeline[i].date,
            type: change > 0 ? "surge" : "decline",
            magnitude: change * 100,
          });
        }
      }
    }

    return shifts;
  }

  compareNarratives(narrative1, narrative2) {
    const total1 = narrative1.timeline.reduce((sum, t) => sum + t.mainCount, 0);
    const total2 = narrative2.timeline.reduce((sum, t) => sum + t.mainCount, 0);

    const velocity1 = this.calculateVelocity(narrative1);
    const velocity2 = this.calculateVelocity(narrative2);

    return {
      dominant: total1 > total2 ? narrative1.name : narrative2.name,
      ratio: total1 / Math.max(total2, 1),
      velocityComparison: {
        [narrative1.name]: velocity1.velocity,
        [narrative2.name]: velocity2.velocity,
      },
      momentum: velocity1.velocity > velocity2.velocity ? narrative1.name : narrative2.name,
    };
  }
}

async function runAnalysis() {
  const analyzer = new NarrativeAnalyzer();

  console.log("NARRATIVE INTELLIGENCE");
  console.log("=".repeat(50));

  console.log("\nTracking narratives...");

  const aiDisruption = await analyzer.trackNarrative(
    "AI Disruption",
    ["AI", "jobs", "replace", "automate"],
    ["AI", "create", "augment", "opportunity"],
    21
  );

  const aiOpportunity = await analyzer.trackNarrative(
    "AI Opportunity",
    ["AI", "productivity", "growth", "innovation"],
    [],
    21
  );

  // Analysis
  for (const narrative of [aiDisruption, aiOpportunity]) {
    console.log(`\n${"=".repeat(40)}`);
    console.log(`NARRATIVE: ${narrative.name}`);

    const total = narrative.timeline.reduce((sum, t) => sum + t.mainCount, 0);
    console.log(`Total coverage: ${total.toLocaleString()}`);

    const velocity = analyzer.calculateVelocity(narrative);
    console.log(`Velocity: ${velocity.velocity.toFixed(1)}% (${velocity.status})`);

    const shifts = analyzer.detectShifts(narrative);
    console.log(`Shifts detected: ${shifts.length}`);
    shifts.slice(-3).forEach((s) => {
      console.log(`  ${s.date}: ${s.type} (${s.magnitude.toFixed(1)}%)`);
    });
  }

  // Comparison
  console.log(`\n${"=".repeat(40)}`);
  console.log("NARRATIVE COMPARISON");
  const comparison = analyzer.compareNarratives(aiDisruption, aiOpportunity);
  console.log(`Dominant: ${comparison.dominant}`);
  console.log(`Ratio: ${comparison.ratio.toFixed(2)}x`);
  console.log(`Momentum leader: ${comparison.momentum}`);
}

runAnalysis();
```

### PHP

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class NarrativeTracker
{
    private string $apiKey;
    private string $baseUrl;

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function trackNarrative(string $name, array $keywords, array $counterKeywords = [], int $days = 30): array
    {
        $timeline = [];

        for ($i = $days; $i > 0; $i--) {
            $date = (new DateTime("-{$i} days"))->format("Y-m-d");
            $nextDate = (new DateTime("-" . ($i - 1) . " days"))->format("Y-m-d");

            // Main
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "title" => implode(",", $keywords),
                "published_at.start" => $date,
                "published_at.end" => $nextDate,
                "language" => "en",
                "per_page" => 1,
            ]);
            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $mainCount = $data["total_results"] ?? 0;

            // Counter
            $counterCount = 0;
            if (!empty($counterKeywords)) {
                $query = http_build_query([
                    "api_key" => $this->apiKey,
                    "title" => implode(",", $counterKeywords),
                    "published_at.start" => $date,
                    "published_at.end" => $nextDate,
                    "language" => "en",
                    "per_page" => 1,
                ]);
                $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
                $counterCount = $data["total_results"] ?? 0;
            }

            $timeline[] = ["date" => $date, "main" => $mainCount, "counter" => $counterCount];
        }

        return ["name" => $name, "keywords" => $keywords, "timeline" => $timeline];
    }

    public function calculateVelocity(array $narrative): array
    {
        $timeline = $narrative["timeline"];
        if (count($timeline) < 14) {
            return ["velocity" => 0, "status" => "insufficient_data"];
        }

        $recent = array_sum(array_column(array_slice($timeline, -7), "main"));
        $prior = array_sum(array_column(array_slice($timeline, -14, 7), "main"));

        $velocity = $prior > 0 ? (($recent - $prior) / $prior) * 100 : 0;

        return [
            "velocity" => $velocity,
            "status" => $velocity > 20 ? "surging" : ($velocity < -20 ? "declining" : "stable"),
        ];
    }

    public function detectShifts(array $narrative, float $threshold = 0.5): array
    {
        $timeline = $narrative["timeline"];
        $shifts = [];

        for ($i = 7; $i < count($timeline); $i++) {
            $recentAvg = array_sum(array_column(array_slice($timeline, $i - 7, 7), "main")) / 7;
            $priorAvg = array_sum(array_column(array_slice($timeline, max(0, $i - 14), 7), "main")) / 7;

            if ($priorAvg > 0) {
                $change = ($recentAvg - $priorAvg) / $priorAvg;
                if (abs($change) > $threshold) {
                    $shifts[] = [
                        "date" => $timeline[$i]["date"],
                        "type" => $change > 0 ? "surge" : "decline",
                        "magnitude" => $change * 100,
                    ];
                }
            }
        }

        return $shifts;
    }
}

$tracker = new NarrativeTracker();

echo "NARRATIVE INTELLIGENCE\n";
echo str_repeat("=", 50) . "\n";

$narrative = $tracker->trackNarrative(
    "AI Disruption",
    ["AI", "jobs", "replace", "automate"],
    ["AI", "create", "opportunity"],
    21
);

echo "\nNARRATIVE: {$narrative['name']}\n";

$total = array_sum(array_column($narrative["timeline"], "main"));
$counter = array_sum(array_column($narrative["timeline"], "counter"));
echo "Total: {$total} | Counter: {$counter}\n";

$velocity = $tracker->calculateVelocity($narrative);
printf("Velocity: %.1f%% (%s)\n", $velocity["velocity"], $velocity["status"]);

$shifts = $tracker->detectShifts($narrative);
echo "Shifts: " . count($shifts) . "\n";
foreach (array_slice($shifts, -3) as $s) {
    printf("  %s: %s (%.1f%%)\n", $s["date"], $s["type"], $s["magnitude"]);
}
```

## Common Use Cases

- **PR campaign measurement** — track how narratives spread after launches.
- **Crisis narrative control** — monitor competing narratives during crises.
- **Political messaging analysis** — analyze campaign narrative strategies.
- **Disinformation research** — detect coordinated narrative campaigns.
- **Brand narrative tracking** — monitor how brand stories evolve.
- **Competitive narrative intelligence** — track competitor messaging strategies.
- **Issue advocacy monitoring** — measure advocacy campaign effectiveness.
- **Media bias analysis** — compare narrative framing across outlets.

## See Also

- [examples.md](./examples.md) — detailed code examples for narrative intelligence.
