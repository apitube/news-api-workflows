# Industry Disruption Radar

Workflow for detecting emerging industry disruptions, tracking technology shifts, monitoring startup activity, identifying market transformations, and analyzing innovation trends using the [APITube News API](https://apitube.io).

## Overview

The **Industry Disruption Radar** workflow identifies early signals of industry transformation by tracking emerging technology coverage, monitoring startup funding and launches, analyzing incumbent response patterns, and detecting narrative shifts around disruptive themes. Combines topic trending, entity emergence detection, sentiment trajectory analysis, and cross-industry correlation to surface disruption signals. Ideal for corporate strategy, venture capital, innovation teams, and competitive intelligence.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
GET https://api.apitube.io/v1/news/topic
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `topic.id`                    | string  | Filter by technology/industry topic.                                |
| `industry.id`                 | string  | Filter by industry.                                                  |
| `entity.name`                 | string  | Filter by company or technology name.                               |
| `entity.type`                 | string  | Filter by type: `organization`, `product`, `technology`.            |
| `title`                       | string  | Filter by disruption-related keywords.                               |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Track AI disruption news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=artificial_intelligence&title=disrupt,transform,replace,automate&source.rank.opr.min=0.6&per_page=20"

# Monitor startup funding news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=startup,funding,Series A,Series B,raised,valuation&language=en&per_page=30"

# Track emerging technology coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=quantum_computing,blockchain,autonomous_vehicles&language=en&per_page=30"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

DISRUPTION_THEMES = {
    "AI & Automation": {
        "topics": ["artificial_intelligence", "machine_learning", "robotics"],
        "keywords": ["AI", "automation", "machine learning", "ChatGPT", "generative AI"],
    },
    "Clean Energy": {
        "topics": ["renewable_energy", "electric_vehicles", "batteries"],
        "keywords": ["EV", "solar", "wind", "battery", "hydrogen", "clean energy"],
    },
    "Fintech": {
        "topics": ["fintech", "cryptocurrency", "blockchain"],
        "keywords": ["fintech", "crypto", "DeFi", "digital payments", "neobank"],
    },
    "Biotech": {
        "topics": ["biotechnology", "gene_therapy", "pharmaceuticals"],
        "keywords": ["biotech", "gene therapy", "CRISPR", "mRNA", "personalized medicine"],
    },
    "Space": {
        "topics": ["space", "satellites"],
        "keywords": ["SpaceX", "satellite", "space tourism", "rocket", "orbital"],
    },
}

DISRUPTION_SIGNALS = [
    "disrupts", "transforms", "replaces", "automates", "revolutionizes",
    "obsolete", "game-changer", "breakthrough", "paradigm shift"
]

def analyze_disruption_theme(theme_name, config, days=30):
    """Analyze disruption signals for a theme."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    metrics = {
        "theme": theme_name,
        "total_coverage": 0,
        "disruption_signals": 0,
        "positive_coverage": 0,
        "tier1_coverage": 0,
        "trending_entities": [],
    }

    # Total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["keywords"]),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["total_coverage"] = resp.json().get("total_results", 0)

    # Disruption signal coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["keywords"] + DISRUPTION_SIGNALS),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["disruption_signals"] = resp.json().get("total_results", 0)

    # Positive coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["keywords"]),
        "sentiment.overall.polarity": "positive",
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["positive_coverage"] = resp.json().get("total_results", 0)

    # Tier-1 coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["keywords"]),
        "source.rank.opr.min": 0.7,
        "published_at.start": start,
        "per_page": 1,
    })
    metrics["tier1_coverage"] = resp.json().get("total_results", 0)

    # Extract trending entities
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["keywords"]),
        "source.rank.opr.min": 0.6,
        "published_at.start": start,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 20,
    })

    entity_counts = defaultdict(int)
    for article in resp.json().get("results", []):
        for entity in article.get("entities", []):
            if entity.get("type") == "organization":
                entity_counts[entity["name"]] += 1

    metrics["trending_entities"] = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Calculate disruption score
    total = metrics["total_coverage"] or 1
    metrics["disruption_intensity"] = metrics["disruption_signals"] / total
    metrics["sentiment_ratio"] = metrics["positive_coverage"] / total
    metrics["tier1_ratio"] = metrics["tier1_coverage"] / total

    metrics["disruption_score"] = (
        metrics["disruption_intensity"] * 40 +
        metrics["sentiment_ratio"] * 30 +
        metrics["tier1_ratio"] * 30
    ) * 100

    return metrics

print("INDUSTRY DISRUPTION RADAR")
print("=" * 70)
print(f"Scan Date: {datetime.utcnow().strftime('%Y-%m-%d')}")
print(f"Analysis Period: Last 30 days\n")

results = []
for theme_name, config in DISRUPTION_THEMES.items():
    result = analyze_disruption_theme(theme_name, config, days=30)
    results.append(result)

# Sort by disruption score
results.sort(key=lambda x: x["disruption_score"], reverse=True)

print(f"{'Theme':<20} {'Coverage':>10} {'Signals':>10} {'Score':>10} {'Trending'}")
print("-" * 70)

for r in results:
    top_entity = r["trending_entities"][0][0] if r["trending_entities"] else "N/A"
    print(f"{r['theme']:<20} {r['total_coverage']:>10,} {r['disruption_signals']:>10,} "
          f"{r['disruption_score']:>9.0f} {top_entity[:15]}")

# Detailed view
print("\n" + "=" * 70)
print("DETAILED DISRUPTION ANALYSIS")
print("=" * 70)

for r in results[:3]:
    print(f"\n{r['theme']}")
    print("-" * 50)
    print(f"  Coverage: {r['total_coverage']:,} articles")
    print(f"  Disruption Signals: {r['disruption_signals']:,} ({r['disruption_intensity']:.1%})")
    print(f"  Positive Sentiment: {r['sentiment_ratio']:.1%}")
    print(f"  Tier-1 Coverage: {r['tier1_ratio']:.1%}")
    print(f"  Disruption Score: {r['disruption_score']:.0f}/100")

    if r["trending_entities"]:
        print("  Top Mentioned Companies:")
        for entity, count in r["trending_entities"][:3]:
            print(f"    • {entity}: {count} mentions")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const DISRUPTION_THEMES = {
  "AI & Automation": ["AI", "automation", "machine learning", "ChatGPT"],
  "Clean Energy": ["EV", "solar", "renewable", "battery", "hydrogen"],
  "Fintech": ["fintech", "crypto", "DeFi", "digital payments"],
  "Biotech": ["biotech", "gene therapy", "CRISPR", "mRNA"],
};

const DISRUPTION_SIGNALS = ["disrupts", "transforms", "replaces", "revolutionizes"];

async function analyzeDisruptionTheme(themeName, keywords, days = 30) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  // Total coverage
  const totalParams = new URLSearchParams({
    api_key: API_KEY,
    title: keywords.join(","),
    "published_at.start": start,
    language: "en",
    per_page: "1",
  });

  let response = await fetch(`${BASE_URL}?${totalParams}`);
  let data = await response.json();
  const totalCoverage = data.total_results || 0;

  // Disruption signals
  const signalParams = new URLSearchParams({
    api_key: API_KEY,
    title: [...keywords, ...DISRUPTION_SIGNALS].join(","),
    "published_at.start": start,
    language: "en",
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${signalParams}`);
  data = await response.json();
  const disruptionSignals = data.total_results || 0;

  const disruptionIntensity = disruptionSignals / Math.max(totalCoverage, 1);
  const disruptionScore = disruptionIntensity * 100;

  return {
    theme: themeName,
    totalCoverage,
    disruptionSignals,
    disruptionIntensity,
    disruptionScore,
  };
}

async function runRadar() {
  console.log("INDUSTRY DISRUPTION RADAR");
  console.log("=".repeat(50));

  const results = [];

  for (const [theme, keywords] of Object.entries(DISRUPTION_THEMES)) {
    const result = await analyzeDisruptionTheme(theme, keywords, 30);
    results.push(result);
  }

  results.sort((a, b) => b.disruptionScore - a.disruptionScore);

  console.log(`\n${"Theme".padEnd(20)} ${"Coverage".padStart(10)} ${"Signals".padStart(10)} ${"Score".padStart(10)}`);
  console.log("-".repeat(55));

  for (const r of results) {
    console.log(
      `${r.theme.padEnd(20)} ${String(r.totalCoverage).padStart(10)} ` +
      `${String(r.disruptionSignals).padStart(10)} ${r.disruptionScore.toFixed(0).padStart(10)}`
    );
  }
}

runRadar();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$themes = [
    "AI & Automation" => ["AI", "automation", "machine learning", "ChatGPT"],
    "Clean Energy"    => ["EV", "solar", "renewable", "battery"],
    "Fintech"         => ["fintech", "crypto", "DeFi", "payments"],
    "Biotech"         => ["biotech", "gene therapy", "CRISPR"],
];

$disruptionSignals = ["disrupts", "transforms", "replaces", "revolutionizes"];

function analyzeTheme(string $theme, array $keywords, int $days = 30): array
{
    global $apiKey, $baseUrl, $disruptionSignals;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Total
    $query = http_build_query([
        "api_key"            => $apiKey,
        "title"              => implode(",", $keywords),
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $totalCoverage = $data["total_results"] ?? 0;

    // Disruption signals
    $query = http_build_query([
        "api_key"            => $apiKey,
        "title"              => implode(",", array_merge($keywords, $disruptionSignals)),
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $signals = $data["total_results"] ?? 0;

    $intensity = $signals / max($totalCoverage, 1);

    return [
        "theme"      => $theme,
        "coverage"   => $totalCoverage,
        "signals"    => $signals,
        "intensity"  => $intensity,
        "score"      => $intensity * 100,
    ];
}

echo "INDUSTRY DISRUPTION RADAR\n";
echo str_repeat("=", 55) . "\n\n";

$results = [];
foreach ($themes as $theme => $keywords) {
    $results[] = analyzeTheme($theme, $keywords, 30);
}

usort($results, fn($a, $b) => $b["score"] <=> $a["score"]);

printf("%-20s %10s %10s %10s\n", "Theme", "Coverage", "Signals", "Score");
echo str_repeat("-", 55) . "\n";

foreach ($results as $r) {
    printf("%-20s %10d %10d %10.0f\n",
        $r["theme"], $r["coverage"], $r["signals"], $r["score"]);
}
```

## Common Use Cases

- **Emerging technology tracking** — monitor coverage of new technologies.
- **Startup ecosystem analysis** — track funding, launches, and exits.
- **Incumbent disruption risk** — assess threats to established players.
- **Innovation trend detection** — identify emerging patterns early.
- **Venture capital intelligence** — surface investment opportunities.
- **Corporate strategy support** — inform M&A and partnership decisions.
- **Technology adoption curves** — track mainstream breakthrough signals.
- **Cross-industry disruption** — detect technology transfer patterns.

## See Also

- [examples.md](./examples.md) — detailed code examples for industry disruption radar.
