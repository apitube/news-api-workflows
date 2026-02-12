# Brand Health Scorecard

Workflow for building comprehensive brand health measurement systems that combine share of voice, sentiment trends, media reach, competitive benchmarking, and topic attribution using the [APITube News API](https://apitube.io).

## Overview

The **Brand Health Scorecard** workflow provides a holistic view of brand performance by analyzing multiple dimensions: media volume, sentiment distribution, source authority mix, topic coverage, competitive positioning, and trend trajectories. Build executive dashboards, track campaign effectiveness, benchmark against competitors, and detect early warning signals. Ideal for brand managers, CMOs, PR agencies, and marketing analytics teams.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
GET https://api.apitube.io/v1/news/entity
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by brand/organization name.                                   |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (0.0–1.0).                                  |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `source.domain`               | string  | Filter by specific media outlets.                                    |
| `topic.id`                    | string  | Filter by topic for message tracking.                               |
| `category.id`                 | string  | Filter by IPTC category.                                             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Get brand coverage volume
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Nike&language.code=en&per_page=100" | jq '.results | length'

# Get sentiment distribution
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Nike&sentiment.overall.polarity=positive&language.code=en&per_page=100" | jq '.results | length'

# Get tier-1 media coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Nike&source.rank.opr.min=6&language.code=en&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

BRAND = "Nike"
COMPETITORS = ["Adidas", "Puma", "Under Armour", "New Balance"]

TIER_1_SOURCES = "reuters.com,bloomberg.com,nytimes.com,wsj.com,ft.com"
TIER_2_SOURCES = "cnbc.com,bbc.com,cnn.com,forbes.com,businessinsider.com"

def calculate_brand_health(brand, days=30):
    """Calculate comprehensive brand health score."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    metrics = {
        "volume": {},
        "sentiment": {},
        "reach": {},
        "share_of_voice": 0,
    }

    # Volume metrics
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": brand,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 100,
    })
    metrics["volume"]["total"] = len(resp.json().get("results", []))
    metrics["volume"]["daily_avg"] = metrics["volume"]["total"] / days

    # Sentiment distribution
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": brand,
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })
        metrics["sentiment"][polarity] = len(resp.json().get("results", []))

    # Media reach (by tier)
    resp_tier1 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": brand,
        "source.domain": TIER_1_SOURCES,
        "published_at.start": start,
        "per_page": 100,
    })
    metrics["reach"]["tier1"] = len(resp_tier1.json().get("results", []))

    resp_tier2 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": brand,
        "source.domain": TIER_2_SOURCES,
        "published_at.start": start,
        "per_page": 100,
    })
    metrics["reach"]["tier2"] = len(resp_tier2.json().get("results", []))

    # Calculate scores
    total_sentiment = sum(metrics["sentiment"].values()) or 1

    sentiment_score = (
        (metrics["sentiment"]["positive"] - metrics["sentiment"]["negative"])
        / total_sentiment
    )

    reach_score = (
        metrics["reach"]["tier1"] * 2 + metrics["reach"]["tier2"]
    ) / max(metrics["volume"]["total"], 1)

    # Composite health score (0-100)
    health_score = 50 + (
        sentiment_score * 30 +
        min(20, reach_score * 100)
    )

    return {
        "brand": brand,
        "metrics": metrics,
        "sentiment_score": sentiment_score,
        "reach_score": reach_score,
        "health_score": max(0, min(100, health_score)),
        "health_grade": "A" if health_score >= 80 else \
                       "B" if health_score >= 65 else \
                       "C" if health_score >= 50 else \
                       "D" if health_score >= 35 else "F",
    }

def calculate_share_of_voice(brand, competitors, days=30):
    """Calculate share of voice against competitors."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    volumes = {}
    all_brands = [brand] + competitors

    for b in all_brands:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": b,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 100,
        })
        volumes[b] = len(resp.json().get("results", []))

    total_volume = sum(volumes.values()) or 1

    return {
        "volumes": volumes,
        "share_of_voice": {b: v / total_volume for b, v in volumes.items()},
        "rank": sorted(volumes.keys(), key=lambda b: volumes[b], reverse=True).index(brand) + 1,
    }

# Generate scorecard
print("BRAND HEALTH SCORECARD")
print("=" * 70)
print(f"Brand: {BRAND}")
print(f"Period: Last 30 days")
print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

# Calculate health
health = calculate_brand_health(BRAND, days=30)

print(f"HEALTH SCORE: {health['health_score']:.0f}/100 (Grade: {health['health_grade']})")
print("-" * 50)
print(f"  Total Coverage: {health['metrics']['volume']['total']} articles")
print(f"  Daily Average: {health['metrics']['volume']['daily_avg']:.1f} articles/day")
print(f"  Sentiment Score: {health['sentiment_score']:+.3f}")
print(f"  Tier-1 Reach: {health['metrics']['reach']['tier1']} articles")
print(f"  Tier-2 Reach: {health['metrics']['reach']['tier2']} articles")

# Share of voice
print("\nSHARE OF VOICE:")
print("-" * 50)

sov = calculate_share_of_voice(BRAND, COMPETITORS, days=30)

for brand_name, share in sorted(sov["share_of_voice"].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(share * 50)
    marker = " ← YOUR BRAND" if brand_name == BRAND else ""
    print(f"  {brand_name:<15}: {share:>6.1%} {bar}{marker}")

print(f"\n  Your Rank: #{sov['rank']} of {len(sov['volumes'])}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function calculateBrandHealth(brand, days = 30) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const metrics = {
    volume: 0,
    sentiment: { positive: 0, negative: 0, neutral: 0 },
    tier1Reach: 0,
  };

  // Total volume
  const volumeParams = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": brand,
    "published_at.start": start,
    "language.code": "en",
    per_page: "100",
  });

  let response = await fetch(`${BASE_URL}?${volumeParams}`);
  let data = await response.json();
  metrics.volume = (data.results || []).length;

  // Sentiment
  for (const polarity of ["positive", "negative", "neutral"]) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": brand,
      "sentiment.overall.polarity": polarity,
      "published_at.start": start,
      "language.code": "en",
      per_page: "100",
    });

    response = await fetch(`${BASE_URL}?${params}`);
    data = await response.json();
    metrics.sentiment[polarity] = (data.results || []).length;
  }

  // Tier-1 reach
  const tier1Params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": brand,
    "source.rank.opr.min": "6",
    "published_at.start": start,
    per_page: "100",
  });

  response = await fetch(`${BASE_URL}?${tier1Params}`);
  data = await response.json();
  metrics.tier1Reach = (data.results || []).length;

  // Calculate scores
  const sentimentTotal = Object.values(metrics.sentiment).reduce((a, b) => a + b, 0) || 1;
  const sentimentScore = (metrics.sentiment.positive - metrics.sentiment.negative) / sentimentTotal;
  const healthScore = Math.max(0, Math.min(100, 50 + sentimentScore * 30 + Math.min(20, (metrics.tier1Reach / metrics.volume) * 100)));

  return {
    brand,
    metrics,
    sentimentScore,
    healthScore,
    grade: healthScore >= 80 ? "A" : healthScore >= 65 ? "B" : healthScore >= 50 ? "C" : healthScore >= 35 ? "D" : "F",
  };
}

async function generateScorecard() {
  const brand = "Nike";
  const competitors = ["Adidas", "Puma", "Under Armour"];

  console.log("BRAND HEALTH SCORECARD");
  console.log("=".repeat(50));

  const health = await calculateBrandHealth(brand, 30);

  console.log(`\n${brand}: ${health.healthScore.toFixed(0)}/100 (Grade: ${health.grade})`);
  console.log(`  Volume: ${health.metrics.volume} articles`);
  console.log(`  Sentiment: ${health.sentimentScore >= 0 ? "+" : ""}${health.sentimentScore.toFixed(3)}`);
  console.log(`  Tier-1 Reach: ${health.metrics.tier1Reach}`);

  // Share of voice
  console.log("\nShare of Voice:");

  const allBrands = [brand, ...competitors];
  const volumes = {};
  let totalVolume = 0;

  for (const b of allBrands) {
    const result = await calculateBrandHealth(b, 30);
    volumes[b] = result.metrics.volume;
    totalVolume += result.metrics.volume;
  }

  for (const [b, vol] of Object.entries(volumes).sort((a, b) => b[1] - a[1])) {
    const share = vol / totalVolume;
    const bar = "#".repeat(Math.round(share * 30));
    const marker = b === brand ? " ← YOUR BRAND" : "";
    console.log(`  ${b.padEnd(15)}: ${(share * 100).toFixed(1).padStart(5)}% ${bar}${marker}`);
  }
}

generateScorecard();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function calculateBrandHealth(string $brand, int $days = 30): array
{
    global $apiKey, $baseUrl;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Volume
    $query = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $brand,
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 100,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $volume = count($data["results"] ?? []);

    // Sentiment
    $sentiment = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "organization.name"          => $brand,
            "sentiment.overall.polarity" => $polarity,
            "published_at.start"         => $start,
            "language.code"              => "en",
            "per_page"                   => 100,
        ]);
        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $sentiment[$polarity] = count($data["results"] ?? []);
    }

    // Tier-1 reach
    $query = http_build_query([
        "api_key"             => $apiKey,
        "organization.name"   => $brand,
        "source.rank.opr.min" => 6,
        "published_at.start"  => $start,
        "per_page"            => 100,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $tier1Reach = count($data["results"] ?? []);

    // Calculate scores
    $sentimentTotal = array_sum($sentiment) ?: 1;
    $sentimentScore = ($sentiment["positive"] - $sentiment["negative"]) / $sentimentTotal;
    $healthScore = max(0, min(100, 50 + $sentimentScore * 30 + min(20, ($tier1Reach / max($volume, 1)) * 100)));

    return [
        "brand"           => $brand,
        "volume"          => $volume,
        "sentiment"       => $sentiment,
        "sentiment_score" => $sentimentScore,
        "tier1_reach"     => $tier1Reach,
        "health_score"    => $healthScore,
        "grade"           => $healthScore >= 80 ? "A" : ($healthScore >= 65 ? "B" : ($healthScore >= 50 ? "C" : ($healthScore >= 35 ? "D" : "F"))),
    ];
}

$brand = "Nike";
$competitors = ["Adidas", "Puma", "Under Armour"];

echo "BRAND HEALTH SCORECARD\n";
echo str_repeat("=", 50) . "\n\n";

$health = calculateBrandHealth($brand, 30);

printf("%s: %.0f/100 (Grade: %s)\n", $brand, $health["health_score"], $health["grade"]);
echo "  Volume: {$health['volume']} articles\n";
printf("  Sentiment: %+.3f\n", $health["sentiment_score"]);
echo "  Tier-1 Reach: {$health['tier1_reach']}\n";

// Share of voice
echo "\nShare of Voice:\n";

$allBrands = array_merge([$brand], $competitors);
$volumes = [];

foreach ($allBrands as $b) {
    $result = calculateBrandHealth($b, 30);
    $volumes[$b] = $result["volume"];
}

arsort($volumes);
$totalVolume = array_sum($volumes) ?: 1;

foreach ($volumes as $b => $vol) {
    $share = $vol / $totalVolume * 100;
    $bar = str_repeat("#", (int) round($share / 3));
    $marker = $b === $brand ? " <- YOUR BRAND" : "";
    printf("  %-15s: %5.1f%% %s%s\n", $b, $share, $bar, $marker);
}
```

## Common Use Cases

- **Brand health dashboards** — comprehensive view of brand performance metrics.
- **Competitive benchmarking** — compare brand metrics against competitors.
- **Share of voice tracking** — measure media presence relative to market.
- **Campaign effectiveness** — track brand metrics before/after campaigns.
- **Crisis impact assessment** — measure brand health during/after crisis.
- **Trend analysis** — track brand metrics over time for patterns.
- **Media mix analysis** — understand which media types cover your brand.
- **Message penetration** — track how key messages spread in coverage.

## See Also

- [examples.md](./examples.md) — detailed code examples for brand health scorecard workflows.
