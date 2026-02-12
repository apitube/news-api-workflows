# Executive Intelligence

Workflow for monitoring C-suite executives, tracking leadership changes, analyzing personal reputation, and detecting executive-related market events using the [APITube News API](https://apitube.io).

## Overview

The **Executive Intelligence** workflow provides comprehensive monitoring of corporate executives by combining person entity tracking, sentiment analysis, topic classification, and cross-reference with company coverage. Track CEO statements, monitor executive departures, detect personal controversies, analyze media presence, and correlate executive news with company performance. Ideal for investor relations, executive search firms, board governance, activist investors, and PR agencies.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/entity
GET https://api.apitube.io/v1/suggest/entities
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by executive name.                                            |
| `entity.type`                 | string  | Use `person` for executive tracking.                                |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (0.0–1.0).                                  |
| `title`                       | string  | Filter by executive-related keywords.                                |
| `topic.id`                    | string  | Filter by topic (leadership, compensation, etc.).                   |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `source.domain`               | string  | Filter by business news sources.                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Track CEO media coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Tim Cook&entity.type=person&language=en&per_page=20"

# Monitor executive changes and departures
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=CEO,CFO,CTO resigns,steps down,departure,appointed,named&source.rank.opr.min=0.6&per_page=20"

# Track executive sentiment
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Elon Musk&entity.type=person&sentiment.overall.polarity=negative&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

EXECUTIVE_KEYWORDS = {
    "leadership_change": ["CEO", "CFO", "CTO", "COO", "appointed", "resigned", "steps down", "departure", "successor"],
    "compensation": ["salary", "bonus", "compensation", "stock options", "pay package", "golden parachute"],
    "controversy": ["scandal", "investigation", "lawsuit", "allegation", "misconduct", "harassment"],
    "public_statements": ["said", "announced", "stated", "revealed", "warned", "predicted"],
}

def analyze_executive(name, days=30):
    """Comprehensive executive media analysis."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "entity.type": "person",
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    total_coverage = resp.json().get("total_results", 0)

    # Sentiment breakdown
    sentiments = {}
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": name,
            "entity.type": "person",
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        sentiments[polarity] = resp.json().get("total_results", 0)

    # Topic breakdown
    topics = {}
    for topic, keywords in EXECUTIVE_KEYWORDS.items():
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": name,
            "title": ",".join(keywords),
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        topics[topic] = resp.json().get("total_results", 0)

    # Get recent high-authority coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "entity.type": "person",
        "source.rank.opr.min": 0.6,
        "published_at.start": start,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 10,
    })
    recent_articles = [
        {
            "title": a["title"],
            "source": a["source"]["domain"],
            "sentiment": a.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
            "date": a["published_at"][:10],
        }
        for a in resp.json().get("results", [])
    ]

    # Calculate reputation score
    total_sentiment = sum(sentiments.values()) or 1
    reputation_score = (sentiments["positive"] - sentiments["negative"]) / total_sentiment

    return {
        "name": name,
        "total_coverage": total_coverage,
        "sentiments": sentiments,
        "topics": topics,
        "reputation_score": reputation_score,
        "recent_articles": recent_articles,
        "media_presence": "HIGH" if total_coverage > 100 else "MEDIUM" if total_coverage > 20 else "LOW",
    }

# Analyze executives
executives = ["Tim Cook", "Satya Nadella", "Elon Musk", "Sundar Pichai", "Andy Jassy"]

print("EXECUTIVE INTELLIGENCE REPORT")
print("=" * 70)
print(f"Analysis Period: Last 30 days\n")

for exec_name in executives:
    analysis = analyze_executive(exec_name, days=30)

    print(f"\n{analysis['name']}")
    print("-" * 50)
    print(f"  Media Presence: {analysis['media_presence']} ({analysis['total_coverage']} articles)")
    print(f"  Reputation Score: {analysis['reputation_score']:+.3f}")
    print(f"  Sentiment: +{analysis['sentiments']['positive']} / -{analysis['sentiments']['negative']} / ={analysis['sentiments']['neutral']}")

    if analysis["topics"]["controversy"] > 0:
        print(f"  ⚠️  Controversy mentions: {analysis['topics']['controversy']}")

    print("  Recent Headlines:")
    for article in analysis["recent_articles"][:3]:
        icon = "📈" if article["sentiment"] == "positive" else "📉" if article["sentiment"] == "negative" else "📰"
        print(f"    {icon} [{article['source']}] {article['title'][:45]}...")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const CONTROVERSY_KEYWORDS = [
  "scandal", "investigation", "lawsuit", "allegation",
  "misconduct", "harassment", "fraud", "controversy"
];

async function analyzeExecutive(name, days = 30) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  // Get total coverage
  const totalParams = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": name,
    "entity.type": "person",
    "published_at.start": start,
    language: "en",
    per_page: "1",
  });

  const totalResp = await fetch(`${BASE_URL}?${totalParams}`);
  const totalData = await totalResp.json();
  const totalCoverage = totalData.total_results || 0;

  // Get sentiment breakdown
  const sentiments = {};
  for (const polarity of ["positive", "negative", "neutral"]) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": name,
      "entity.type": "person",
      "sentiment.overall.polarity": polarity,
      "published_at.start": start,
      language: "en",
      per_page: "1",
    });

    const resp = await fetch(`${BASE_URL}?${params}`);
    const data = await resp.json();
    sentiments[polarity] = data.total_results || 0;
  }

  // Check for controversies
  const controversyParams = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": name,
    title: CONTROVERSY_KEYWORDS.join(","),
    "published_at.start": start,
    per_page: "1",
  });

  const controversyResp = await fetch(`${BASE_URL}?${controversyParams}`);
  const controversyData = await controversyResp.json();
  const controversies = controversyData.total_results || 0;

  // Calculate reputation
  const total = sentiments.positive + sentiments.negative + sentiments.neutral || 1;
  const reputationScore = (sentiments.positive - sentiments.negative) / total;

  return {
    name,
    totalCoverage,
    sentiments,
    controversies,
    reputationScore,
    mediaPresence: totalCoverage > 100 ? "HIGH" : totalCoverage > 20 ? "MEDIUM" : "LOW",
  };
}

async function generateReport() {
  const executives = ["Tim Cook", "Satya Nadella", "Elon Musk"];

  console.log("EXECUTIVE INTELLIGENCE REPORT");
  console.log("=".repeat(50));

  for (const name of executives) {
    const analysis = await analyzeExecutive(name, 30);

    console.log(`\n${analysis.name}`);
    console.log("-".repeat(40));
    console.log(`  Coverage: ${analysis.totalCoverage} (${analysis.mediaPresence})`);
    console.log(`  Reputation: ${analysis.reputationScore >= 0 ? "+" : ""}${analysis.reputationScore.toFixed(3)}`);
    console.log(`  Sentiment: +${analysis.sentiments.positive} / -${analysis.sentiments.negative}`);

    if (analysis.controversies > 0) {
      console.log(`  ⚠️  Controversies: ${analysis.controversies}`);
    }
  }
}

generateReport();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$controversyKeywords = ["scandal", "investigation", "lawsuit", "allegation", "misconduct"];

function analyzeExecutive(string $name, int $days = 30): array
{
    global $apiKey, $baseUrl, $controversyKeywords;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Total coverage
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $name,
        "entity.type"        => "person",
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $totalCoverage = $data["total_results"] ?? 0;

    // Sentiment breakdown
    $sentiments = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "entity.name"                => $name,
            "entity.type"                => "person",
            "sentiment.overall.polarity" => $polarity,
            "published_at.start"         => $start,
            "language"                   => "en",
            "per_page"                   => 1,
        ]);
        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $sentiments[$polarity] = $data["total_results"] ?? 0;
    }

    // Controversies
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $name,
        "title"              => implode(",", $controversyKeywords),
        "published_at.start" => $start,
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $controversies = $data["total_results"] ?? 0;

    $total = array_sum($sentiments) ?: 1;
    $reputationScore = ($sentiments["positive"] - $sentiments["negative"]) / $total;

    return [
        "name"             => $name,
        "total_coverage"   => $totalCoverage,
        "sentiments"       => $sentiments,
        "controversies"    => $controversies,
        "reputation_score" => $reputationScore,
        "media_presence"   => $totalCoverage > 100 ? "HIGH" : ($totalCoverage > 20 ? "MEDIUM" : "LOW"),
    ];
}

$executives = ["Tim Cook", "Satya Nadella", "Elon Musk"];

echo "EXECUTIVE INTELLIGENCE REPORT\n";
echo str_repeat("=", 50) . "\n";

foreach ($executives as $name) {
    $analysis = analyzeExecutive($name, 30);

    echo "\n{$analysis['name']}\n";
    echo str_repeat("-", 40) . "\n";
    echo "  Coverage: {$analysis['total_coverage']} ({$analysis['media_presence']})\n";
    printf("  Reputation: %+.3f\n", $analysis["reputation_score"]);
    echo "  Sentiment: +{$analysis['sentiments']['positive']} / -{$analysis['sentiments']['negative']}\n";

    if ($analysis["controversies"] > 0) {
        echo "  ⚠️  Controversies: {$analysis['controversies']}\n";
    }
}
```

## Common Use Cases

- **CEO reputation tracking** — monitor executive personal brand and media presence.
- **Leadership change detection** — track C-suite appointments, departures, and successions.
- **Executive controversy alerts** — detect scandals, investigations, and PR crises.
- **Board governance monitoring** — track board member news and director changes.
- **Activist investor intelligence** — monitor activist campaigns targeting executives.
- **Executive compensation analysis** — track pay-related news and shareholder reactions.
- **Spokesperson effectiveness** — analyze media coverage of executive public statements.
- **Executive due diligence** — pre-hire background research via news analysis.

## See Also

- [examples.md](./examples.md) — detailed code examples for executive intelligence workflows.
