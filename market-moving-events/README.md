# Market-Moving Events Detection

Workflow for detecting market-moving events in real-time by combining breaking news analysis, sentiment velocity tracking, mention volume anomalies, and cross-asset correlation signals using the [APITube News API](https://apitube.io).

## Overview

The **Market-Moving Events Detection** workflow uses advanced multi-signal analysis to identify news events likely to impact financial markets before they're fully priced in. Combines breaking news detection, abnormal volume spikes, sentiment velocity changes, cross-entity correlation, and source authority weighting to generate actionable trading signals. Ideal for quantitative trading desks, hedge funds, algorithmic trading systems, and market surveillance teams.

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
| `entity.name`                 | string  | Filter by company, index, commodity, or currency.                   |
| `entity.type`                 | string  | Filter by type: `organization`, `product`, `location`.              |
| `is_breaking`                 | boolean | Filter for breaking news articles.                                   |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (0.0–1.0).                                  |
| `sentiment.overall.score.max` | number  | Maximum sentiment score (0.0–1.0).                                  |
| `title`                       | string  | Filter by market-moving keywords.                                    |
| `topic.id`                    | string  | Filter by topic (earnings, mergers, macro, etc.).                   |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `source.domain`               | string  | Filter by financial news sources.                                    |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Detect breaking market news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&is_breaking=true&title=earnings,merger,acquisition,bankruptcy,FDA,SEC,lawsuit&source.rank.opr.min=0.7&per_page=20"

# Monitor high-impact financial keywords
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=beat expectations,missed estimates,guidance raised,guidance cut,stock buyback,dividend&language=en&per_page=30"

# Track sudden negative sentiment spikes
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sentiment.overall.polarity=negative&sentiment.overall.score.max=0.2&is_breaking=true&source.rank.opr.min=0.6&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

MARKET_MOVING_KEYWORDS = {
    "earnings": ["earnings", "beat", "miss", "EPS", "revenue", "guidance"],
    "corporate_actions": ["merger", "acquisition", "buyout", "spinoff", "IPO", "bankruptcy"],
    "regulatory": ["FDA approval", "SEC", "antitrust", "investigation", "fine", "settlement"],
    "macro": ["Fed", "interest rate", "inflation", "GDP", "unemployment", "recession"],
    "geopolitical": ["sanctions", "tariff", "trade war", "conflict", "OPEC"],
}

TIER_1_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com"

def detect_market_moving_events(minutes=30):
    """Detect potential market-moving events in real-time."""

    start = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"
    events = []

    # Check breaking news from tier-1 sources
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "is_breaking": "true",
        "source.domain": TIER_1_SOURCES,
        "published_at.start": start,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 20,
    })

    for article in resp.json().get("results", []):
        # Classify event type
        title_lower = article["title"].lower()
        event_type = "general"

        for category, keywords in MARKET_MOVING_KEYWORDS.items():
            if any(kw.lower() in title_lower for kw in keywords):
                event_type = category
                break

        # Extract affected entities
        entities = article.get("entities", [])
        tickers = [e["name"] for e in entities if e.get("type") == "organization"]

        # Calculate impact score
        sentiment = article.get("sentiment", {}).get("overall", {})
        sentiment_score = sentiment.get("score", 0.5)
        polarity = sentiment.get("polarity", "neutral")
        source_rank = article["source"].get("rank", {}).get("opr", 0.5)

        # Higher score = more likely to move markets
        impact_score = (
            (1 - sentiment_score if polarity == "negative" else sentiment_score) * 0.3 +
            source_rank * 0.4 +
            (0.3 if article.get("is_breaking") else 0)
        ) * 100

        events.append({
            "title": article["title"],
            "source": article["source"]["domain"],
            "published_at": article["published_at"],
            "event_type": event_type,
            "affected_tickers": tickers[:5],
            "sentiment": polarity,
            "sentiment_score": sentiment_score,
            "source_authority": source_rank,
            "impact_score": impact_score,
            "url": article["href"],
        })

    # Sort by impact score
    events.sort(key=lambda x: x["impact_score"], reverse=True)

    return events

print("MARKET-MOVING EVENTS DETECTOR")
print("=" * 70)
print(f"Scan Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Lookback: 30 minutes\n")

events = detect_market_moving_events(minutes=30)

if not events:
    print("No significant market-moving events detected.")
else:
    print(f"Detected {len(events)} potential market-moving events:\n")

    for i, event in enumerate(events[:10], 1):
        impact_emoji = "🔴" if event["impact_score"] >= 70 else \
                      "🟠" if event["impact_score"] >= 50 else \
                      "🟡" if event["impact_score"] >= 30 else "🟢"

        print(f"{i}. {impact_emoji} [{event['event_type'].upper()}] Impact: {event['impact_score']:.0f}/100")
        print(f"   {event['title'][:70]}...")
        print(f"   Source: {event['source']} | Sentiment: {event['sentiment']} ({event['sentiment_score']:.2f})")
        if event["affected_tickers"]:
            print(f"   Tickers: {', '.join(event['affected_tickers'])}")
        print()
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const MARKET_KEYWORDS = {
  earnings: ["earnings", "beat", "miss", "EPS", "revenue", "guidance"],
  corporate: ["merger", "acquisition", "buyout", "IPO", "bankruptcy"],
  regulatory: ["FDA", "SEC", "antitrust", "investigation", "settlement"],
};

const TIER_1_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

async function detectMarketEvents(minutes = 30) {
  const start = new Date(Date.now() - minutes * 60 * 1000).toISOString();

  const params = new URLSearchParams({
    api_key: API_KEY,
    is_breaking: "true",
    "source.domain": TIER_1_SOURCES,
    "published_at.start": start,
    language: "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "20",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  const events = (data.results || []).map(article => {
    const titleLower = article.title.toLowerCase();
    let eventType = "general";

    for (const [category, keywords] of Object.entries(MARKET_KEYWORDS)) {
      if (keywords.some(kw => titleLower.includes(kw.toLowerCase()))) {
        eventType = category;
        break;
      }
    }

    const sentiment = article.sentiment?.overall || {};
    const sourceRank = article.source?.rank?.opr || 0.5;

    const impactScore = (
      (sentiment.polarity === "negative" ? 1 - (sentiment.score || 0.5) : sentiment.score || 0.5) * 0.3 +
      sourceRank * 0.4 +
      (article.is_breaking ? 0.3 : 0)
    ) * 100;

    return {
      title: article.title,
      source: article.source.domain,
      publishedAt: article.published_at,
      eventType,
      tickers: (article.entities || [])
        .filter(e => e.type === "organization")
        .map(e => e.name)
        .slice(0, 5),
      sentiment: sentiment.polarity || "neutral",
      impactScore,
    };
  });

  return events.sort((a, b) => b.impactScore - a.impactScore);
}

async function runDetector() {
  console.log("MARKET-MOVING EVENTS DETECTOR");
  console.log("=".repeat(60));

  const events = await detectMarketEvents(30);

  if (events.length === 0) {
    console.log("No significant events detected.");
    return;
  }

  console.log(`\nDetected ${events.length} potential events:\n`);

  for (const [i, event] of events.slice(0, 10).entries()) {
    const emoji = event.impactScore >= 70 ? "🔴" :
                  event.impactScore >= 50 ? "🟠" :
                  event.impactScore >= 30 ? "🟡" : "🟢";

    console.log(`${i + 1}. ${emoji} [${event.eventType.toUpperCase()}] Impact: ${event.impactScore.toFixed(0)}/100`);
    console.log(`   ${event.title.slice(0, 65)}...`);
    console.log(`   Source: ${event.source} | Sentiment: ${event.sentiment}`);
    if (event.tickers.length > 0) {
      console.log(`   Tickers: ${event.tickers.join(", ")}`);
    }
    console.log();
  }
}

runDetector();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$marketKeywords = [
    "earnings"   => ["earnings", "beat", "miss", "EPS", "revenue"],
    "corporate"  => ["merger", "acquisition", "buyout", "IPO", "bankruptcy"],
    "regulatory" => ["FDA", "SEC", "antitrust", "investigation"],
];

$tier1Sources = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

function detectMarketEvents(int $minutes = 30): array
{
    global $apiKey, $baseUrl, $marketKeywords, $tier1Sources;

    $start = (new DateTime("-{$minutes} minutes"))->format("c");

    $query = http_build_query([
        "api_key"            => $apiKey,
        "is_breaking"        => "true",
        "source.domain"      => $tier1Sources,
        "published_at.start" => $start,
        "language"           => "en",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "per_page"           => 20,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $events = [];

    foreach ($data["results"] ?? [] as $article) {
        $titleLower = strtolower($article["title"]);
        $eventType = "general";

        foreach ($marketKeywords as $category => $keywords) {
            foreach ($keywords as $kw) {
                if (stripos($titleLower, strtolower($kw)) !== false) {
                    $eventType = $category;
                    break 2;
                }
            }
        }

        $sentiment = $article["sentiment"]["overall"] ?? [];
        $sourceRank = $article["source"]["rank"]["opr"] ?? 0.5;
        $sentimentScore = $sentiment["score"] ?? 0.5;
        $polarity = $sentiment["polarity"] ?? "neutral";

        $impactScore = (
            ($polarity === "negative" ? 1 - $sentimentScore : $sentimentScore) * 0.3 +
            $sourceRank * 0.4 +
            (!empty($article["is_breaking"]) ? 0.3 : 0)
        ) * 100;

        $events[] = [
            "title"        => $article["title"],
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "event_type"   => $eventType,
            "sentiment"    => $polarity,
            "impact_score" => $impactScore,
        ];
    }

    usort($events, fn($a, $b) => $b["impact_score"] <=> $a["impact_score"]);

    return $events;
}

echo "MARKET-MOVING EVENTS DETECTOR\n";
echo str_repeat("=", 60) . "\n\n";

$events = detectMarketEvents(30);

foreach (array_slice($events, 0, 10) as $i => $event) {
    $emoji = match (true) {
        $event["impact_score"] >= 70 => "🔴",
        $event["impact_score"] >= 50 => "🟠",
        $event["impact_score"] >= 30 => "🟡",
        default => "🟢",
    };

    $num = $i + 1;
    echo "{$num}. {$emoji} [{$event['event_type']}] Impact: " . round($event["impact_score"]) . "/100\n";
    echo "   " . substr($event["title"], 0, 65) . "...\n";
    echo "   Source: {$event['source']} | Sentiment: {$event['sentiment']}\n\n";
}
```

## Common Use Cases

- **Real-time trading signals** — detect market-moving news before price impact.
- **Earnings surprise detection** — identify beat/miss announcements instantly.
- **M&A rumor tracking** — catch acquisition news and deal announcements.
- **Regulatory event alerts** — FDA approvals, SEC actions, antitrust decisions.
- **Macro event monitoring** — Fed decisions, economic data releases.
- **Cross-asset correlation** — track how news affects related securities.
- **Volatility prediction** — anticipate market volatility from news flow.
- **Risk management alerts** — early warning for portfolio exposures.

## See Also

- [examples.md](./examples.md) — detailed code examples for market-moving event detection.
