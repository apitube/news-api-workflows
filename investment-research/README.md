# Investment Research

Workflow for building data-driven investment research systems that analyze market sentiment, track earnings coverage, monitor insider activity, and identify market-moving events using the [APITube News API](https://apitube.io).

## Overview

The **Investment Research** workflow combines entity tracking, sentiment analysis, topic filtering, and time-series analysis to support investment decision-making. Monitor pre-earnings sentiment shifts, track analyst coverage, detect M&A rumors, analyze sector rotations, and build quantitative signals from news data. Ideal for hedge funds, asset managers, research analysts, and algorithmic trading systems.

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
| `organization.name`           | string  | Filter by company/stock name (e.g., `Apple`, `NVIDIA`).             |
| `person.name`                 | string  | Filter by person name.                                               |
| `topic.id`                    | string  | Filter by topic (e.g., `earnings`, `ipo`, `mergers_acquisitions`). |
| `industry.id`                 | string  | Filter by industry ID.                                               |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (0.0–1.0).                                  |
| `sentiment.overall.score.max` | number  | Maximum sentiment score (0.0–1.0).                                  |
| `title`                       | string  | Filter by keywords (e.g., `earnings,revenue,guidance,forecast`).    |
| `source.domain`               | string  | Filter by financial news sources.                                    |
| `source.rank.opr.min`         | number  | Minimum source authority (0–7).                                     |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language.code`               | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |
| `page`                        | integer | Page number for pagination.                                          |

## Quick Start

### cURL

```bash
# Get earnings-related news for a stock
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Apple&title=earnings,revenue,guidance,forecast&language.code=en&per_page=20"

# Monitor M&A rumors
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=mergers_acquisitions&language.code=en&source.rank.opr.min=4&per_page=20"

# Track sentiment for a sector (AI/semiconductors)
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=artificial_intelligence&industry.id=semiconductors&sentiment.overall.polarity=positive&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

FINANCIAL_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com,marketwatch.com"

# Pre-earnings sentiment analysis
stock = "NVIDIA"
earnings_date = "2026-02-26"

# Get sentiment 7 days before earnings
start = (datetime.fromisoformat(earnings_date) - timedelta(days=7)).strftime("%Y-%m-%d")
end = earnings_date

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "organization.name": stock,
    "source.domain": FINANCIAL_SOURCES,
    "published_at.start": start,
    "published_at.end": end,
    "language.code": "en",
    "per_page": 50,
})

data = response.json()
articles = data.get("results", [])

# Calculate sentiment distribution
sentiments = {"positive": 0, "negative": 0, "neutral": 0}
for article in articles:
    polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
    sentiments[polarity] += 1

total = sum(sentiments.values()) or 1
print(f"Pre-Earnings Sentiment for {stock}")
print(f"Period: {start} to {end}")
print(f"Total articles: {total}")
print(f"Positive: {sentiments['positive']} ({sentiments['positive']/total*100:.1f}%)")
print(f"Negative: {sentiments['negative']} ({sentiments['negative']/total*100:.1f}%)")
print(f"Neutral: {sentiments['neutral']} ({sentiments['neutral']/total*100:.1f}%)")

# Sentiment score
net_sentiment = (sentiments["positive"] - sentiments["negative"]) / total
print(f"Net Sentiment Score: {net_sentiment:+.3f}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const FINANCIAL_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

async function analyzePreEarningsSentiment(stock, earningsDate, daysBefore = 7) {
  const end = new Date(earningsDate);
  const start = new Date(end.getTime() - daysBefore * 24 * 60 * 60 * 1000);

  const params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": stock,
    "source.domain": FINANCIAL_SOURCES,
    "published_at.start": start.toISOString().split("T")[0],
    "published_at.end": end.toISOString().split("T")[0],
    "language.code": "en",
    per_page: "50",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  const sentiments = { positive: 0, negative: 0, neutral: 0 };
  for (const article of data.results || []) {
    const polarity = article.sentiment?.overall?.polarity || "neutral";
    sentiments[polarity]++;
  }

  const total = Object.values(sentiments).reduce((a, b) => a + b, 0) || 1;
  const netSentiment = (sentiments.positive - sentiments.negative) / total;

  console.log(`Pre-Earnings Sentiment: ${stock}`);
  console.log(`Total: ${total} articles`);
  console.log(`Positive: ${sentiments.positive} (${(sentiments.positive / total * 100).toFixed(1)}%)`);
  console.log(`Negative: ${sentiments.negative} (${(sentiments.negative / total * 100).toFixed(1)}%)`);
  console.log(`Net Score: ${netSentiment >= 0 ? "+" : ""}${netSentiment.toFixed(3)}`);

  return { sentiments, netSentiment, total };
}

analyzePreEarningsSentiment("NVIDIA", "2026-02-26");
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$financialSources = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

$stock        = "NVIDIA";
$earningsDate = "2026-02-26";
$daysBefore   = 7;

$end   = new DateTime($earningsDate);
$start = (clone $end)->modify("-{$daysBefore} days");

$query = http_build_query([
    "api_key"            => $apiKey,
    "organization.name"  => $stock,
    "source.domain"      => $financialSources,
    "published_at.start" => $start->format("Y-m-d"),
    "published_at.end"   => $end->format("Y-m-d"),
    "language.code"      => "en",
    "per_page"           => 50,
]);

$data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$articles = $data["results"] ?? [];

$sentiments = ["positive" => 0, "negative" => 0, "neutral" => 0];
foreach ($articles as $article) {
    $polarity = $article["sentiment"]["overall"]["polarity"] ?? "neutral";
    $sentiments[$polarity]++;
}

$total        = array_sum($sentiments) ?: 1;
$netSentiment = ($sentiments["positive"] - $sentiments["negative"]) / $total;

echo "Pre-Earnings Sentiment: {$stock}\n";
echo "Period: {$start->format('Y-m-d')} to {$end->format('Y-m-d')}\n";
echo "Total: {$total} articles\n";
printf("Positive: %d (%.1f%%)\n", $sentiments["positive"], $sentiments["positive"] / $total * 100);
printf("Negative: %d (%.1f%%)\n", $sentiments["negative"], $sentiments["negative"] / $total * 100);
printf("Net Score: %+.3f\n", $netSentiment);
```

## Common Use Cases

- **Pre-earnings sentiment analysis** — track sentiment shifts in the 7-14 days before earnings announcements.
- **M&A rumor detection** — monitor merger and acquisition news with source authority filtering.
- **Analyst coverage tracking** — identify analyst upgrades, downgrades, and price target changes.
- **Sector rotation signals** — compare sentiment across sectors to identify rotation patterns.
- **Event-driven alpha signals** — detect market-moving events before they're fully priced in.
- **ESG controversy monitoring** — track environmental, social, and governance issues.
- **Insider activity alerts** — monitor news about executive trades and insider transactions.
- **IPO sentiment tracking** — analyze coverage sentiment for upcoming IPOs.

## See Also

- [examples.md](./examples.md) — detailed code examples for investment research workflows.
