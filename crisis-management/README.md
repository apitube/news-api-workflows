# Crisis Management

Workflow for comprehensive crisis detection, reputation monitoring, and rapid response coordination using the [APITube News API](https://apitube.io).

## Overview

The **Crisis Management** workflow combines real-time breaking news detection, multi-entity sentiment tracking, source authority analysis, and cross-language monitoring to build an enterprise-grade crisis management system. Detect PR crises early, track escalation patterns, coordinate rapid response, and measure recovery metrics. Ideal for corporate communications, brand protection, and executive reputation management.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by entity name (company, person, product).                   |
| `entity.type`                 | string  | Filter by entity type: `organization`, `person`, `brand`, `product`.|
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | number  | Minimum sentiment score (0.0–1.0).                                  |
| `sentiment.overall.score.max` | number  | Maximum sentiment score (0.0–1.0).                                  |
| `is_breaking`                 | boolean | Filter for breaking news articles.                                   |
| `title`                       | string  | Filter by keywords in title (supports boolean operators).           |
| `title_pattern`               | string  | Filter by regex pattern in title.                                    |
| `source.rank.opr.min`         | number  | Minimum source authority rank (0.0–1.0).                            |
| `source.domain`               | string  | Filter by source domains (comma-separated).                         |
| `source.country.code`         | string  | Filter by source country (ISO 3166-1).                              |
| `language`                    | string  | Filter by language code (comma-separated for multi-lang).           |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `category.id`                 | string  | Filter by IPTC category (e.g., `medtop:20000763` for scandals).    |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |
| `page`                        | integer | Page number for pagination.                                          |

## Quick Start

### cURL

```bash
# Detect negative breaking news about your brand
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Tesla&entity.type=organization&is_breaking=true&sentiment.overall.polarity=negative&per_page=10" | jq '.results[] | {title, source: .source.domain, sentiment: .sentiment.overall}'

# Monitor crisis keywords across multiple languages
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=lawsuit,scandal,fraud,investigation&entity.name=Tesla&language=en,de,fr,es&per_page=20"

# Get high-authority negative coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Tesla&sentiment.overall.polarity=negative&source.rank.opr.min=0.7&sort.by=published_at&sort.order=desc&per_page=10"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

BRAND = "Tesla"
CRISIS_KEYWORDS = ["lawsuit", "scandal", "fraud", "recall", "investigation", "bankruptcy", "layoff"]

# Multi-signal crisis detection
crisis_signals = {
    "breaking_negative": 0,
    "high_authority_negative": 0,
    "crisis_keywords": 0,
    "negative_velocity": 0,
}

# Check breaking negative news
response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "entity.name": BRAND,
    "entity.type": "organization",
    "is_breaking": "true",
    "sentiment.overall.polarity": "negative",
    "per_page": 1,
})
crisis_signals["breaking_negative"] = response.json().get("total_results", 0)

# Check high-authority negative coverage
response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "entity.name": BRAND,
    "entity.type": "organization",
    "sentiment.overall.polarity": "negative",
    "source.rank.opr.min": 0.7,
    "published_at.start": (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z",
    "per_page": 1,
})
crisis_signals["high_authority_negative"] = response.json().get("total_results", 0)

# Check crisis keywords
response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "entity.name": BRAND,
    "title": ",".join(CRISIS_KEYWORDS),
    "published_at.start": (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z",
    "per_page": 1,
})
crisis_signals["crisis_keywords"] = response.json().get("total_results", 0)

# Calculate crisis score (0-100)
crisis_score = min(100, (
    crisis_signals["breaking_negative"] * 20 +
    crisis_signals["high_authority_negative"] * 10 +
    crisis_signals["crisis_keywords"] * 15
))

print(f"Crisis Score for {BRAND}: {crisis_score}/100")
print(f"Signals: {crisis_signals}")

if crisis_score >= 50:
    print("ALERT: Crisis threshold exceeded!")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const BRAND = "Tesla";
const CRISIS_KEYWORDS = ["lawsuit", "scandal", "fraud", "recall", "investigation"];

async function detectCrisis() {
  const signals = {
    breakingNegative: 0,
    highAuthorityNegative: 0,
    crisisKeywords: 0,
  };

  const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

  // Breaking negative news
  const breakingParams = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": BRAND,
    "entity.type": "organization",
    is_breaking: "true",
    "sentiment.overall.polarity": "negative",
    per_page: "1",
  });

  let response = await fetch(`${BASE_URL}?${breakingParams}`);
  let data = await response.json();
  signals.breakingNegative = data.total_results || 0;

  // High-authority negative coverage
  const authorityParams = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": BRAND,
    "sentiment.overall.polarity": "negative",
    "source.rank.opr.min": "0.7",
    "published_at.start": yesterday,
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${authorityParams}`);
  data = await response.json();
  signals.highAuthorityNegative = data.total_results || 0;

  // Crisis keywords
  const keywordParams = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": BRAND,
    title: CRISIS_KEYWORDS.join(","),
    "published_at.start": yesterday,
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${keywordParams}`);
  data = await response.json();
  signals.crisisKeywords = data.total_results || 0;

  // Calculate score
  const crisisScore = Math.min(100,
    signals.breakingNegative * 20 +
    signals.highAuthorityNegative * 10 +
    signals.crisisKeywords * 15
  );

  console.log(`Crisis Score for ${BRAND}: ${crisisScore}/100`);
  console.log("Signals:", signals);

  if (crisisScore >= 50) {
    console.log("ALERT: Crisis threshold exceeded!");
  }

  return { crisisScore, signals };
}

detectCrisis();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$brand = "Tesla";
$crisisKeywords = ["lawsuit", "scandal", "fraud", "recall", "investigation"];

$signals = [
    "breaking_negative"       => 0,
    "high_authority_negative" => 0,
    "crisis_keywords"         => 0,
];

$yesterday = (new DateTime("-24 hours"))->format("c");

// Breaking negative news
$query = http_build_query([
    "api_key"                    => $apiKey,
    "entity.name"                => $brand,
    "entity.type"                => "organization",
    "is_breaking"                => "true",
    "sentiment.overall.polarity" => "negative",
    "per_page"                   => 1,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["breaking_negative"] = $data["total_results"] ?? 0;

// High-authority negative
$query = http_build_query([
    "api_key"                    => $apiKey,
    "entity.name"                => $brand,
    "sentiment.overall.polarity" => "negative",
    "source.rank.opr.min"        => 0.7,
    "published_at.start"         => $yesterday,
    "per_page"                   => 1,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["high_authority_negative"] = $data["total_results"] ?? 0;

// Crisis keywords
$query = http_build_query([
    "api_key"            => $apiKey,
    "entity.name"        => $brand,
    "title"              => implode(",", $crisisKeywords),
    "published_at.start" => $yesterday,
    "per_page"           => 1,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["crisis_keywords"] = $data["total_results"] ?? 0;

// Calculate score
$crisisScore = min(100,
    $signals["breaking_negative"] * 20 +
    $signals["high_authority_negative"] * 10 +
    $signals["crisis_keywords"] * 15
);

echo "Crisis Score for {$brand}: {$crisisScore}/100\n";
print_r($signals);

if ($crisisScore >= 50) {
    echo "ALERT: Crisis threshold exceeded!\n";
}
```

## Common Use Cases

- **Multi-signal crisis detection** — combine breaking news, sentiment, authority, and keyword signals into a composite crisis score.
- **Escalation tracking** — monitor how crisis coverage spreads from niche to mainstream media.
- **Cross-language crisis monitoring** — track crisis mentions across global markets in multiple languages.
- **Executive reputation protection** — monitor personal mentions of C-suite executives with sentiment analysis.
- **Recovery metrics** — track sentiment normalization and positive coverage recovery after crisis events.
- **Competitor crisis intelligence** — monitor competitor crises for market opportunity identification.
- **Regulatory crisis alerts** — detect news about investigations, lawsuits, and compliance issues.

## See Also

- [examples.md](./examples.md) — detailed code examples for crisis management workflows.
