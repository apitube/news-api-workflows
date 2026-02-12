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
| `organization.name`           | string  | Filter by organization name (e.g., `Tesla`, `Boeing`).               |
| `person.name`                 | string  | Filter by person name (e.g., `Elon Musk`).                           |
| `brand.name`                  | string  | Filter by brand name.                                                |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `sentiment.overall.score.min` | float   | Minimum sentiment score (-1.0 to 1.0).                               |
| `sentiment.overall.score.max` | float   | Maximum sentiment score (-1.0 to 1.0).                               |
| `is_breaking`                 | integer | Filter for breaking news articles (`1` = yes, `0` = no).             |
| `title`                       | string  | Filter by keywords in title (comma-separated).                       |
| `source.rank.opr.min`         | integer | Minimum source authority rank (0–7).                                 |
| `source.domain`               | string  | Filter by source domains (comma-separated).                         |
| `source.country.code`         | string  | Filter by source country (ISO 3166-1).                              |
| `language.code`               | string  | Filter by language code (comma-separated for multi-lang).           |
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
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Tesla&is_breaking=1&sentiment.overall.polarity=negative&per_page=10" | jq '.results[] | {title, source: .source.domain, sentiment: .sentiment.overall}'

# Monitor crisis keywords across multiple languages
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=lawsuit,scandal,fraud,investigation&organization.name=Tesla&language.code=en,de,fr&per_page=20"

# Get high-authority negative coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&organization.name=Tesla&sentiment.overall.polarity=negative&source.rank.opr.min=6&sort.by=published_at&sort.order=desc&per_page=10"
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
    "organization.name": BRAND,
    "is_breaking": 1,
    "sentiment.overall.polarity": "negative",
    "per_page": 100,
})
crisis_signals["breaking_negative"] = len(response.json().get("results", []))

# Check high-authority negative coverage
response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "organization.name": BRAND,
    "sentiment.overall.polarity": "negative",
    "source.rank.opr.min": 6,
    "published_at.start": (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z",
    "per_page": 100,
})
crisis_signals["high_authority_negative"] = len(response.json().get("results", []))

# Check crisis keywords
response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "organization.name": BRAND,
    "title": ",".join(CRISIS_KEYWORDS),
    "published_at.start": (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z",
    "per_page": 100,
})
crisis_signals["crisis_keywords"] = len(response.json().get("results", []))

# Calculate crisis score (0-100)
crisis_score = min(100, (
    crisis_signals["breaking_negative"] * 2 +
    crisis_signals["high_authority_negative"] * 1 +
    crisis_signals["crisis_keywords"] * 1
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
    "organization.name": BRAND,
    is_breaking: "1",
    "sentiment.overall.polarity": "negative",
    per_page: "100",
  });

  let response = await fetch(`${BASE_URL}?${breakingParams}`);
  let data = await response.json();
  signals.breakingNegative = data.results?.length || 0;

  // High-authority negative coverage
  const authorityParams = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": BRAND,
    "sentiment.overall.polarity": "negative",
    "source.rank.opr.min": "6",
    "published_at.start": yesterday,
    per_page: "100",
  });

  response = await fetch(`${BASE_URL}?${authorityParams}`);
  data = await response.json();
  signals.highAuthorityNegative = data.results?.length || 0;

  // Crisis keywords
  const keywordParams = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": BRAND,
    title: CRISIS_KEYWORDS.join(","),
    "published_at.start": yesterday,
    per_page: "100",
  });

  response = await fetch(`${BASE_URL}?${keywordParams}`);
  data = await response.json();
  signals.crisisKeywords = data.results?.length || 0;

  // Calculate score
  const crisisScore = Math.min(100,
    signals.breakingNegative * 2 +
    signals.highAuthorityNegative * 1 +
    signals.crisisKeywords * 1
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
    "organization.name"          => $brand,
    "is_breaking"                => 1,
    "sentiment.overall.polarity" => "negative",
    "per_page"                   => 100,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["breaking_negative"] = count($data["results"] ?? []);

// High-authority negative
$query = http_build_query([
    "api_key"                    => $apiKey,
    "organization.name"          => $brand,
    "sentiment.overall.polarity" => "negative",
    "source.rank.opr.min"        => 6,
    "published_at.start"         => $yesterday,
    "per_page"                   => 100,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["high_authority_negative"] = count($data["results"] ?? []);

// Crisis keywords
$query = http_build_query([
    "api_key"            => $apiKey,
    "organization.name"  => $brand,
    "title"              => implode(",", $crisisKeywords),
    "published_at.start" => $yesterday,
    "per_page"           => 100,
]);
$data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
$signals["crisis_keywords"] = count($data["results"] ?? []);

// Calculate score
$crisisScore = min(100,
    $signals["breaking_negative"] * 2 +
    $signals["high_authority_negative"] * 1 +
    $signals["crisis_keywords"] * 1
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
