# Share of Voice

Workflow for measuring share of voice across a set of competing companies using the [APITube News API](https://apitube.io).

## Overview

The **Share of Voice** workflow polls the company profile endpoint for a set of competitors and turns their coverage metrics into a comparable benchmark. By reading `coverage.article_count` for each company you can compute share of voice (SOV) as one company's article volume divided by the total across the set. The same profiles expose a `sentiment` breakdown and a `momentum` block, so you can benchmark tone and detect who is gaining or losing attention. This is useful for competitive PR reporting, market positioning, and brand tracking.

## API Endpoint

```
GET https://api.apitube.io/v1/companies/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY
```

## Parameters

| Parameter  | Type    | Description                                                            |
|------------|---------|-----------------------------------------------------------------------|
| `api_key`  | string  | **Required.** Your API key.                                           |
| `coverage` | boolean | Leave default (coverage included). Set to `false` to omit coverage.   |

The path segment `:id` is the company entity ID. Companies are entities of type `organization` or `brand`. A request for an ID that is not found, or that is not a company, returns `404` with error code `ER0151`.

To resolve company IDs from names first, use the directory endpoint `GET /v1/companies?name=...` (see the [Company Profiles](../company-profiles/) workflow).

Note: `momentum.change_pct`, `first_seen`, and `last_seen` inside `coverage` can be `null`, and the whole `coverage` block can be `null` when analytics are unavailable — guard for these before computing SOV, sentiment, or momentum.

## Quick Start

### cURL

```bash
# Pull coverage for each competitor, then compute SOV from article_count
curl -s "https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY"
curl -s "https://api.apitube.io/v1/companies/4501?api_key=YOUR_API_KEY"
curl -s "https://api.apitube.io/v1/companies/7720?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"
COMPETITORS = [312, 4501, 7720]

volumes = {}
for company_id in COMPETITORS:
    response = requests.get(
        f"https://api.apitube.io/v1/companies/{company_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    cov = profile.get("coverage")
    if cov:
        volumes[profile["name"]] = cov["article_count"]

total = sum(volumes.values()) or 1
for name, count in volumes.items():
    print(f"{name:<20} {count / total:6.1%}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const COMPETITORS = [312, 4501, 7720];

const volumes = {};
for (const id of COMPETITORS) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/companies/${id}?${params}`);
  const profile = await response.json();
  if (profile.coverage) volumes[profile.name] = profile.coverage.article_count;
}

const total = Object.values(volumes).reduce((a, b) => a + b, 0) || 1;
for (const [name, count] of Object.entries(volumes)) {
  console.log(`${name.padEnd(20)} ${((count / total) * 100).toFixed(1)}%`);
}
```

### PHP

```php
<?php

$apiKey      = "YOUR_API_KEY";
$competitors = [312, 4501, 7720];

$volumes = [];
foreach ($competitors as $id) {
    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/companies/{$id}?{$query}"
    ), true);
    if (!empty($profile["coverage"])) {
        $volumes[$profile["name"]] = $profile["coverage"]["article_count"];
    }
}

$total = array_sum($volumes) ?: 1;
foreach ($volumes as $name => $count) {
    printf("%-20s %5.1f%%\n", $name, $count / $total * 100);
}
```

## Response Example

A single competitor profile (`/v1/companies/:id`). Share of voice is derived from the `article_count` field across all polled companies:

```json
{
  "id": 312,
  "name": "Apple",
  "type": "organization",
  "coverage": {
    "article_count": 12840,
    "first_seen": "2019-03-11",
    "last_seen": "2026-05-29",
    "sentiment": { "positive": 4200, "neutral": 6100, "negative": 2540 },
    "momentum": { "last_30_days": 920, "previous_30_days": 760, "change_pct": 21 },
    "timeline": [
      { "period": "2024-06-01", "count": 410 },
      { "period": "2024-07-01", "count": 455 }
    ],
    "top_sources": [
      { "id": 4232, "name": "Example News", "domain": "example.com", "count": 320 }
    ],
    "top_topics": [
      { "id": "technology", "name": "Technology", "count": 540 }
    ],
    "top_countries": [
      { "id": 840, "name": "United States", "code": "us", "count": 6100 }
    ],
    "top_languages": [
      { "id": 1, "name": "English", "code": "en", "count": 9800 }
    ],
    "related_entities": [
      { "id": 5021, "name": "Tim Cook", "count": 1200 }
    ]
  }
}
```

## Common Use Cases

- **Share of voice reporting** — compute `article_count_i / sum(article_count)` across a competitor set and render an SOV table.
- **Sentiment benchmarking** — compare the positive/negative balance of each competitor's coverage side by side.
- **Momentum leaderboard** — rank competitors by `momentum.change_pct` to see who is gaining or losing media attention.
- **PR competitive briefs** — combine SOV, sentiment, and momentum into a single periodic snapshot for stakeholders.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
- [Company Profiles](../company-profiles/) — resolving company IDs and reading full profiles.
