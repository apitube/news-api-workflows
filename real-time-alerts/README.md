# Real-Time Alerts

Workflow for building automated alerting pipelines that detect breaking news, sentiment spikes, entity mention surges, and anomalous coverage patterns using the [APITube News API](https://apitube.io).

## Overview

The **Real-Time Alerts** workflow combines polling, thresholds, and multi-signal detection to monitor news streams in real time. Detect breaking events as they happen, identify sentiment anomalies, track entity mention velocity, and build composite crisis scoring dashboards. Ideal for reputation management, crisis detection, competitive intelligence, and event-driven automation.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                        |
|-------------------------------|---------|-------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                       |
| `entity.name`                 | string  | Filter by entity name (e.g., `Tesla`, `Elon Musk`).              |
| `entity.type`                 | string  | Filter by entity type: `organization`, `person`, `location`, etc. |
| `topic.id`                    | string  | Filter by topic ID (comma-separated).                             |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.           |
| `is_breaking`                 | boolean | Filter for breaking news articles (`true` or `false`).            |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                           |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                             |
| `sort.by`                     | string  | Sort field (e.g., `published_at`, `sentiment.overall.score`).     |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                  |
| `per_page`                    | integer | Number of results per page (default: 50).                         |
| `page`                        | integer | Page number for pagination.                                       |
| `language`                    | string  | Filter by language code (e.g., `en`, `fr`).                      |
| `source.domain`               | string  | Filter by source domain (comma-separated).                        |
| `source.rank.opr.min`         | integer | Filter sources by minimum OPR rank.                               |

## Quick Start

### cURL

```bash
# Get breaking news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&is_breaking=true&language=en&per_page=10"

# Get recent negative news about an entity
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Tesla&sentiment.overall.polarity=negative&published_at.start=2026-02-07&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "is_breaking": "true",
    "language": "en",
    "per_page": 10,
})

data = response.json()
for article in data["results"]:
    print(f"[BREAKING] {article['title']}")
    print(f"  {article['source']['domain']} — {article['href']}\n")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  is_breaking: "true",
  language: "en",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(`[BREAKING] ${article.title}`);
  console.log(`  ${article.source.domain} — ${article.href}\n`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"     => "YOUR_API_KEY",
    "is_breaking" => "true",
    "language"    => "en",
    "per_page"    => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($data["results"] as $article) {
    echo "[BREAKING] {$article['title']}\n";
    echo "  {$article['source']['domain']} — {$article['href']}\n\n";
}
```

## Common Use Cases

- **Breaking news detection** — poll for `is_breaking=true` and alert immediately on new articles.
- **Sentiment spike alerts** — track hourly sentiment scores and detect sudden negative swings.
- **Entity mention surge detection** — monitor entity mention velocity and alert when it exceeds baseline by 2x or more.
- **Multi-signal crisis dashboards** — combine breaking news count, negative sentiment ratio, mention velocity, and source diversity into composite scores.
- **Scheduled threshold monitors** — run periodic checks (e.g., every 5 minutes) and trigger alerts when metrics cross predefined thresholds.

## See Also

- [examples.md](./examples.md) — detailed code examples for real-time alerting workflows.
