# Entity Tracking

Workflow for tracking specific entities (companies, people, organizations) in news using the [APITube News API](https://apitube.io).

## Overview

The **Entity Tracking** workflow uses the API's entity recognition capabilities to monitor mentions of specific companies, people, locations, and other named entities across news sources. Combine entity filters with sentiment analysis to understand how entities are perceived in the media.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Entity Parameters

| Parameter                      | Type    | Description                                                        |
|-------------------------------|---------|-------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                       |
| `entity.name`                 | string  | Filter by entity name (e.g., `Apple`, `Elon Musk`).              |
| `entity.type`                 | string  | Filter by entity type: `organization`, `person`, `location`, etc. |
| `title`                       | string  | Filter by keywords in article title (comma-separated).            |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.           |
| `sort.by`                     | string  | Sort field (e.g., `published_at`, `sentiment.overall.score`).     |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                  |
| `source.domain`               | string  | Filter by source domain (comma-separated).                        |
| `language`                    | string  | Filter by language code (e.g., `en`, `fr`).                      |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                           |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                             |
| `per_page`                    | integer | Number of results per page (default: 50).                         |
| `page`                        | integer | Page number for pagination.                                       |

## Quick Start

### cURL

```bash
# Track news about Apple as an organization
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Apple&entity.type=organization&language=en&per_page=10"

# Track a person with sentiment filter
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Elon+Musk&entity.type=person&sentiment.overall.polarity=negative&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "entity.name": "Apple",
    "entity.type": "organization",
    "language": "en",
    "per_page": 10,
})

data = response.json()
for article in data["results"]:
    print(f"{article['title']} — {article['source']['domain']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  "entity.name": "Apple",
  "entity.type": "organization",
  language: "en",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.results.forEach((a) => {
  console.log(`${a.title} — ${a.source.domain}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"     => "YOUR_API_KEY",
    "entity.name" => "Apple",
    "entity.type" => "organization",
    "language"    => "en",
    "per_page"    => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($data["results"] as $article) {
    echo "{$article['title']} — {$article['source']['domain']}\n";
}
```

## Common Use Cases

- **Company monitoring** — track all mentions of a company in global news.
- **Executive tracking** — monitor coverage of key executives and public figures.
- **Geopolitical analysis** — track entity mentions in specific countries or regions.
- **Reputation scoring** — combine entity tracking with sentiment to score reputation over time.

## See Also

- [examples.md](./examples.md) — detailed code examples for entity tracking workflows.
