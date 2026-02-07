# Multilingual Analysis

Workflow for cross-language news monitoring, comparison, and analytics using the [APITube News API](https://apitube.io).

## Overview

The **Multilingual Analysis** workflow lets you track how the same story is covered across different languages and regions. Compare sentiment, volume, and source diversity across languages. Monitor global brand perception, track international events, and analyze how topics resonate in different markets.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type   | Description                                                        |
|-------------------------------|--------|--------------------------------------------------------------------|
| `api_key`                     | string | **Required.** Your API key.                                        |
| `language`                    | string | Filter by language code (e.g., `en`, `fr`, `de`, `es`, `zh`).     |
| `source.country.code`         | string | Filter by source country (ISO 3166-1 alpha-2).                    |
| `entity.name`                 | string | Filter by entity name (person, organization, location, etc.).     |
| `entity.type`                 | string | Filter by entity type: `person`, `organization`, `location`.      |
| `topic.id`                    | string | Filter by topic (e.g., `cryptocurrency`, `climate_change`).       |
| `sentiment.overall.polarity`  | string | Filter by polarity: `positive`, `negative`, or `neutral`.         |
| `title`                       | string | Filter by keywords in article title (comma-separated).            |
| `sort.by`                     | string | Sort field (e.g., `published_at`, `sentiment.overall.score`).     |
| `sort.order`                  | string | Sort direction: `asc` or `desc`.                                  |
| `per_page`                    | integer| Number of results per page (default: 50).                         |
| `page`                        | integer| Page number for pagination.                                       |
| `published_at.start`          | string | Start date for filtering (ISO 8601 or `YYYY-MM-DD`).             |
| `published_at.end`            | string | End date for filtering (ISO 8601 or `YYYY-MM-DD`).               |

## Quick Start

### cURL

```bash
# Get Tesla coverage across multiple languages
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Tesla&language=en&per_page=10"

# Compare climate change coverage in German vs French sources
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&topic.id=climate_change&language=de&per_page=10"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "entity.name": "Tesla",
    "language": "en",
    "per_page": 10,
})

data = response.json()
for article in data["results"]:
    print(f"[{article['language']}] {article['title']}")
    print(f"  Source: {article['source']['domain']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  "entity.name": "Tesla",
  language: "en",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.results.forEach((article) => {
  console.log(`[${article.language}] ${article.title}`);
  console.log(`  Source: ${article.source.domain}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"     => "YOUR_API_KEY",
    "entity.name" => "Tesla",
    "language"    => "en",
    "per_page"    => 10,
]);

$response = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($response["results"] as $article) {
    echo "[{$article['language']}] {$article['title']}\n";
    echo "  Source: {$article['source']['domain']}\n";
}
```

## Common Use Cases

- **Global brand monitoring** — track how a brand is covered across different languages and markets.
- **Cross-language sentiment comparison** — compare positive, negative, and neutral coverage across languages.
- **Regional coverage analysis** — identify which regions/languages provide the most coverage for a topic.
- **Language-specific topic trends** — discover which topics resonate differently in different language markets.
- **International PR monitoring** — track press releases and announcements across global media.

## See Also

- [examples.md](./examples.md) — detailed code examples for multilingual analysis workflows.
