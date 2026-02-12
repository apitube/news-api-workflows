# Multi-Source Monitoring

Workflow for monitoring and comparing news from specific sources using the [APITube News API](https://apitube.io).

## Overview

The **Multi-Source Monitoring** workflow lets you filter news by source domain, source country, and source rank. Build custom feeds from trusted sources, compare coverage across outlets, and filter by source quality metrics.

## API Endpoint

```
GET https://api.apitube.io/v1/news/everything
```

## Source Parameters

| Parameter                      | Type    | Description                                                            |
|-------------------------------|---------|------------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                            |
| `source.domain`               | string  | Filter by source domain (comma-separated, e.g., `cnn.com,bbc.com`).  |
| `source.country.code`         | string  | Filter by source country code (comma-separated, e.g., `us,gb`).      |
| `source.rank.opr.min`         | integer | Minimum source OPR rank (0–7). Higher = more authoritative.            |
| `source.rank.opr.max`         | integer | Maximum source OPR rank (0–7).                                         |
| `title`                       | string  | Filter by keywords in article title (comma-separated).                |
| `topic.id`                    | string  | Filter by topic (e.g., `crypto_news`, `climate_change`).              |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.               |
| `sort.by`                     | string  | Sort field: `published_at`, `source.domain`, `sentiment.overall.score`.|
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                      |
| `language.code`               | string  | Filter by language code.                                               |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                                |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                                  |
| `per_page`                    | integer | Number of results per page (default: 50).                              |
| `page`                        | integer | Page number for pagination.                                            |

## Quick Start

### cURL

```bash
# Get news from specific sources
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&source.domain=reuters.com,bbc.com,bloomberg.com&language.code=en&per_page=10"

# Get news from high-ranked sources only
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&source.rank.opr.min=6&language.code=en&per_page=10&sort.by=published_at&sort.order=desc"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/news/everything", params={
    "api_key": "YOUR_API_KEY",
    "source.domain": "reuters.com,bbc.com,bloomberg.com",
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 10,
})

data = response.json()
for article in data["results"]:
    print(f"[{article['source']['domain']}] {article['title']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  "source.domain": "reuters.com,bbc.com,bloomberg.com",
  "language.code": "en",
  "sort.by": "published_at",
  "sort.order": "desc",
  per_page: "10",
});

const response = await fetch(
  `https://api.apitube.io/v1/news/everything?${params}`
);
const data = await response.json();

data.results.forEach((a) => {
  console.log(`[${a.source.domain}] ${a.title}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"       => "YOUR_API_KEY",
    "source.domain" => "reuters.com,bbc.com,bloomberg.com",
    "language.code" => "en",
    "sort.by"       => "published_at",
    "sort.order"    => "desc",
    "per_page"      => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/news/everything?{$query}"
), true);

foreach ($data["results"] as $article) {
    echo "[{$article['source']['domain']}] {$article['title']}\n";
}
```

## Common Use Cases

- **Custom news feeds** — aggregate articles from a curated list of trusted sources.
- **Source quality filtering** — use OPR rank to ensure only authoritative sources appear.
- **Regional monitoring** — track news from sources in specific countries.
- **Coverage comparison** — compare how different outlets report on the same topic.
- **Media mix analysis** — understand the distribution of coverage across sources.

## See Also

- [examples.md](./examples.md) — detailed code examples for multi-source monitoring workflows.
