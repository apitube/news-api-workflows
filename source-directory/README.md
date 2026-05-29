# Source Directory

Workflow for browsing the publisher catalog and inspecting individual source profiles using the [APITube News API](https://apitube.io).

## Overview

The **Source Directory** workflow lets you list news publishers, filter them by name or country, and open a detailed profile for any source. The source list (`/v1/sources`) returns catalog records (domain, resource type, political bias, OpenPageRank), while the profile endpoint (`/v1/sources/:id`) adds publishing coverage and recent articles. From a source you can pivot straight to its articles through the `links.articles` URL, which targets `/v1/news/everything` with `source.id`.

## API Endpoint

```
GET https://api.apitube.io/v1/sources
GET https://api.apitube.io/v1/sources/:id
GET https://api.apitube.io/v1/news/everything   (for the source -> articles pivot)
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/sources?api_key=YOUR_API_KEY
```

## Parameters

### List — `GET /v1/sources`

| Parameter   | Type    | Description                                                        |
|-------------|---------|--------------------------------------------------------------------|
| `api_key`   | string  | **Required.** Your API key.                                        |
| `name`      | string  | Filter by source name (partial match).                             |
| `country`   | integer | Filter by integer country ID (e.g., `840` for the United States).  |
| `page`      | integer | Page number for pagination.                                        |
| `per_page`  | integer | Number of results per page.                                        |

### Profile — `GET /v1/sources/:id`

| Parameter   | Type    | Description                                                                  |
|-------------|---------|------------------------------------------------------------------------------|
| `api_key`   | string  | **Required.** Your API key.                                                  |
| `coverage`  | boolean | Set to `false` to omit the `coverage` block. Defaults to including coverage. |

Returns `404` (`ER0151`) if the source ID is not found.

## Quick Start

### cURL

```bash
# Find sources by name
curl -s "https://api.apitube.io/v1/sources?api_key=YOUR_API_KEY&name=guardian&per_page=10"

# List sources from a country (840 = United States)
curl -s "https://api.apitube.io/v1/sources?api_key=YOUR_API_KEY&country=840&per_page=20"

# Open a source profile (coverage + recent articles)
curl -s "https://api.apitube.io/v1/sources/4232?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/sources", params={
    "api_key": "YOUR_API_KEY",
    "name": "guardian",
    "per_page": 10,
})
response.raise_for_status()

for source in response.json()["results"]:
    print(f"{source['id']:>8}  {source['name']} ({source['domain']}) bias={source['bias']}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  name: "guardian",
  per_page: "10",
});

const response = await fetch(`https://api.apitube.io/v1/sources?${params}`);
const data = await response.json();

data.results.forEach((source) => {
  console.log(`${source.id}  ${source.name} (${source.domain}) bias=${source.bias}`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key"  => "YOUR_API_KEY",
    "name"     => "guardian",
    "per_page" => 10,
]);

$response = json_decode(file_get_contents(
    "https://api.apitube.io/v1/sources?{$query}"
), true);

foreach ($response["results"] as $source) {
    printf("%8d  %s (%s) bias=%s\n",
        $source["id"], $source["name"], $source["domain"], $source["bias"]);
}
```

## Response Example

### List — `GET /v1/sources`

```json
{
  "status": "ok",
  "limit": 10,
  "page": 1,
  "has_next_pages": false,
  "results": [
    {
      "id": 4232,
      "name": "Example News",
      "domain": "example.com",
      "resource_type": "news",
      "country_id": 840,
      "language_id": 1,
      "bias": "center",
      "rank": { "opr": 5 },
      "links": {
        "self": "https://api.apitube.io/v1/sources/4232",
        "articles": "https://api.apitube.io/v1/news/everything?source.id=4232",
        "website": "https://example.com"
      }
    }
  ]
}
```

### Profile — `GET /v1/sources/4232`

```json
{
  "id": 4232,
  "name": "Example News",
  "domain": "example.com",
  "resource_type": "news",
  "country_id": 840,
  "language_id": 1,
  "bias": "center",
  "rank": { "opr": 5 },
  "links": {
    "self": "https://api.apitube.io/v1/sources/4232",
    "articles": "https://api.apitube.io/v1/news/everything?source.id=4232",
    "website": "https://example.com"
  },
  "coverage": {
    "article_count": 502310,
    "first_seen": "2015-01-02",
    "last_seen": "2026-05-29",
    "sentiment": { "positive": 180400, "neutral": 250100, "negative": 71810 },
    "momentum": { "last_30_days": 8200, "previous_30_days": 7900, "change_pct": 3 },
    "timeline": [
      { "period": "2024-06-01", "count": 8100 }
    ]
  },
  "recent_articles": []
}
```

The source `coverage` block is the summary form: it contains `article_count`, `first_seen`, `last_seen`, `sentiment`, `momentum`, and `timeline` only. It does not include any `top_*` breakdowns. Note that `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (e.g. no articles in the previous 30-day window), and the entire `coverage` block may be `null` when analytics are unavailable — guard for these before formatting.

## Common Use Cases

- **Publisher lookup** — resolve a publisher name typed by a user into its source ID, domain, and bias.
- **Country catalog** — list every publisher tracked for a given country ID.
- **Source profile cards** — render a card with bias, OpenPageRank, total article count, and publishing momentum.
- **Source-to-articles pivot** — follow `links.articles` (or pass `source.id` to `/v1/news/everything`) to load a publisher's latest stories.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
