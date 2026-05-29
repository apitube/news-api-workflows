# Company Profiles

Workflow for browsing the company directory and pulling per-company media coverage profiles using the [APITube News API](https://apitube.io).

## Overview

The **Company Profiles** workflow lets you look up organizations and brands in the APITube entity directory and retrieve a structured coverage profile for any single company. Use `GET /v1/companies` to search and paginate the directory (filtering by `name` or `wikidata_id`), then `GET /v1/companies/:id` to fetch a full profile with media metrics: article volume, sentiment breakdown, momentum, a monthly timeline, top sources/topics/countries/languages, related entities, and the most recent articles. This is useful for building company reference pages, competitive intelligence views, and media monitoring dashboards.

## API Endpoint

```
GET https://api.apitube.io/v1/companies
GET https://api.apitube.io/v1/companies/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/companies?api_key=YOUR_API_KEY
```

## Parameters

### List: `GET /v1/companies`

| Parameter     | Type    | Description                                                   |
|---------------|---------|---------------------------------------------------------------|
| `api_key`     | string  | **Required.** Your API key.                                   |
| `name`        | string  | Partial name match to filter companies.                       |
| `wikidata_id` | string  | Exact Wikidata ID match (e.g. `Q312`).                        |
| `page`        | integer | Page number for pagination (default: 1).                      |
| `per_page`    | integer | Number of results per page.                                   |

### Profile: `GET /v1/companies/:id`

| Parameter  | Type    | Description                                                              |
|------------|---------|-------------------------------------------------------------------------|
| `api_key`  | string  | **Required.** Your API key.                                             |
| `coverage` | boolean | Set to `false` to omit the `coverage` block and return identity only.   |

Companies are entities of type `organization` or `brand`. A request for an ID that is not found, or that does not resolve to a company, returns `404` with error code `ER0151`.

Within `coverage`, the fields `momentum.change_pct`, `first_seen`, and `last_seen` can be `null` (e.g. no articles in the previous 30-day window), and the whole `coverage` block can be `null` when analytics are unavailable — guard for these in your code.

## Quick Start

### cURL

```bash
# Search the directory by name
curl -s "https://api.apitube.io/v1/companies?api_key=YOUR_API_KEY&name=apple&per_page=10"

# Fetch a full profile with coverage metrics
curl -s "https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY"

# Fetch identity only, without coverage
curl -s "https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY&coverage=false"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/companies", params={
    "api_key": "YOUR_API_KEY",
    "name": "apple",
    "per_page": 10,
})
response.raise_for_status()

data = response.json()
for company in data["results"]:
    print(f"{company['id']}  {company['name']} ({company['type']})")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  name: "apple",
  per_page: "10",
});

const response = await fetch(`https://api.apitube.io/v1/companies?${params}`);
const data = await response.json();

data.results.forEach((company) => {
  console.log(`${company.id}  ${company.name} (${company.type})`);
});
```

### PHP

```php
<?php

$query = http_build_query([
    "api_key"  => "YOUR_API_KEY",
    "name"     => "apple",
    "per_page" => 10,
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/companies?{$query}"
), true);

foreach ($data["results"] as $company) {
    echo "{$company['id']}  {$company['name']} ({$company['type']})\n";
}
```

## Response Example

### List response (`/v1/companies`)

```json
{
  "status": "ok",
  "limit": 50,
  "page": 1,
  "has_next_pages": true,
  "results": [
    {
      "id": 312,
      "name": "Apple",
      "type": "organization",
      "links": {
        "self": "https://api.apitube.io/v1/companies/312",
        "articles": "https://api.apitube.io/v1/news/entity/312",
        "wikipedia": "https://en.wikipedia.org/wiki/Apple_Inc.",
        "wikidata": "https://www.wikidata.org/wiki/Q312"
      },
      "profile": {
        "name": "Apple",
        "type": "organization",
        "country": { "code": "US", "name": "United States" },
        "description": "Technology company"
      }
    }
  ]
}
```

### Profile response (`/v1/companies/:id`)

```json
{
  "id": 312,
  "name": "Apple",
  "type": "organization",
  "links": {
    "self": "https://api.apitube.io/v1/companies/312",
    "articles": "https://api.apitube.io/v1/news/entity/312",
    "wikipedia": "https://en.wikipedia.org/wiki/Apple_Inc.",
    "wikidata": "https://www.wikidata.org/wiki/Q312"
  },
  "profile": {
    "name": "Apple",
    "type": "organization",
    "country": { "code": "US", "name": "United States" },
    "description": "Technology company"
  },
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
  },
  "recent_articles": [
    {
      "id": 988111,
      "title": "Apple unveils new product line",
      "sentiment": { "overall": { "polarity": "positive", "score": 0.42 } },
      "source": { "domain": "example.com" }
    }
  ]
}
```

## Common Use Cases

- **Company reference pages** — search the directory by name and render an identity card plus coverage metrics.
- **Competitive intelligence** — read `related_entities` to surface partners, competitors, and key people associated with a company.
- **Media volume monitoring** — track `article_count`, `momentum`, and the monthly `timeline` for a watched organization.
- **Reputation snapshots** — combine the `sentiment` breakdown with `recent_articles` to assess how a brand is being covered right now.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
