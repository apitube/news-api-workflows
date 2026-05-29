# People Profiles

Search public figures and retrieve their media coverage profiles using the [APITube News API](https://apitube.io).

## Overview

The **People Profiles** workflow lets you find people (named entities of type `person`) by name or Wikidata ID, then pull a full profile for any person including their media coverage statistics. The list endpoint returns lightweight directory entries; the profile endpoint adds a `coverage` block (article counts, sentiment split, momentum, timeline, top sources/topics/countries/languages, related entities) and up to five `recent_articles`. This is useful for building people directories, briefing pages, and entity-centric dashboards.

## API Endpoint

```
GET https://api.apitube.io/v1/people
GET https://api.apitube.io/v1/people/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/people?api_key=YOUR_API_KEY
```

## Parameters

### List — `/v1/people`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `name` | string | Partial name match. |
| `wikidata_id` | string | Filter by Wikidata ID (e.g. `Q317521`). |
| `page` | integer | Page number for pagination. |
| `per_page` | integer | Number of results per page. |

### Profile — `/v1/people/:id`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `coverage` | boolean | Set to `false` to omit the `coverage` block and return only the profile and recent articles. |

Returns `404` (`ER0151`) if the ID is not found or is not a person.

Inside `coverage`, the fields `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (no prior 30-day window or no articles), and the whole `coverage` object may be `null` when analytics are unavailable — guard for these before formatting or sorting.

## Quick Start

### cURL

```bash
# Find people by name
curl -s "https://api.apitube.io/v1/people?api_key=YOUR_API_KEY&name=elon%20musk"

# Resolve a person by Wikidata ID
curl -s "https://api.apitube.io/v1/people?api_key=YOUR_API_KEY&wikidata_id=Q317521"

# Get a full profile with coverage
curl -s "https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY"

# Get a profile without the coverage block
curl -s "https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY&coverage=false"
```

### Python

```python
import requests

response = requests.get("https://api.apitube.io/v1/people", params={
    "api_key": "YOUR_API_KEY",
    "name": "elon musk",
})
response.raise_for_status()

for person in response.json()["results"]:
    print(f"{person['id']}: {person['name']} ({person['type']})")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({
  api_key: "YOUR_API_KEY",
  name: "elon musk",
});

const response = await fetch(`https://api.apitube.io/v1/people?${params}`);
const data = await response.json();

data.results.forEach((person) => {
  console.log(`${person.id}: ${person.name} (${person.type})`);
});
```

### PHP

```php
$query = http_build_query([
    "api_key" => "YOUR_API_KEY",
    "name"    => "elon musk",
]);

$data = json_decode(file_get_contents(
    "https://api.apitube.io/v1/people?{$query}"
), true);

foreach ($data["results"] as $person) {
    echo "{$person['id']}: {$person['name']} ({$person['type']})\n";
}
```

## Response Example

List — `/v1/people?name=elon%20musk`:

```json
{
  "status": "ok",
  "limit": 50,
  "page": 1,
  "has_next_pages": true,
  "results": [
    {
      "id": 5021,
      "name": "Elon Musk",
      "type": "person",
      "links": {
        "self": "https://api.apitube.io/v1/people/5021",
        "articles": "https://api.apitube.io/v1/news/entity/5021",
        "wikipedia": "https://en.wikipedia.org/wiki/Elon_Musk",
        "wikidata": "https://www.wikidata.org/wiki/Q317521"
      },
      "profile": {
        "name": "Elon Musk",
        "type": "person",
        "country": { "code": "US", "name": "United States" },
        "description": "Business magnate"
      }
    }
  ]
}
```

Profile — `/v1/people/5021`:

```json
{
  "id": 5021,
  "name": "Elon Musk",
  "type": "person",
  "links": {
    "self": "https://api.apitube.io/v1/people/5021",
    "articles": "https://api.apitube.io/v1/news/entity/5021",
    "wikipedia": "https://en.wikipedia.org/wiki/Elon_Musk",
    "wikidata": "https://www.wikidata.org/wiki/Q317521"
  },
  "profile": {
    "name": "Elon Musk",
    "type": "person",
    "country": { "code": "US", "name": "United States" },
    "description": "Business magnate"
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
      { "id": 312, "name": "Tesla", "count": 4100 }
    ]
  },
  "recent_articles": []
}
```

## Common Use Cases

- **People directory search** — resolve a free-text name or Wikidata ID into a stable person ID.
- **Briefing pages** — render a person's coverage profile with sentiment split and momentum.
- **Entity graph traversal** — follow `related_entities` to discover connected people and organizations.
- **Lightweight lookups** — pass `coverage=false` when you only need the profile and recent articles.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
