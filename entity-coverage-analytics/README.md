# Entity Coverage Analytics

Analyze the media coverage profile of a person or company in depth using the [APITube News API](https://apitube.io).

## Overview

The **Entity Coverage Analytics** workflow takes the `coverage` block returned by `/v1/people/:id` and `/v1/companies/:id` and turns it into usable analytics: a monthly volume trend from `timeline`, a sentiment breakdown from `sentiment`, momentum from `momentum`, share-of-coverage tables from `top_sources` / `top_countries`, and a co-mention graph from `related_entities`. Both endpoints return the same ENTITY-form coverage object (including `related_entities`), so a single analysis layer works for people and companies alike.

## API Endpoint

```
GET https://api.apitube.io/v1/people/:id
GET https://api.apitube.io/v1/companies/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `coverage` | boolean | Coverage is included by default. Set to `false` to omit it (not used by this workflow). |

`:id` is the stable entity ID from `/v1/people` or `/v1/companies`. Returns `404` (`ER0151`) if the ID is not found or is not the expected entity type.

### Coverage fields used by this workflow

| Field | Description |
|-------|-------------|
| `article_count` | Total articles mentioning the entity. |
| `first_seen` / `last_seen` | Coverage date range. |
| `sentiment` | Counts split into `positive`, `neutral`, `negative`. |
| `momentum` | `last_30_days`, `previous_30_days`, `change_pct`. |
| `timeline` | Array of `{ period, count }` points (monthly buckets). |
| `top_sources` | `{ id, name, domain, count }` per source. |
| `top_topics` | `{ id, name, count }` per topic. |
| `top_countries` | `{ id, name, code, count }` per country. |
| `top_languages` | `{ id, name, code, count }` per language. |
| `related_entities` | `{ id, name, count }` co-mentioned entities. |

Note: `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (no prior 30-day window or no articles), and the whole `coverage` object may be `null` when analytics are unavailable — guard for these before formatting, comparing, or sorting.

## Quick Start

### cURL

```bash
# Coverage analytics for a person
curl -s "https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY"

# Coverage analytics for a company
curl -s "https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

person = requests.get("https://api.apitube.io/v1/people/5021", params={
    "api_key": "YOUR_API_KEY",
}).json()

cov = person["coverage"]
if not cov:
    raise SystemExit(f"No coverage available for {person['name']}")

print(f"{person['name']}: {cov['article_count']:,} articles, "
      f"{len(cov['timeline'])} months on file")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({ api_key: "YOUR_API_KEY" });
const person = await (await fetch(`https://api.apitube.io/v1/people/5021?${params}`)).json();

const cov = person.coverage;
if (!cov) throw new Error(`No coverage available for ${person.name}`);
console.log(`${person.name}: ${cov.article_count.toLocaleString()} articles, ${cov.timeline.length} months on file`);
```

### PHP

```php
$query  = http_build_query(["api_key" => "YOUR_API_KEY"]);
$person = json_decode(file_get_contents(
    "https://api.apitube.io/v1/people/5021?{$query}"
), true);

$cov = $person["coverage"];
if (!$cov) {
    exit("No coverage available for {$person['name']}\n");
}
printf("%s: %s articles, %d months on file\n",
    $person["name"], number_format($cov["article_count"]), count($cov["timeline"]));
```

## Response Example

The relevant slice of `/v1/people/5021` (companies return the same shape):

```json
{
  "id": 5021,
  "name": "Elon Musk",
  "type": "person",
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
  }
}
```

## Common Use Cases

- **Coverage trend chart** — render `timeline` as a monthly ASCII bar chart to spot peaks and lulls.
- **Sentiment breakdown** — compute positive/neutral/negative shares and show the balance of coverage.
- **Share of coverage** — rank `top_sources` and `top_countries` by their share of total mentions.
- **Co-mention graph** — treat `related_entities` as weighted edges to map an entity's network.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
