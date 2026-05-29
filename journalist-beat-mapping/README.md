# Journalist Beat Mapping

Infer a journalist's beat (topical specialization) from their coverage profile using the [APITube News API](https://apitube.io).

## Overview

The **Journalist Beat Mapping** workflow turns the `coverage` block of a journalist profile into an expertise fingerprint. The profile endpoint returns `top_topics` and `top_entities` (what the author writes about and who they cover), `top_countries` and `top_languages` (their geographic and linguistic footprint), and a `sentiment` breakdown (their tonal lean). By reading these fields you can classify a journalist's beat, compare several journalists by subject area, and decide who to approach for a given story.

## API Endpoint

```
GET https://api.apitube.io/v1/journalists/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/journalists/88123?api_key=YOUR_API_KEY
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `coverage` | boolean | Set to `false` to omit the `coverage` block. Beat mapping requires coverage, so leave it enabled. |

Returns `404` (`ER0151`) if the journalist is not found.

The `coverage` block (JOURNALIST form) used for beat mapping contains:

| Field | Description |
|-------|-------------|
| `article_count` | Total articles attributed to the journalist. |
| `sentiment` | Counts of `positive`, `neutral`, and `negative` articles. |
| `momentum` | `last_30_days`, `previous_30_days`, `change_pct`. |
| `timeline` | Article counts per period. |
| `top_topics` | Most-covered topics: `{ id, name, count }`. |
| `top_entities` | Most-covered entities: `{ id, name, count }`. |
| `top_countries` | Geographic footprint: `{ id, name, code, count }`. |
| `top_languages` | Publishing languages: `{ id, name, code, count }`. |

## Quick Start

### cURL

```bash
curl -s "https://api.apitube.io/v1/journalists/88123?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

response = requests.get(
    "https://api.apitube.io/v1/journalists/88123",
    params={"api_key": "YOUR_API_KEY"},
)
response.raise_for_status()
coverage = response.json()["coverage"]

beat = coverage["top_topics"][0]["name"] if coverage["top_topics"] else "Unknown"
print(f"Primary beat: {beat}")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({ api_key: "YOUR_API_KEY" });
const response = await fetch(`https://api.apitube.io/v1/journalists/88123?${params}`);
const { coverage } = await response.json();

const beat = coverage.top_topics[0]?.name ?? "Unknown";
console.log(`Primary beat: ${beat}`);
```

### PHP

```php
$query = http_build_query(["api_key" => "YOUR_API_KEY"]);
$coverage = json_decode(file_get_contents(
    "https://api.apitube.io/v1/journalists/88123?{$query}"
), true)["coverage"];

$beat = $coverage["top_topics"][0]["name"] ?? "Unknown";
echo "Primary beat: {$beat}\n";
```

## Response Example

```json
{
  "id": 88123,
  "name": "Jane Doe",
  "outlets": [ { "id": 4232, "name": "Example News", "domain": "example.com" } ],
  "outlet_count": 1,
  "coverage": {
    "article_count": 1820,
    "sentiment": { "positive": 600, "neutral": 900, "negative": 320 },
    "momentum": { "last_30_days": 42, "previous_30_days": 51, "change_pct": -18 },
    "timeline": [ { "period": "2024-06-01", "count": 38 } ],
    "top_topics": [
      { "id": "politics", "name": "Politics", "count": 410 },
      { "id": "technology", "name": "Technology", "count": 180 }
    ],
    "top_entities": [ { "id": 5021, "name": "Elon Musk", "count": 120 } ],
    "top_countries": [ { "id": 840, "name": "United States", "code": "us", "count": 1500 } ],
    "top_languages": [ { "id": 1, "name": "English", "code": "en", "count": 1700 } ]
  },
  "recent_articles": []
}
```

Note that `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (e.g. no articles in the previous 30-day window), and the entire `coverage` block may be `null` when analytics are unavailable — guard for these before formatting.

## Common Use Cases

- **Beat classification** — derive a journalist's primary and secondary beats from `top_topics` share.
- **Source mapping for PR** — find which journalists cover a specific topic or entity and how heavily.
- **Comparative analysis** — rank several journalists by their coverage of the same subject area.
- **Tone profiling** — read the `sentiment` split to gauge whether an author skews positive or critical.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
