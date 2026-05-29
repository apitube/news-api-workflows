# Source Benchmarking

Workflow for comparing the publishing coverage of several news sources side by side using the [APITube News API](https://apitube.io).

## Overview

The **Source Benchmarking** workflow polls the source profile endpoint (`/v1/sources/:id`) for a list of publisher IDs and compares their coverage metrics: total article count, sentiment balance (positive / neutral / negative), and 30-day publishing momentum. With those numbers you can rank publishers by output, rank them by positivity, and spot which outlets are accelerating or slowing down. The endpoint returns the source summary coverage form, so the comparison uses only `article_count`, `first_seen`, `last_seen`, `sentiment`, `momentum`, and `timeline`.

## API Endpoint

```
GET https://api.apitube.io/v1/sources/:id
```

Call the endpoint once per source ID you want to benchmark.

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/sources/4232?api_key=YOUR_API_KEY
```

## Parameters

| Parameter   | Type    | Description                                                                  |
|-------------|---------|------------------------------------------------------------------------------|
| `api_key`   | string  | **Required.** Your API key.                                                  |
| `:id`       | integer | **Required.** Source ID in the path (e.g., `/v1/sources/4232`).              |
| `coverage`  | boolean | Set to `false` to omit the `coverage` block. Leave default to keep coverage. |

Returns `404` (`ER0151`) if a source ID is not found.

## Quick Start

### cURL

```bash
# Pull the coverage block for one source
curl -s "https://api.apitube.io/v1/sources/4232?api_key=YOUR_API_KEY"

# Benchmark several sources by calling the endpoint per ID
for id in 4232 771 5510; do
  curl -s "https://api.apitube.io/v1/sources/${id}?api_key=YOUR_API_KEY"
done
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]

for source_id in SOURCE_IDS:
    response = requests.get(
        f"https://api.apitube.io/v1/sources/{source_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    source = response.json()
    cov = source["coverage"] or {}
    change = cov["momentum"]["change_pct"]
    change_str = "n/a" if change is None else f"{change:+d}%"
    print(f"{source['name']:<25} {cov['article_count']:>10,}  {change_str}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

for (const id of SOURCE_IDS) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/sources/${id}?${params}`);
  const source = await response.json();
  const cov = source.coverage ?? {};
  const change = cov.momentum.change_pct == null
    ? "n/a"
    : `${cov.momentum.change_pct >= 0 ? "+" : ""}${cov.momentum.change_pct}%`;
  console.log(
    `${source.name.padEnd(25)} ${cov.article_count.toLocaleString().padStart(10)}  ${change}`
  );
}
```

### PHP

```php
$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

foreach ($sourceIds as $id) {
    $query  = http_build_query(["api_key" => $apiKey]);
    $source = json_decode(file_get_contents(
        "https://api.apitube.io/v1/sources/{$id}?{$query}"
    ), true);
    $cov = $source["coverage"] ?? [];
    $changeStr = $cov["momentum"]["change_pct"] === null
        ? "n/a"
        : sprintf("%+d%%", $cov["momentum"]["change_pct"]);
    printf("%-25s %10s  %s\n",
        $source["name"], number_format($cov["article_count"]), $changeStr);
}
```

## Response Example

A single `/v1/sources/:id` response used in benchmarking:

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

The source `coverage` block is the summary form. It exposes `article_count`, `first_seen`, `last_seen`, `sentiment`, `momentum`, and `timeline`, and contains no `top_*` breakdowns. Note that `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (e.g. no articles in the previous 30-day window), and the entire `coverage` block may be `null` when analytics are unavailable — guard for these before formatting, comparing, or sorting.

## Common Use Cases

- **Output benchmark** — rank a set of publishers by total tracked article count.
- **Sentiment profiling** — compare the positive / neutral / negative balance of each publisher's coverage.
- **Momentum scan** — find which outlets are publishing more (or fewer) articles than the previous 30 days via `momentum.change_pct`.
- **Positivity ranking** — order publishers by their share of positive coverage.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
