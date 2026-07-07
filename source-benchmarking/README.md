# Source Benchmarking

Workflow for comparing several news sources side by side using the [APITube News API](https://apitube.io).

## Overview

The **Source Benchmarking** workflow compares a list of publishers on three metrics — total article volume, sentiment balance (positive / neutral / negative), and 30-day publishing momentum — and ranks them by output, positivity, or acceleration. Because APITube does not expose a per-source coverage endpoint, every metric is derived from **`/v1/news/count`** scoped with `source.id`: a plain count for volume, three counts filtered by `sentiment.overall.polarity` for the sentiment split, and two date-windowed counts for momentum. Resolve publisher names into IDs first with [`/v1/suggest/sources`](../source-directory/README.md).

## API Endpoints

```
GET https://api.apitube.io/v1/suggest/sources    (resolve a name/domain prefix into source IDs)
GET https://api.apitube.io/v1/news/count           (one metric per call, scoped by source.id)
```

`/v1/news/count` costs **1 point** per call and returns `{ "status": "ok", "count": <int>, "request_id": "..." }`.

## Authentication

All requests require an API key passed via the `api_key` query parameter (or the `X-API-Key` header):

```
https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&source.id=4232
```

## Parameters

| Parameter                   | Type    | Description                                                            |
|-----------------------------|---------|-----------------------------------------------------------------------|
| `api_key`                   | string  | **Required.** Your API key.                                           |
| `source.id`                 | integer | Publisher to scope the count to. Accepts up to 3 comma-separated IDs. |
| `published_at.start` / `.end` | string | Date window (ISO 8601 or date math like `NOW-30DAYS`).                |
| `sentiment.overall.polarity` | string | `positive`, `negative`, or `neutral` — for the sentiment split.       |

## Quick Start

### cURL

```bash
# Total volume for one source
curl -s "https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&source.id=4232"

# Last-30-days volume (for momentum)
curl -s "https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&source.id=4232&published_at.start=NOW-30DAYS"

# Positive-coverage volume
curl -s "https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&source.id=4232&sentiment.overall.polarity=positive"
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]
COUNT_URL = "https://api.apitube.io/v1/news/count"


def count(**filters):
    response = requests.get(COUNT_URL, params={"api_key": API_KEY, **filters})
    response.raise_for_status()
    return response.json()["count"]


for source_id in SOURCE_IDS:
    total = count(**{"source.id": source_id})
    last_30 = count(**{"source.id": source_id, "published_at.start": "NOW-30DAYS"})
    prev_30 = count(**{"source.id": source_id, "published_at.start": "NOW-60DAYS", "published_at.end": "NOW-30DAYS"})
    change = None if prev_30 == 0 else round((last_30 - prev_30) / prev_30 * 100)
    change_str = "n/a" if change is None else f"{change:+d}%"
    print(f"source {source_id:<8} total={total:>10,}  30d={last_30:>7,}  momentum={change_str}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function count(filters) {
  const params = new URLSearchParams({ api_key: API_KEY, ...filters });
  const response = await fetch(`https://api.apitube.io/v1/news/count?${params}`);
  const data = await response.json();
  return data.count;
}

for (const id of SOURCE_IDS) {
  const total = await count({ "source.id": id });
  const last30 = await count({ "source.id": id, "published_at.start": "NOW-30DAYS" });
  const prev30 = await count({ "source.id": id, "published_at.start": "NOW-60DAYS", "published_at.end": "NOW-30DAYS" });
  const change = prev30 === 0 ? "n/a" : `${Math.round(((last30 - prev30) / prev30) * 100)}%`;
  console.log(`source ${id}  total=${total.toLocaleString()}  30d=${last30.toLocaleString()}  momentum=${change}`);
}
```

### PHP

```php
$apiKey    = "YOUR_API_KEY";
$sourceIds = [4232, 771, 5510];

function count_articles(string $apiKey, array $filters): int {
    $query = http_build_query(array_merge(["api_key" => $apiKey], $filters));
    $data  = json_decode(file_get_contents(
        "https://api.apitube.io/v1/news/count?{$query}"
    ), true);
    return $data["count"];
}

foreach ($sourceIds as $id) {
    $total  = count_articles($apiKey, ["source.id" => $id]);
    $last30 = count_articles($apiKey, ["source.id" => $id, "published_at.start" => "NOW-30DAYS"]);
    $prev30 = count_articles($apiKey, ["source.id" => $id, "published_at.start" => "NOW-60DAYS", "published_at.end" => "NOW-30DAYS"]);
    $change = $prev30 === 0 ? "n/a" : sprintf("%+d%%", round(($last30 - $prev30) / $prev30 * 100));
    printf("source %-8d total=%10s  30d=%7s  momentum=%s\n",
        $id, number_format($total), number_format($last30), $change);
}
```

## Response Example

Each `/v1/news/count` call returns a small envelope:

```json
{
  "status": "ok",
  "count": 502310,
  "request_id": "b1e2..."
}
```

Build the benchmark by combining several counts per source: one unscoped (total volume), one per `sentiment.overall.polarity` value (sentiment split), and two date-windowed (`NOW-30DAYS` vs the preceding 30 days) for momentum. Guard against a zero previous-window count before computing a percentage change.

## Common Use Cases

- **Output benchmark** — rank a set of publishers by total tracked article count (`source.id` + `/v1/news/count`).
- **Sentiment profiling** — compare positive / neutral / negative shares with three `sentiment.overall.polarity` counts per source.
- **Momentum scan** — find which outlets are accelerating via last-30-days vs previous-30-days counts.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
- [source-directory](../source-directory/README.md) — resolve publisher names into source IDs.
