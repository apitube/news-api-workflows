# Coverage Momentum Alerts

Workflow for detecting surges and declines in media coverage of companies and people using the [APITube News API](https://apitube.io).

## Overview

The **Coverage Momentum Alerts** workflow periodically polls profile endpoints for a watchlist of entities and reads the `momentum` block from each profile's `coverage`. The `momentum` object reports `last_30_days`, `previous_30_days`, and `change_pct`, which together describe whether attention is accelerating or fading. By thresholding `change_pct` you can classify each entity as a surge, a decline, or stable, and you can combine momentum with the `sentiment` breakdown to raise a targeted alert such as a negative coverage surge. Profiles for organizations and brands come from `GET /v1/companies/:id`; profiles for people come from `GET /v1/people/:id`. This is useful for PR crisis detection, executive reputation monitoring, and market signal tracking.

## API Endpoint

```
GET https://api.apitube.io/v1/companies/:id
GET https://api.apitube.io/v1/people/:id
```

## Authentication

All requests require an API key passed via the `api_key` query parameter:

```
https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY
https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY
```

## Parameters

| Parameter  | Type    | Description                                                            |
|------------|---------|-----------------------------------------------------------------------|
| `api_key`  | string  | **Required.** Your API key.                                           |
| `coverage` | boolean | Leave default (coverage included). The `momentum` block lives inside `coverage`. |

The path segment `:id` is the entity ID. People are entities of type `person`; companies are entities of type `organization` or `brand`. A request for an ID that is not found, or that does not match the expected type, returns `404` with error code `ER0151`.

The `momentum` block (identical shape for people and companies):

| Field              | Type    | Description                                              |
|--------------------|---------|----------------------------------------------------------|
| `last_30_days`     | integer | Article count in the trailing 30 days.                   |
| `previous_30_days` | integer | Article count in the 30 days before that.                |
| `change_pct`       | integer | Percent change between the two windows (can be negative; can be `null` when there were no articles in the previous window).|

A `null` `change_pct` means "insufficient data" — treat it as stable, not as a surge. The `first_seen` and `last_seen` fields can also be `null`, and the whole `coverage` block can be `null` when analytics are unavailable.

## Quick Start

### cURL

```bash
# A company profile (momentum is inside coverage)
curl -s "https://api.apitube.io/v1/companies/312?api_key=YOUR_API_KEY"

# A person profile
curl -s "https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

API_KEY = "YOUR_API_KEY"
THRESHOLD = 25  # percent

def momentum_for(kind, entity_id):
    response = requests.get(
        f"https://api.apitube.io/v1/{kind}/{entity_id}",
        params={"api_key": API_KEY},
    )
    response.raise_for_status()
    profile = response.json()
    return profile["name"], (profile.get("coverage") or {}).get("momentum")

name, m = momentum_for("companies", 312)
if m is None:
    print(f"{name}: no coverage")
    raise SystemExit
change = m["change_pct"] if m["change_pct"] is not None else 0
status = "SURGE" if change >= THRESHOLD else "stable"
change_str = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"
print(f"{name}: {change_str} [{status}]")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const THRESHOLD = 25;

async function momentumFor(kind, entityId) {
  const params = new URLSearchParams({ api_key: API_KEY });
  const response = await fetch(`https://api.apitube.io/v1/${kind}/${entityId}?${params}`);
  const profile = await response.json();
  return { name: profile.name, momentum: profile.coverage?.momentum };
}

const { name, momentum } = await momentumFor("companies", 312);
if (!momentum) {
  console.log(`${name}: no coverage`);
} else {
  const status = (momentum.change_pct ?? 0) >= THRESHOLD ? "SURGE" : "stable";
  const changeStr = momentum.change_pct == null ? "n/a" : `${momentum.change_pct > 0 ? "+" : ""}${momentum.change_pct}%`;
  console.log(`${name}: ${changeStr} [${status}]`);
}
```

### PHP

```php
<?php

$apiKey    = "YOUR_API_KEY";
$threshold = 25;

function momentumFor(string $kind, int $entityId): array
{
    global $apiKey;

    $query   = http_build_query(["api_key" => $apiKey]);
    $profile = json_decode(file_get_contents(
        "https://api.apitube.io/v1/{$kind}/{$entityId}?{$query}"
    ), true);

    return ["name" => $profile["name"], "momentum" => $profile["coverage"]["momentum"] ?? null];
}

["name" => $name, "momentum" => $m] = momentumFor("companies", 312);
if ($m === null) {
    printf("%s: no coverage\n", $name);
    exit;
}
$change    = $m["change_pct"] ?? 0;
$status    = $change >= $threshold ? "SURGE" : "stable";
$changeStr = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
printf("%s: %s [%s]\n", $name, $changeStr, $status);
```

## Response Example

A profile response. The momentum signal used for alerting lives at `coverage.momentum`:

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

- **Coverage spike alerts** — trigger when `momentum.change_pct` for a watched entity crosses a threshold.
- **Surge / decline classification** — bucket each entity as surge, decline, or stable from the sign and size of `change_pct`.
- **Acceleration ranking** — sort a watchlist by `change_pct` to see which entities are heating up fastest.
- **Negative surge detection** — combine a rising `momentum` with a negative-leaning `sentiment` balance to flag reputation risk.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
- [Share of Voice](../share-of-voice/) — comparing coverage volume across competitors.
- [Company Profiles](../company-profiles/) — resolving entity IDs and reading full profiles.
