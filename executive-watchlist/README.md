# Executive Watchlist

Track media coverage and sentiment for a list of top executives using the [APITube News API](https://apitube.io).

## Overview

The **Executive Watchlist** workflow builds a stable watchlist of public figures, then polls each one to track how their coverage is moving. You first resolve each executive from a free-text name or Wikidata ID into a stable person ID via `/v1/people`, then poll `/v1/people/:id` on a schedule to watch `momentum.change_pct` (coverage spikes) and the `sentiment` balance for each person. The result is a comparison table you can render to a terminal, a report, or an alerting pipeline.

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

### Resolve — `/v1/people`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `name` | string | Partial name match used to resolve an executive. |
| `wikidata_id` | string | Filter by Wikidata ID for an exact, unambiguous match. |
| `page` | integer | Page number for pagination. |
| `per_page` | integer | Number of results per page. |

### Poll — `/v1/people/:id`

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | **Required.** Your API key. |
| `coverage` | boolean | Set to `false` to skip coverage. Leave default (`true`) for watchlist polling. |

Returns `404` (`ER0151`) if the ID is not found or is not a person.

Note: `momentum.change_pct`, `first_seen`, and `last_seen` may be `null` (no prior 30-day window or no articles), and the whole `coverage` object may be `null` when analytics are unavailable — guard for these before formatting, comparing, or sorting.

## Quick Start

### cURL

```bash
# Step 1: resolve an executive to a stable id (prefer wikidata_id for accuracy)
curl -s "https://api.apitube.io/v1/people?api_key=YOUR_API_KEY&wikidata_id=Q317521"

# Step 2: poll the profile for momentum and sentiment
curl -s "https://api.apitube.io/v1/people/5021?api_key=YOUR_API_KEY"
```

### Python

```python
import requests

person = requests.get("https://api.apitube.io/v1/people/5021", params={
    "api_key": "YOUR_API_KEY",
}).json()

m = person["coverage"]["momentum"]
change = "n/a" if m["change_pct"] is None else f"{m['change_pct']:+d}%"
print(f"{person['name']}: {m['last_30_days']} articles in 30d ({change})")
```

### JavaScript (Node.js)

```javascript
const params = new URLSearchParams({ api_key: "YOUR_API_KEY" });
const person = await (await fetch(`https://api.apitube.io/v1/people/5021?${params}`)).json();

const m = person.coverage.momentum;
const change = m.change_pct == null ? "n/a" : `${m.change_pct >= 0 ? "+" : ""}${m.change_pct}%`;
console.log(`${person.name}: ${m.last_30_days} articles in 30d (${change})`);
```

### PHP

```php
$query  = http_build_query(["api_key" => "YOUR_API_KEY"]);
$person = json_decode(file_get_contents(
    "https://api.apitube.io/v1/people/5021?{$query}"
), true);

$m      = $person["coverage"]["momentum"];
$change = $m["change_pct"] === null ? "n/a" : sprintf("%+d%%", $m["change_pct"]);
printf("%s: %d articles in 30d (%s)\n", $person["name"], $m["last_30_days"], $change);
```

## Response Example

Polling `/v1/people/5021` returns the person profile plus a `coverage` block. The watchlist reads `momentum` and `sentiment`:

```json
{
  "id": 5021,
  "name": "Elon Musk",
  "type": "person",
  "coverage": {
    "article_count": 12840,
    "sentiment": { "positive": 4200, "neutral": 6100, "negative": 2540 },
    "momentum": { "last_30_days": 920, "previous_30_days": 760, "change_pct": 21 }
  }
}
```

## Common Use Cases

- **Resolve and pin** — turn an analyst's free-text name list into stable person IDs once, then reuse them.
- **Coverage spike ranking** — rank the watchlist by `momentum.change_pct` to see who is breaking out.
- **Negative sentiment alerts** — flag any executive whose negative share of coverage crosses a threshold.
- **Daily comparison report** — render an ASCII table comparing momentum and sentiment across the list.

## See Also

- [examples.md](./examples.md) — detailed code examples in Python, JavaScript, and PHP.
