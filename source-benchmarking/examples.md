# Source Benchmarking — Code Examples

Detailed examples for comparing publishers using the APITube News API in **Python**, **JavaScript**, and **PHP**.

Because APITube has no per-source coverage endpoint, every metric is derived from **`/v1/news/count`** scoped with `source.id`: a plain count for total volume, three counts filtered by `sentiment.overall.polarity` for the sentiment split, and two date-windowed counts (`NOW-30DAYS` vs the preceding 30 days) for momentum. Resolve names into IDs first with `/v1/suggest/sources` (see the [source-directory](../source-directory/README.md) workflow).

## Python

### Benchmark volume, sentiment and momentum

```python
import requests

API_KEY = "YOUR_API_KEY"
SOURCE_IDS = [4232, 771, 5510]
COUNT_URL = "https://api.apitube.io/v1/news/count"


def count(**filters):
    response = requests.get(COUNT_URL, params={"api_key": API_KEY, **filters})
    response.raise_for_status()
    return response.json()["count"]


def benchmark(source_id):
    total = count(**{"source.id": source_id})
    sentiment = {
        p: count(**{"source.id": source_id, "sentiment.overall.polarity": p})
        for p in ("positive", "neutral", "negative")
    }
    last_30 = count(**{"source.id": source_id, "published_at.start": "NOW-30DAYS"})
    prev_30 = count(**{"source.id": source_id, "published_at.start": "NOW-60DAYS", "published_at.end": "NOW-30DAYS"})
    change_pct = None if prev_30 == 0 else round((last_30 - prev_30) / prev_30 * 100)
    return {"id": source_id, "total": total, "sentiment": sentiment,
            "last_30": last_30, "change_pct": change_pct}


rows = [benchmark(sid) for sid in SOURCE_IDS]

# Rank by total volume
for row in sorted(rows, key=lambda r: r["total"], reverse=True):
    pos_share = 0 if row["total"] == 0 else row["sentiment"]["positive"] / row["total"] * 100
    change = "n/a" if row["change_pct"] is None else f"{row['change_pct']:+d}%"
    print(f"source {row['id']:<8} total={row['total']:>10,}  pos={pos_share:5.1f}%  momentum={change}")
```

## JavaScript (Node.js)

### Benchmark volume, sentiment and momentum

```javascript
const API_KEY = "YOUR_API_KEY";
const SOURCE_IDS = [4232, 771, 5510];

async function count(filters) {
  const params = new URLSearchParams({ api_key: API_KEY, ...filters });
  const response = await fetch(`https://api.apitube.io/v1/news/count?${params}`);
  const data = await response.json();
  return data.count;
}

async function benchmark(sourceId) {
  const total = await count({ "source.id": sourceId });
  const sentiment = {};
  for (const p of ["positive", "neutral", "negative"]) {
    sentiment[p] = await count({ "source.id": sourceId, "sentiment.overall.polarity": p });
  }
  const last30 = await count({ "source.id": sourceId, "published_at.start": "NOW-30DAYS" });
  const prev30 = await count({ "source.id": sourceId, "published_at.start": "NOW-60DAYS", "published_at.end": "NOW-30DAYS" });
  const changePct = prev30 === 0 ? null : Math.round(((last30 - prev30) / prev30) * 100);
  return { id: sourceId, total, sentiment, last30, changePct };
}

const rows = [];
for (const id of SOURCE_IDS) rows.push(await benchmark(id));

rows
  .sort((a, b) => b.total - a.total)
  .forEach((row) => {
    const posShare = row.total === 0 ? 0 : (row.sentiment.positive / row.total) * 100;
    const change = row.changePct === null ? "n/a" : `${row.changePct}%`;
    console.log(`source ${row.id}  total=${row.total.toLocaleString()}  pos=${posShare.toFixed(1)}%  momentum=${change}`);
  });
```

## PHP

### Benchmark volume, sentiment and momentum

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

function benchmark(string $apiKey, int $sourceId): array {
    $total = count_articles($apiKey, ["source.id" => $sourceId]);
    $sentiment = [];
    foreach (["positive", "neutral", "negative"] as $p) {
        $sentiment[$p] = count_articles($apiKey, ["source.id" => $sourceId, "sentiment.overall.polarity" => $p]);
    }
    $last30 = count_articles($apiKey, ["source.id" => $sourceId, "published_at.start" => "NOW-30DAYS"]);
    $prev30 = count_articles($apiKey, ["source.id" => $sourceId, "published_at.start" => "NOW-60DAYS", "published_at.end" => "NOW-30DAYS"]);
    $changePct = $prev30 === 0 ? null : (int) round(($last30 - $prev30) / $prev30 * 100);
    return ["id" => $sourceId, "total" => $total, "sentiment" => $sentiment, "last30" => $last30, "change_pct" => $changePct];
}

$rows = array_map(fn ($id) => benchmark($apiKey, $id), $sourceIds);
usort($rows, fn ($a, $b) => $b["total"] <=> $a["total"]);

foreach ($rows as $row) {
    $posShare  = $row["total"] === 0 ? 0 : $row["sentiment"]["positive"] / $row["total"] * 100;
    $change    = $row["change_pct"] === null ? "n/a" : sprintf("%+d%%", $row["change_pct"]);
    printf("source %-8d total=%10s  pos=%5.1f%%  momentum=%s\n",
        $row["id"], number_format($row["total"]), $posShare, $change);
}
```

## Notes

- Each `/v1/news/count` call costs **1 point** and returns `{ "status": "ok", "count": <int>, "request_id": "..." }`.
- A full benchmark is 6 count calls per source (1 total + 3 sentiment + 2 momentum). Narrow with extra filters (`language.code`, `category.id`, …) or a `published_at` window to compare like-for-like.
- Guard against a zero previous-window count before computing a percentage change, and against a zero total before computing a sentiment share.
- `source.id` accepts up to 3 comma-separated IDs, so you can also count a small group of publishers in one call.
