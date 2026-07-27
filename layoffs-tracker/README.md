# Layoffs Tracker

Workflow for tracking workforce reductions, bankruptcies and shutdowns — event filtering, industry breakdowns, and week-over-week timelines using the [APITube News API](https://apitube.io).

## Overview

The **Layoffs Tracker** workflow builds a live record of companies cutting staff, filing for bankruptcy or closing operations. It filters coverage by detected corporate event type, breaks the result set down by industry with facets, reconstructs a weekly timeline with range faceting, and tracks named companies on a watchlist. Uses aggregation endpoints rather than article fetching wherever a count will do, so a sector-wide view costs one request instead of hundreds. Ideal for labour market analysts, recruiters, business journalists, and investors tracking operational distress.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/count
```

## Key Parameters

| Parameter             | Type    | Description                                                                          |
|-----------------------|---------|--------------------------------------------------------------------------------------|
| `api_key`             | string  | **Required.** Your API key.                                                          |
| `event.type`          | string  | `layoffs`, `bankruptcy`. Up to 5 comma-separated codes, OR logic.                    |
| `ignore.event.type`   | string  | Exclude event classes, e.g. `earnings,partnership` to strip routine corporate noise. |
| `organization.name`   | string  | Match a company entity anywhere in the article.                                       |
| `title`               | string  | Require the company name in the headline. Stricter than `organization.name`.         |
| `industry.id`         | string  | Up to 3 numeric industry IDs, OR logic.                                              |
| `facet`               | boolean | Enable faceting to get counts grouped by a field.                                     |
| `facet.field`         | string  | Fields to group by, e.g. `industry.id,source.country.id`. Max 5.                      |
| `facet.limit`         | integer | Facet values returned per field.                                                      |
| `facet.range`         | boolean | Enable range faceting for timelines.                                                  |
| `facet.range.field`   | string  | `published_at` for a timeline.                                                        |
| `facet.range.gap`     | string  | `1DAY`, `1WEEK`, `1MONTH`.                                                            |
| `published_at.start`  | string  | Window start (ISO 8601, `YYYY-MM-DD`, or `NOW-30DAYS`).                               |
| `published_at.end`    | string  | Window end.                                                                           |
| `source.country.code` | string  | Filter by the outlet's country.                                                       |
| `is_breaking`         | integer | `1` keeps only urgent coverage.                                                       |
| `per_page`            | integer | Results per page. Keep at `1` for aggregation-only requests.                          |

## Quick Start

### cURL

```bash
# Everything workforce-related in the last 30 days
curl -s "https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&event.type=layoffs,bankruptcy&published_at.start=NOW-30DAYS"

# Which industries are cutting — one request, no article payloads
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&event.type=layoffs&facet=true&facet.field=industry.id&facet.limit=10&per_page=1"

# Weekly timeline of layoff coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&event.type=layoffs&facet.range=true&facet.range.field=published_at&facet.range.start=2026-07-01&facet.range.end=2026-07-27&facet.range.gap=1WEEK&per_page=1"

# One company across both distress events
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&event.type=layoffs,bankruptcy&organization.name=Microsoft&per_page=20"
```

### Python

```python
import requests
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
COUNT_URL = "https://api.apitube.io/v1/news/count"

DISTRESS_EVENTS = ["layoffs", "bankruptcy"]


class LayoffsTracker:
    """Track workforce reductions by event type, industry, company and week."""

    def __init__(self, api_key):
        self.api_key = api_key

    def count(self, **filters):
        """One number, no article payloads. The cheapest question you can ask."""
        params = {"api_key": self.api_key, **filters}

        try:
            payload = requests.get(COUNT_URL, params=params, timeout=45).json()
        except requests.RequestException:
            return 0

        return payload.get("count", 0) if payload.get("status") == "ok" else 0

    def by_event(self, days=30, **filters):
        """Split the distress signal into its components."""
        return {
            event: self.count(
                **{"event.type": event, "published_at.start": f"NOW-{days}DAYS"},
                **filters,
            )
            for event in DISTRESS_EVENTS
        }

    def by_industry(self, event="layoffs", limit=12, **filters):
        """Facet the event feed by industry. Values are numeric industry IDs."""
        params = {
            "api_key": self.api_key,
            "event.type": event,
            "facet": "true",
            "facet.field": "industry.id",
            "facet.limit": limit,
            "per_page": 1,
            **filters,
        }

        try:
            payload = requests.get(BASE_URL, params=params, timeout=60).json()
        except requests.RequestException:
            return []

        if payload.get("status") != "ok":
            return []

        buckets = (payload.get("facets") or {}).get("industry.id", [])
        return [{"industry_id": b["value"], "articles": b["count"]} for b in buckets]

    def timeline(self, event="layoffs", start=None, end=None, gap="1WEEK", **filters):
        """Range-facet publication dates into buckets you can plot directly."""
        params = {
            "api_key": self.api_key,
            "event.type": event,
            "facet.range": "true",
            "facet.range.field": "published_at",
            "facet.range.start": start,
            "facet.range.end": end,
            "facet.range.gap": gap,
            "per_page": 1,
            **filters,
        }

        try:
            payload = requests.get(BASE_URL, params=params, timeout=60).json()
        except requests.RequestException:
            return []

        if payload.get("status") != "ok":
            return []

        buckets = (payload.get("facets") or {}).get("published_at_ranges", [])
        return [
            {"start": b["range_start"], "end": b["range_end"], "articles": b["count"]}
            for b in buckets
        ]

    def watchlist(self, companies, days=90, headline_only=False):
        """Compare distress coverage across a set of companies.

        `organization.name` matches the company entity anywhere in the article,
        which catches more but also picks up passing mentions. `title` requires
        the name in the headline — stricter, and usually what you want for a
        ranked scoreboard.
        """
        key = "title" if headline_only else "organization.name"
        rows = []

        for company in companies:
            per_event = {
                event: self.count(**{
                    "event.type": event,
                    key: company,
                    "published_at.start": f"NOW-{days}DAYS",
                })
                for event in DISTRESS_EVENTS
            }

            rows.append({
                "company": company,
                **per_event,
                "total": sum(per_event.values()),
            })

        return sorted(rows, key=lambda r: -r["total"])

    def recent(self, limit=25, days=7, **filters):
        """Actual articles, newest first, trimmed to what a feed row needs."""
        params = {
            "api_key": self.api_key,
            "event.type": ",".join(DISTRESS_EVENTS),
            "published_at.start": f"NOW-{days}DAYS",
            "sort.by": "published_at",
            "sort.order": "desc",
            "is_duplicate": 0,
            "per_page": limit,
            "fl": "id,title,href,published_at,source.domain,industries",
            **filters,
        }

        try:
            payload = requests.get(BASE_URL, params=params, timeout=45).json()
        except requests.RequestException:
            return []

        if payload.get("status") != "ok":
            return []

        rows = []
        for article in payload.get("results", []):
            rows.append({
                "title": article["title"],
                "url": article.get("href"),
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "industries": [i["name"] for i in (article.get("industries") or [])][:2],
            })

        return rows


tracker = LayoffsTracker(API_KEY)

split = tracker.by_event(days=30)
print("Distress coverage, last 30 days:")
for event, count in split.items():
    print(f"  {event:<12} {count:>8}")
print(f"  {'TOTAL':<12} {sum(split.values()):>8}")

print("\nTop industries by layoff coverage:")
for row in tracker.by_industry(limit=8):
    print(f"  industry {row['industry_id']:<6} {row['articles']:>8} articles")

print("\nWeekly layoff timeline:")
for bucket in tracker.timeline(start="2026-07-01", end="2026-07-27", gap="1WEEK"):
    bar = "#" * min(int(bucket["articles"] / 1000), 50)
    print(f"  {bucket['start']} {bucket['articles']:>7} {bar}")

print("\nCompany watchlist (headline matches, 90 days):")
for row in tracker.watchlist(["Microsoft", "Intel", "Boeing"], days=90, headline_only=True):
    print(f"  {row['company']:<12} layoffs={row['layoffs']:<6} "
          f"bankruptcy={row['bankruptcy']:<5} total={row['total']}")

print("\nMost recent distress coverage:")
for row in tracker.recent(limit=6, days=7):
    industries = ", ".join(row["industries"]) or "unclassified"
    print(f"  {row['published_at'][:10]} {row['source'][:22]:<22} {row['title'][:48]}")
    print(f"    {industries}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

const DISTRESS_EVENTS = ["layoffs", "bankruptcy"];

class LayoffsTracker {
  constructor(apiKey) {
    this.apiKey = apiKey;
  }

  async count(filters = {}) {
    const params = new URLSearchParams({ api_key: this.apiKey, ...filters });

    try {
      const payload = await (await fetch(`${COUNT_URL}?${params}`)).json();
      return payload.status === "ok" ? payload.count : 0;
    } catch {
      return 0;
    }
  }

  async byEvent({ days = 30, ...filters } = {}) {
    const out = {};

    for (const event of DISTRESS_EVENTS) {
      out[event] = await this.count({
        "event.type": event,
        "published_at.start": `NOW-${days}DAYS`,
        ...filters
      });
    }

    return out;
  }

  async byIndustry({ event = "layoffs", limit = 12, ...filters } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "event.type": event,
      facet: "true",
      "facet.field": "industry.id",
      "facet.limit": String(limit),
      per_page: "1",
      ...filters
    });

    try {
      const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
      if (payload.status !== "ok") return [];

      return (payload.facets?.["industry.id"] || []).map((bucket) => ({
        industryId: bucket.value,
        articles: bucket.count
      }));
    } catch {
      return [];
    }
  }

  async timeline({ event = "layoffs", start, end, gap = "1WEEK", ...filters } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "event.type": event,
      "facet.range": "true",
      "facet.range.field": "published_at",
      "facet.range.start": start,
      "facet.range.end": end,
      "facet.range.gap": gap,
      per_page: "1",
      ...filters
    });

    try {
      const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
      if (payload.status !== "ok") return [];

      return (payload.facets?.published_at_ranges || []).map((bucket) => ({
        start: bucket.range_start,
        end: bucket.range_end,
        articles: bucket.count
      }));
    } catch {
      return [];
    }
  }

  // organization.name matches the entity anywhere in the article; title
  // requires the name in the headline. Use the headline form for scoreboards.
  async watchlist(companies, { days = 90, headlineOnly = false } = {}) {
    const key = headlineOnly ? "title" : "organization.name";
    const rows = [];

    for (const company of companies) {
      const perEvent = {};

      for (const event of DISTRESS_EVENTS) {
        perEvent[event] = await this.count({
          "event.type": event,
          [key]: company,
          "published_at.start": `NOW-${days}DAYS`
        });
      }

      rows.push({
        company,
        ...perEvent,
        total: Object.values(perEvent).reduce((a, b) => a + b, 0)
      });
    }

    return rows.sort((a, b) => b.total - a.total);
  }

  async recent({ limit = 25, days = 7, ...filters } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "event.type": DISTRESS_EVENTS.join(","),
      "published_at.start": `NOW-${days}DAYS`,
      "sort.by": "published_at",
      "sort.order": "desc",
      is_duplicate: "0",
      per_page: String(limit),
      fl: "id,title,href,published_at,source.domain,industries",
      ...filters
    });

    try {
      const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
      if (payload.status !== "ok") return [];

      return (payload.results || []).map((article) => ({
        title: article.title,
        url: article.href,
        source: article.source.domain,
        publishedAt: article.published_at,
        industries: (article.industries || []).slice(0, 2).map((i) => i.name)
      }));
    } catch {
      return [];
    }
  }
}

const tracker = new LayoffsTracker(API_KEY);

const split = await tracker.byEvent({ days: 30 });
console.log("Distress coverage, last 30 days:");
for (const [event, count] of Object.entries(split)) {
  console.log(`  ${event.padEnd(12)} ${String(count).padStart(8)}`);
}
const total = Object.values(split).reduce((a, b) => a + b, 0);
console.log(`  ${"TOTAL".padEnd(12)} ${String(total).padStart(8)}`);

console.log("\nTop industries by layoff coverage:");
for (const row of await tracker.byIndustry({ limit: 8 })) {
  console.log(`  industry ${String(row.industryId).padEnd(6)} ${String(row.articles).padStart(8)} articles`);
}

console.log("\nWeekly layoff timeline:");
for (const bucket of await tracker.timeline({ start: "2026-07-01", end: "2026-07-27", gap: "1WEEK" })) {
  const bar = "#".repeat(Math.min(Math.floor(bucket.articles / 1000), 50));
  console.log(`  ${bucket.start} ${String(bucket.articles).padStart(7)} ${bar}`);
}

console.log("\nCompany watchlist (headline matches, 90 days):");
for (const row of await tracker.watchlist(["Microsoft", "Intel", "Boeing"], { days: 90, headlineOnly: true })) {
  console.log(
    `  ${row.company.padEnd(12)} layoffs=${String(row.layoffs).padEnd(6)} ` +
      `bankruptcy=${String(row.bankruptcy).padEnd(5)} total=${row.total}`
  );
}

console.log("\nMost recent distress coverage:");
for (const row of await tracker.recent({ limit: 6, days: 7 })) {
  console.log(`  ${row.publishedAt.slice(0, 10)} ${row.source.slice(0, 22).padEnd(22)} ${row.title.slice(0, 48)}`);
  console.log(`    ${row.industries.join(", ") || "unclassified"}`);
}
```

### PHP

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

const DISTRESS_EVENTS = ["layoffs", "bankruptcy"];

function apiCall(string $url, array $params, int $timeout = 60): array
{
    $handle = curl_init($url . "?" . http_build_query($params + ["api_key" => API_KEY]));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, $timeout);

    $body = curl_exec($handle);
    $failed = curl_errno($handle) !== 0;
    curl_close($handle);

    if ($failed) {
        return [];
    }

    $payload = json_decode($body, true) ?: [];

    return ($payload["status"] ?? "") === "ok" ? $payload : [];
}

function countArticles(array $filters): int
{
    return apiCall(COUNT_URL, $filters, 45)["count"] ?? 0;
}

function byEvent(int $days = 30): array
{
    $out = [];

    foreach (DISTRESS_EVENTS as $event) {
        $out[$event] = countArticles([
            "event.type"         => $event,
            "published_at.start" => "NOW-{$days}DAYS",
        ]);
    }

    return $out;
}

function byIndustry(string $event = "layoffs", int $limit = 12): array
{
    $payload = apiCall(BASE_URL, [
        "event.type"  => $event,
        "facet"       => "true",
        "facet.field" => "industry.id",
        "facet.limit" => $limit,
        "per_page"    => 1,
    ]);

    $buckets = $payload["facets"]["industry.id"] ?? [];

    return array_map(
        fn($b) => ["industry_id" => $b["value"], "articles" => $b["count"]],
        $buckets
    );
}

function timeline(string $event, string $start, string $end, string $gap = "1WEEK"): array
{
    $payload = apiCall(BASE_URL, [
        "event.type"        => $event,
        "facet.range"       => "true",
        "facet.range.field" => "published_at",
        "facet.range.start" => $start,
        "facet.range.end"   => $end,
        "facet.range.gap"   => $gap,
        "per_page"          => 1,
    ]);

    $buckets = $payload["facets"]["published_at_ranges"] ?? [];

    return array_map(
        fn($b) => ["start" => $b["range_start"], "end" => $b["range_end"], "articles" => $b["count"]],
        $buckets
    );
}

/**
 * organization.name matches the company entity anywhere in the article;
 * title requires the name in the headline. Headline matching is stricter and
 * produces a cleaner scoreboard.
 */
function watchlist(array $companies, int $days = 90, bool $headlineOnly = false): array
{
    $key = $headlineOnly ? "title" : "organization.name";
    $rows = [];

    foreach ($companies as $company) {
        $perEvent = [];

        foreach (DISTRESS_EVENTS as $event) {
            $perEvent[$event] = countArticles([
                "event.type"         => $event,
                $key                 => $company,
                "published_at.start" => "NOW-{$days}DAYS",
            ]);
        }

        $rows[] = ["company" => $company] + $perEvent + ["total" => array_sum($perEvent)];
    }

    usort($rows, fn($a, $b) => $b["total"] <=> $a["total"]);

    return $rows;
}

function recentDistress(int $limit = 25, int $days = 7): array
{
    $payload = apiCall(BASE_URL, [
        "event.type"         => implode(",", DISTRESS_EVENTS),
        "published_at.start" => "NOW-{$days}DAYS",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "is_duplicate"       => 0,
        "per_page"           => $limit,
        "fl"                 => "id,title,href,published_at,source.domain,industries",
    ], 45);

    $rows = [];

    foreach ($payload["results"] ?? [] as $article) {
        $industries = array_slice(array_column($article["industries"] ?? [], "name"), 0, 2);

        $rows[] = [
            "title"        => $article["title"],
            "url"          => $article["href"] ?? "",
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "industries"   => $industries,
        ];
    }

    return $rows;
}

$split = byEvent(30);
printf("Distress coverage, last 30 days:\n");
foreach ($split as $event => $count) {
    printf("  %-12s %8d\n", $event, $count);
}
printf("  %-12s %8d\n", "TOTAL", array_sum($split));

printf("\nTop industries by layoff coverage:\n");
foreach (byIndustry("layoffs", 8) as $row) {
    printf("  industry %-6d %8d articles\n", $row["industry_id"], $row["articles"]);
}

printf("\nWeekly layoff timeline:\n");
foreach (timeline("layoffs", "2026-07-01", "2026-07-27", "1WEEK") as $bucket) {
    $bar = str_repeat("#", min(intdiv($bucket["articles"], 1000), 50));
    printf("  %s %7d %s\n", $bucket["start"], $bucket["articles"], $bar);
}

printf("\nCompany watchlist (headline matches, 90 days):\n");
foreach (watchlist(["Microsoft", "Intel", "Boeing"], 90, true) as $row) {
    printf(
        "  %-12s layoffs=%-6d bankruptcy=%-5d total=%d\n",
        $row["company"],
        $row["layoffs"],
        $row["bankruptcy"],
        $row["total"]
    );
}

printf("\nMost recent distress coverage:\n");
foreach (recentDistress(6, 7) as $row) {
    printf("  %s %-22s %s\n", substr($row["published_at"], 0, 10), substr($row["source"], 0, 22), substr($row["title"], 0, 48));
    printf("    %s\n", implode(", ", $row["industries"]) ?: "unclassified");
}
```

## Reading the Numbers Honestly

This workflow counts **coverage**, not people. A layoff of 12,000 at a well-known employer generates thousands of articles; an identical cut at a private regional firm may generate three. That makes the data excellent for detecting *when* and *where* distress is being reported, and unsuitable for estimating how many jobs were lost.

Three consequences worth designing around:

- **Normalise before comparing companies.** A raw article count ranks companies by press attention as much as by distress. Divide by the company's baseline coverage over the same window to get a distress *share*.
- **`event.type` is not echoed on the article.** The classifier decides which articles carry `layoffs`, but the label is not returned in the response body. Treat the filter as the signal and read `title` and `body` for specifics such as headcount.
- **Industry facets return IDs, not names.** Facet buckets give numeric `industry.id` values. Resolve them through the industry list or from the `industries` array on any article in that bucket, which carries both `id` and `name`.
- **Some valid event codes carry no coverage.** `/v1/news/event-types` lists 44 codes, but `closure`, `expansion` and `spin-off` currently return zero articles. They are accepted by the filter and produce no error — they just match nothing. Verify any new code with `/v1/news/count` before building a panel on it. `layoffs` (~272k articles) and `bankruptcy` (~45k) are the populated distress codes.

## Common Use Cases

- **Labour market monitoring** — weekly distress volume as a leading indicator.
- **Sector rotation signals** — spot which industries are cutting before it shows in filings.
- **Recruiter sourcing** — find companies shedding staff in a specific sector or country.
- **Supplier risk screening** — watch a vendor list for bankruptcy coverage.
- **Newsroom desks** — a standing layoffs feed with industry attribution.
- **Investor distress screens** — combine bankruptcy coverage with negative sentiment.
- **Regional impact reporting** — facet by `source.country.id` to see where cuts land.

## See Also

- [examples.md](./examples.md) — detailed code examples for layoffs tracking.
