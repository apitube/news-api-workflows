# Launch and Funding Radar

Workflow for tracking product launches, funding rounds and IPOs — event filtering, industry trend scoring, and period-over-period comparison using the [APITube News API](https://apitube.io).

## Overview

The **Launch and Funding Radar** workflow surfaces where new products, capital and public listings are landing. It filters coverage by launch and financing event types, ranks industries by trending score against their own history, compares the current window with the previous one to separate genuine acceleration from steady volume, and drills into named companies. Uses the trends endpoint's built-in `trending` and `compare` modes rather than hand-rolled period maths. Ideal for venture analysts, competitive intelligence teams, product marketers, and tech journalists.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/count
GET https://api.apitube.io/v1/news/trends
```

## Key Parameters

| Parameter          | Type    | Description                                                                          |
|--------------------|---------|--------------------------------------------------------------------------------------|
| `api_key`          | string  | **Required.** Your API key.                                                          |
| `event.type`       | string  | `product-launch`, `funding-round`, `ipo`, `merger-acquisition`. Up to 5, OR logic.    |
| `field`            | string  | Trends only. `source.id`, `category.id`, `topic.id`, `industry.id`, `entity.id`.      |
| `trending`         | boolean | Trends only. Adds `trending_score`, `trending_history` and `growth_rate`.             |
| `trending_days`    | integer | Trends only. History window, 7–30. Default `14`.                                      |
| `compare`          | boolean | Trends only. Adds `previous_count`, `change_absolute`, `change_percent`.              |
| `compare_window`   | string  | Trends only. Required with `compare`, e.g. `7DAYS`, `24HOURS`, `1WEEK`, `1m`.         |
| `sort`             | string  | Trends only. `count`, `value`, `growth_rate`, `change`, `trending_score`.             |
| `mincount`         | integer | Trends only. Minimum articles for a value to appear.                                  |
| `industry.id`      | string  | Up to 3 numeric industry IDs, OR logic.                                              |
| `organization.name`| string  | Company entity. Must match a name in the entity list exactly.                         |
| `title`            | string  | Headline keywords — the reliable fallback when a company is not in the entity list.   |
| `published_at.start` | string | Window start (ISO 8601, `YYYY-MM-DD`, or `NOW-30DAYS`).                              |
| `per_page`         | integer | Results per page. Keep at `1` for aggregation-only requests.                          |

## Event Coverage

Four event types carry launch and financing coverage. Volumes at the time of writing:

| `event.type`         | Approx. articles | What it captures                              |
|----------------------|------------------|------------------------------------------------|
| `product-launch`     | 1,012,000        | New products, models, features, releases.      |
| `merger-acquisition` | 803,000          | Deals, takeovers, acquisitions.                |
| `funding-round`      | 348,000          | Seed through late-stage rounds.                |
| `ipo`                | 56,000           | Public listings and filings.                   |

`/v1/news/event-types` lists 44 codes in total, but several business codes — `expansion`, `spin-off`, `closure` — currently return zero articles while still being accepted by the filter. Verify any additional code with `/v1/news/count` before building on it.

## Quick Start

### cURL

```bash
# All launch and financing coverage in the last 30 days
curl -s "https://api.apitube.io/v1/news/count?api_key=YOUR_API_KEY&event.type=product-launch,funding-round,ipo&published_at.start=NOW-30DAYS"

# Which industries are trending, scored against their own history
curl -s "https://api.apitube.io/v1/news/trends?api_key=YOUR_API_KEY&field=industry.id&trending=true&trending_days=14&per_page=15"

# Period-over-period comparison, sorted by percentage change
curl -s "https://api.apitube.io/v1/news/trends?api_key=YOUR_API_KEY&field=industry.id&compare=true&compare_window=7DAYS&sort=change&per_page=15"

# Launch coverage for one company
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&event.type=product-launch&title=Nvidia&published_at.start=NOW-30DAYS&per_page=20"
```

### Python

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
COUNT_URL = "https://api.apitube.io/v1/news/count"
TRENDS_URL = "https://api.apitube.io/v1/news/trends"

LAUNCH_EVENTS = ["product-launch", "funding-round", "ipo", "merger-acquisition"]


class LaunchRadar:
    """Track launches and financing by event, industry trend and company."""

    def __init__(self, api_key, pause=0.8):
        self.api_key = api_key
        self.pause = pause

    def _get(self, url, params, timeout=60):
        try:
            payload = requests.get(
                url, params={"api_key": self.api_key, **params}, timeout=timeout
            ).json()
        except (requests.RequestException, ValueError):
            return None

        time.sleep(self.pause)
        return payload if payload.get("status") == "ok" else None

    def event_split(self, days=30, **filters):
        """Volume per event type over a window."""
        out = {}

        for event in LAUNCH_EVENTS:
            payload = self._get(
                COUNT_URL,
                {"event.type": event, "published_at.start": f"NOW-{days}DAYS", **filters},
                timeout=45,
            )
            out[event] = payload.get("count", 0) if payload else 0

        return out

    def trending_industries(self, limit=15, trending_days=14, mincount=100):
        """Rank industries by how far current activity exceeds their own baseline.

        `trending_score` is the ratio of recent to historical average: above 1
        means rising, below 1 means cooling. It is the honest way to compare a
        niche sector against a huge one, because each is scored against itself.
        """
        payload = self._get(TRENDS_URL, {
            "field": "industry.id",
            "trending": "true",
            "trending_days": trending_days,
            "mincount": mincount,
            "per_page": limit,
        }, timeout=90)

        if not payload:
            return []

        rows = []
        for trend in payload.get("trends", []):
            value = trend["value"]

            rows.append({
                "industry_id": value["id"] if isinstance(value, dict) else value,
                "name": value.get("name") if isinstance(value, dict) else str(value),
                "articles": trend["count"],
                "percentage": trend.get("percentage"),
                "trending_score": trend.get("trending_score"),
                "growth_rate": trend.get("growth_rate"),
            })

        return rows

    def period_comparison(self, window="7DAYS", limit=15, sort="change", mincount=100):
        """Compare the current window against the previous one.

        `compare=true` requires `compare_window` and unlocks sorting by
        `change` or `trending_score`. Without compare those sort values are
        rejected.
        """
        payload = self._get(TRENDS_URL, {
            "field": "industry.id",
            "compare": "true",
            "compare_window": window,
            "sort": sort,
            "order": "desc",
            "mincount": mincount,
            "per_page": limit,
        }, timeout=90)

        if not payload:
            return []

        rows = []
        for trend in payload.get("trends", []):
            value = trend["value"]

            rows.append({
                "industry_id": value["id"] if isinstance(value, dict) else value,
                "name": value.get("name") if isinstance(value, dict) else str(value),
                "current": trend["count"],
                "previous": trend.get("previous_count"),
                "change_absolute": trend.get("change_absolute"),
                "change_percent": trend.get("change_percent"),
            })

        return rows

    def company_activity(self, company, days=90, headline_only=True):
        """Event breakdown for one company.

        `organization.name` requires an exact match against the entity list and
        raises ER0222 for names that are missing. `title` never fails that way,
        which makes it the safer default for arbitrary watchlists.
        """
        key = "title" if headline_only else "organization.name"
        out = {}

        for event in LAUNCH_EVENTS:
            payload = self._get(
                COUNT_URL,
                {"event.type": event, key: company, "published_at.start": f"NOW-{days}DAYS"},
                timeout=45,
            )
            out[event] = payload.get("count", 0) if payload else 0

        return {"company": company, **out, "total": sum(out.values())}

    def recent_launches(self, limit=20, days=7, **filters):
        """Newest launch and financing coverage, trimmed to feed-row fields."""
        payload = self._get(BASE_URL, {
            "event.type": "product-launch,funding-round,ipo",
            "published_at.start": f"NOW-{days}DAYS",
            "sort.by": "published_at",
            "sort.order": "desc",
            "is_duplicate": 0,
            "per_page": limit,
            "fl": "id,title,href,published_at,source.domain,industries,entities",
            **filters,
        }, timeout=60)

        if not payload:
            return []

        rows = []
        for article in payload.get("results", []):
            orgs = [e["name"] for e in (article.get("entities") or []) if e.get("type") == "organization"]

            rows.append({
                "title": article["title"],
                "url": article.get("href"),
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "orgs": orgs[:3],
                "industries": [i["name"] for i in (article.get("industries") or [])][:2],
            })

        return rows


radar = LaunchRadar(API_KEY)

print("Launch and financing volume, last 30 days:\n")
split = radar.event_split(days=30)
for event, count in sorted(split.items(), key=lambda kv: -kv[1]):
    print(f"  {event:<20} {count:>9,}")

print("\nIndustries trending above their own baseline:\n")
print(f"  {'Industry':<44}{'articles':>10}{'score':>8}{'growth/h':>10}")
for row in radar.trending_industries(limit=10)[:10]:
    score = row["trending_score"]
    growth = row["growth_rate"]
    print(f"  {(row['name'] or '')[:42]:<44}{row['articles']:>10,}"
          f"{score if score is not None else '—':>8}{round(growth) if growth else '—':>10}")

print("\nBiggest week-over-week movers:\n")
print(f"  {'Industry':<44}{'current':>10}{'previous':>10}{'change':>10}")
for row in radar.period_comparison(window="7DAYS", limit=10)[:10]:
    change = f"{row['change_percent']:+.1f}%" if row["change_percent"] is not None else "—"
    print(f"  {(row['name'] or '')[:42]:<44}{row['current']:>10,}"
          f"{(row['previous'] or 0):>10,}{change:>10}")

print("\nCompany activity (headline matches, 90 days):\n")
for company in ["Nvidia", "OpenAI", "Stripe"]:
    row = radar.company_activity(company, days=90)
    print(f"  {row['company']:<12} launch={row['product-launch']:<6} "
          f"funding={row['funding-round']:<5} ipo={row['ipo']:<4} m&a={row['merger-acquisition']:<5} "
          f"total={row['total']}")

print("\nMost recent launch and financing coverage:\n")
for row in radar.recent_launches(limit=6, days=7):
    print(f"  {row['published_at'][:10]} {row['source'][:22]:<22} {row['title'][:46]}")
    print(f"    orgs: {', '.join(row['orgs']) or '—'}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";
const TRENDS_URL = "https://api.apitube.io/v1/news/trends";

const LAUNCH_EVENTS = ["product-launch", "funding-round", "ipo", "merger-acquisition"];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class LaunchRadar {
  constructor(apiKey, pauseMs = 800) {
    this.apiKey = apiKey;
    this.pauseMs = pauseMs;
  }

  async get(url, params) {
    const query = new URLSearchParams({ api_key: this.apiKey, ...params });

    try {
      const payload = await (await fetch(`${url}?${query}`)).json();
      await sleep(this.pauseMs);
      return payload.status === "ok" ? payload : null;
    } catch {
      return null;
    }
  }

  async eventSplit({ days = 30, ...filters } = {}) {
    const out = {};

    for (const event of LAUNCH_EVENTS) {
      const payload = await this.get(COUNT_URL, {
        "event.type": event,
        "published_at.start": `NOW-${days}DAYS`,
        ...filters
      });
      out[event] = payload?.count ?? 0;
    }

    return out;
  }

  normalizeTrend(trend) {
    const value = trend.value;
    const isObject = value !== null && typeof value === "object";

    return {
      industryId: isObject ? value.id : value,
      name: isObject ? value.name : String(value),
      articles: trend.count,
      percentage: trend.percentage,
      trendingScore: trend.trending_score,
      growthRate: trend.growth_rate,
      previous: trend.previous_count,
      changeAbsolute: trend.change_absolute,
      changePercent: trend.change_percent
    };
  }

  // trending_score is the ratio of recent to historical activity, so a niche
  // industry and a huge one are each scored against themselves.
  async trendingIndustries({ limit = 15, trendingDays = 14, mincount = 100 } = {}) {
    const payload = await this.get(TRENDS_URL, {
      field: "industry.id",
      trending: "true",
      trending_days: String(trendingDays),
      mincount: String(mincount),
      per_page: String(limit)
    });

    if (!payload) return [];
    return (payload.trends || []).map((trend) => this.normalizeTrend(trend));
  }

  // compare=true requires compare_window, and only then can you sort by
  // `change` or `trending_score`.
  async periodComparison({ window = "7DAYS", limit = 15, sort = "change", mincount = 100 } = {}) {
    const payload = await this.get(TRENDS_URL, {
      field: "industry.id",
      compare: "true",
      compare_window: window,
      sort,
      order: "desc",
      mincount: String(mincount),
      per_page: String(limit)
    });

    if (!payload) return [];
    return (payload.trends || []).map((trend) => this.normalizeTrend(trend));
  }

  async companyActivity(company, { days = 90, headlineOnly = true } = {}) {
    const key = headlineOnly ? "title" : "organization.name";
    const out = {};

    for (const event of LAUNCH_EVENTS) {
      const payload = await this.get(COUNT_URL, {
        "event.type": event,
        [key]: company,
        "published_at.start": `NOW-${days}DAYS`
      });
      out[event] = payload?.count ?? 0;
    }

    return { company, ...out, total: Object.values(out).reduce((a, b) => a + b, 0) };
  }

  async recentLaunches({ limit = 20, days = 7, ...filters } = {}) {
    const payload = await this.get(BASE_URL, {
      "event.type": "product-launch,funding-round,ipo",
      "published_at.start": `NOW-${days}DAYS`,
      "sort.by": "published_at",
      "sort.order": "desc",
      is_duplicate: "0",
      per_page: String(limit),
      fl: "id,title,href,published_at,source.domain,industries,entities",
      ...filters
    });

    if (!payload) return [];

    return (payload.results || []).map((article) => ({
      title: article.title,
      url: article.href,
      source: article.source.domain,
      publishedAt: article.published_at,
      orgs: (article.entities || [])
        .filter((e) => e.type === "organization")
        .slice(0, 3)
        .map((e) => e.name),
      industries: (article.industries || []).slice(0, 2).map((i) => i.name)
    }));
  }
}

const radar = new LaunchRadar(API_KEY);

console.log("Launch and financing volume, last 30 days:\n");
const split = await radar.eventSplit({ days: 30 });
for (const [event, count] of Object.entries(split).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${event.padEnd(20)}${count.toLocaleString().padStart(10)}`);
}

console.log("\nIndustries trending above their own baseline:\n");
console.log(`  ${"Industry".padEnd(44)}${"articles".padStart(10)}${"score".padStart(8)}`);
for (const row of await radar.trendingIndustries({ limit: 10 })) {
  console.log(
    `  ${(row.name || "").slice(0, 42).padEnd(44)}${row.articles.toLocaleString().padStart(10)}` +
      `${String(row.trendingScore ?? "—").padStart(8)}`
  );
}

console.log("\nBiggest week-over-week movers:\n");
console.log(`  ${"Industry".padEnd(44)}${"current".padStart(10)}${"previous".padStart(10)}${"change".padStart(10)}`);
for (const row of await radar.periodComparison({ window: "7DAYS", limit: 10 })) {
  const change = row.changePercent === undefined || row.changePercent === null
    ? "—"
    : `${row.changePercent > 0 ? "+" : ""}${row.changePercent}%`;

  console.log(
    `  ${(row.name || "").slice(0, 42).padEnd(44)}${row.articles.toLocaleString().padStart(10)}` +
      `${(row.previous ?? 0).toLocaleString().padStart(10)}${change.padStart(10)}`
  );
}

console.log("\nCompany activity (headline matches, 90 days):\n");
for (const company of ["Nvidia", "OpenAI", "Stripe"]) {
  const row = await radar.companyActivity(company, { days: 90 });
  console.log(
    `  ${row.company.padEnd(12)} launch=${String(row["product-launch"]).padEnd(6)} ` +
      `funding=${String(row["funding-round"]).padEnd(5)} ipo=${String(row.ipo).padEnd(4)} ` +
      `m&a=${String(row["merger-acquisition"]).padEnd(5)} total=${row.total}`
  );
}
```

### PHP

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";
const TRENDS_URL = "https://api.apitube.io/v1/news/trends";

const LAUNCH_EVENTS = ["product-launch", "funding-round", "ipo", "merger-acquisition"];

function radarGet(string $url, array $params, int $timeout = 60): ?array
{
    $handle = curl_init($url . "?" . http_build_query($params + ["api_key" => API_KEY]));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, $timeout);

    $body = curl_exec($handle);
    $failed = curl_errno($handle) !== 0;
    curl_close($handle);

    usleep(800000);

    if ($failed) {
        return null;
    }

    $payload = json_decode($body, true) ?: [];

    return ($payload["status"] ?? "") === "ok" ? $payload : null;
}

function eventSplit(int $days = 30): array
{
    $out = [];

    foreach (LAUNCH_EVENTS as $event) {
        $payload = radarGet(COUNT_URL, [
            "event.type"         => $event,
            "published_at.start" => "NOW-{$days}DAYS",
        ], 45);

        $out[$event] = $payload["count"] ?? 0;
    }

    arsort($out);

    return $out;
}

function normalizeTrend(array $trend): array
{
    $value = $trend["value"];
    $isObject = is_array($value);

    return [
        "industry_id"     => $isObject ? $value["id"] : $value,
        "name"            => $isObject ? $value["name"] : (string) $value,
        "articles"        => $trend["count"],
        "trending_score"  => $trend["trending_score"] ?? null,
        "growth_rate"     => $trend["growth_rate"] ?? null,
        "previous"        => $trend["previous_count"] ?? null,
        "change_percent"  => $trend["change_percent"] ?? null,
    ];
}

/**
 * trending_score compares recent activity to that value's own history, so a
 * small industry is not buried by a large one. Above 1 means rising.
 */
function trendingIndustries(int $limit = 15, int $trendingDays = 14, int $mincount = 100): array
{
    $payload = radarGet(TRENDS_URL, [
        "field"         => "industry.id",
        "trending"      => "true",
        "trending_days" => $trendingDays,
        "mincount"      => $mincount,
        "per_page"      => $limit,
    ], 90);

    if ($payload === null) {
        return [];
    }

    return array_map("normalizeTrend", $payload["trends"] ?? []);
}

/**
 * compare=true needs compare_window; only with compare enabled can you sort
 * by `change` or `trending_score`.
 */
function periodComparison(string $window = "7DAYS", int $limit = 15, string $sort = "change"): array
{
    $payload = radarGet(TRENDS_URL, [
        "field"          => "industry.id",
        "compare"        => "true",
        "compare_window" => $window,
        "sort"           => $sort,
        "order"          => "desc",
        "mincount"       => 100,
        "per_page"       => $limit,
    ], 90);

    if ($payload === null) {
        return [];
    }

    return array_map("normalizeTrend", $payload["trends"] ?? []);
}

/**
 * organization.name demands an exact entity-list match and errors with ER0222
 * when the name is missing, so headline matching is the safer default.
 */
function companyActivity(string $company, int $days = 90, bool $headlineOnly = true): array
{
    $key = $headlineOnly ? "title" : "organization.name";
    $out = [];

    foreach (LAUNCH_EVENTS as $event) {
        $payload = radarGet(COUNT_URL, [
            "event.type"         => $event,
            $key                 => $company,
            "published_at.start" => "NOW-{$days}DAYS",
        ], 45);

        $out[$event] = $payload["count"] ?? 0;
    }

    return ["company" => $company] + $out + ["total" => array_sum($out)];
}

function recentLaunches(int $limit = 20, int $days = 7): array
{
    $payload = radarGet(BASE_URL, [
        "event.type"         => "product-launch,funding-round,ipo",
        "published_at.start" => "NOW-{$days}DAYS",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "is_duplicate"       => 0,
        "per_page"           => $limit,
        "fl"                 => "id,title,href,published_at,source.domain,industries,entities",
    ]);

    if ($payload === null) {
        return [];
    }

    $rows = [];

    foreach ($payload["results"] ?? [] as $article) {
        $orgs = [];

        foreach ($article["entities"] ?? [] as $entity) {
            if (($entity["type"] ?? "") === "organization") {
                $orgs[] = $entity["name"];
            }
        }

        $rows[] = [
            "title"        => $article["title"],
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "orgs"         => array_slice($orgs, 0, 3),
            "industries"   => array_slice(array_column($article["industries"] ?? [], "name"), 0, 2),
        ];
    }

    return $rows;
}

printf("Launch and financing volume, last 30 days:\n\n");
foreach (eventSplit(30) as $event => $count) {
    printf("  %-20s %9s\n", $event, number_format($count));
}

printf("\nIndustries trending above their own baseline:\n\n");
printf("  %-44s%10s%8s\n", "Industry", "articles", "score");
foreach (array_slice(trendingIndustries(10), 0, 10) as $row) {
    printf(
        "  %-44s%10s%8s\n",
        substr($row["name"], 0, 42),
        number_format($row["articles"]),
        $row["trending_score"] ?? "—"
    );
}

printf("\nBiggest week-over-week movers:\n\n");
printf("  %-44s%10s%10s%10s\n", "Industry", "current", "previous", "change");
foreach (array_slice(periodComparison("7DAYS", 10), 0, 10) as $row) {
    $change = $row["change_percent"] === null ? "—" : sprintf("%+.1f%%", $row["change_percent"]);

    printf(
        "  %-44s%10s%10s%10s\n",
        substr($row["name"], 0, 42),
        number_format($row["articles"]),
        number_format($row["previous"] ?? 0),
        $change
    );
}

printf("\nCompany activity (headline matches, 90 days):\n\n");
foreach (["Nvidia", "OpenAI", "Stripe"] as $company) {
    $row = companyActivity($company, 90);
    printf(
        "  %-12s launch=%-6d funding=%-5d ipo=%-4d m&a=%-5d total=%d\n",
        $row["company"],
        $row["product-launch"],
        $row["funding-round"],
        $row["ipo"],
        $row["merger-acquisition"],
        $row["total"]
    );
}

printf("\nMost recent launch and financing coverage:\n\n");
foreach (recentLaunches(6, 7) as $row) {
    printf("  %s %-22s %s\n", substr($row["published_at"], 0, 10), substr($row["source"], 0, 22), substr($row["title"], 0, 46));
    printf("    orgs: %s\n", implode(", ", $row["orgs"]) ?: "—");
}
```

## Trending Score vs. Raw Count

Ranking industries by article count always returns the same large sectors. `trending=true` fixes that by scoring each value against its own history:

| Field              | Meaning                                                                       |
|--------------------|--------------------------------------------------------------------------------|
| `trending_score`   | Recent (2-day) average divided by historical (`trending_days`) average. Above 1 rising, below 1 cooling. |
| `trending_history` | Daily counts keyed by date — plot it directly.                                 |
| `growth_rate`      | Articles per hour between first and last seen.                                 |
| `percentage`       | Share of all matching articles this value represents.                          |

`compare=true` with `compare_window` adds a second, simpler view — `previous_count`, `change_absolute` and `change_percent` against the prior period — and is the only way to unlock `sort=change` or `sort=trending_score`. Requesting those sort values without `compare=true` is rejected.

The two answer different questions. `trending_score` asks "is this unusual for this sector?"; `change_percent` asks "did this sector grow versus last week?". A sector can post a large percentage change and still score below 1 if the previous period was itself depressed.

## Common Use Cases

- **Venture deal flow** — funding coverage by industry, ranked by trending score.
- **Competitive product tracking** — launch coverage for a watchlist of rivals.
- **IPO pipeline monitoring** — listings and filings as they are reported.
- **Sector heat maps** — week-over-week change across every industry in one request.
- **Analyst briefings** — the top movers with their previous-period baselines.
- **Product marketing** — see which categories are crowded before a launch date.
- **M&A screening** — deal coverage alongside launches to spot consolidation.

## See Also

- [examples.md](./examples.md) — detailed code examples for launch and funding tracking.
