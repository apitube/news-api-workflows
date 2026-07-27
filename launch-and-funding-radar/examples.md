# Launch and Funding Radar — Examples

Advanced code examples for competitive launch scoreboards, funding heat maps built from trend history, and IPO pipeline monitoring.

---

## Python — Competitive Launch Scoreboard

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
COUNT_URL = "https://api.apitube.io/v1/news/count"
BASE_URL = "https://api.apitube.io/v1/news/everything"

EVENTS = ["product-launch", "funding-round", "ipo", "merger-acquisition"]


class LaunchScoreboard:
    """Compare companies on launch and financing coverage, normalised.

    Raw event counts rank companies by how famous they are. A launch from a
    household name generates thousands of articles; the same launch from a
    challenger generates dozens. Dividing by the company's overall coverage in
    the same window turns that into a comparable intensity figure.
    """

    def __init__(self, api_key, pause=0.9):
        self.api_key = api_key
        self.pause = pause

    def count(self, **filters):
        try:
            payload = requests.get(
                COUNT_URL, params={"api_key": self.api_key, **filters}, timeout=45
            ).json()
        except (requests.RequestException, ValueError):
            return None

        time.sleep(self.pause)
        return payload.get("count", 0) if payload.get("status") == "ok" else None

    def profile(self, company, days=90):
        """Event breakdown plus intensity relative to the company's own volume."""
        window = {"title": company, "published_at.start": f"NOW-{days}DAYS"}

        baseline = self.count(**window)
        if baseline is None:
            return None

        per_event = {}
        for event in EVENTS:
            value = self.count(**window, **{"event.type": event})
            if value is None:
                return None
            per_event[event] = value

        launch_events = per_event["product-launch"] + per_event["funding-round"]

        return {
            "company": company,
            "baseline": baseline,
            **per_event,
            "launch_intensity": round(launch_events / baseline * 100, 2) if baseline else 0.0,
            "deal_intensity": round(
                (per_event["merger-acquisition"] + per_event["ipo"]) / baseline * 100, 2
            ) if baseline else 0.0,
        }

    def scoreboard(self, companies, days=90, min_baseline=100):
        rows = []
        skipped = []

        for company in companies:
            row = self.profile(company, days=days)

            if row is None:
                continue
            if row["baseline"] < min_baseline:
                skipped.append(row)
                continue

            rows.append(row)

        return sorted(rows, key=lambda r: -r["launch_intensity"]), skipped

    def headline_examples(self, company, event="product-launch", days=90, limit=3):
        """A few real headlines behind the numbers, ranked by source authority."""
        try:
            payload = requests.get(BASE_URL, params={
                "api_key": self.api_key,
                "title": company,
                "event.type": event,
                "published_at.start": f"NOW-{days}DAYS",
                "sort.by": "trust",
                "is_duplicate": 0,
                "per_page": limit,
                "fl": "id,title,href,published_at,source.domain,source.rankings",
            }, timeout=45).json()
        except (requests.RequestException, ValueError):
            return []

        time.sleep(self.pause)

        if payload.get("status") != "ok":
            return []

        return [{
            "title": a["title"],
            "domain": a["source"]["domain"],
            "authority": (a["source"].get("rankings") or {}).get("opr") or 0,
            "published_at": a["published_at"],
        } for a in payload.get("results", [])]


board = LaunchScoreboard(API_KEY)

WATCHLIST = ["Nvidia", "OpenAI", "Stripe", "Anthropic"]

rows, skipped = board.scoreboard(WATCHLIST, days=90)

print("Launch and deal intensity (share of each company's own coverage):\n")
print(f"{'Company':<14}{'total':>9}{'launch':>8}{'funding':>9}{'ipo':>7}{'m&a':>7}{'launch%':>10}{'deal%':>9}")

for row in rows:
    print(f"{row['company']:<14}{row['baseline']:>9}{row['product-launch']:>8}"
          f"{row['funding-round']:>9}{row['ipo']:>7}{row['merger-acquisition']:>7}"
          f"{row['launch_intensity']:>9.2f}%{row['deal_intensity']:>8.2f}%")

for row in skipped:
    print(f"{row['company']:<14}{row['baseline']:>9}  too little coverage to score")

if rows:
    leader = rows[0]
    print(f"\nLaunch coverage behind {leader['company']}:\n")

    for article in board.headline_examples(leader["company"], days=90, limit=3):
        print(f"  opr={article['authority']} {article['published_at'][:10]} "
              f"{article['domain'][:24]:<24} {article['title'][:48]}")
```

---

## JavaScript — Funding Heat Map from Trend History

```javascript
const API_KEY = "YOUR_API_KEY";
const TRENDS_URL = "https://api.apitube.io/v1/news/trends";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class FundingHeatMap {
  constructor(apiKey, pauseMs = 900) {
    this.apiKey = apiKey;
    this.pauseMs = pauseMs;
  }

  // The trends endpoint returns intermittent 500s (ER0183) on queries that
  // succeed moments later — in one 12-request sample only 1 came back OK.
  // Without retries this heat map renders an empty table most runs.
  async get(params, retries = 5) {
    const query = new URLSearchParams({ api_key: this.apiKey, ...params });

    for (let attempt = 0; attempt < retries; attempt++) {
      let payload = null;

      try {
        payload = await (await fetch(`${TRENDS_URL}?${query}`)).json();
      } catch {
        payload = null;
      }

      await sleep(this.pauseMs);

      if (payload?.status === "ok" && payload.trends?.length) return payload;
      if (attempt < retries - 1) await sleep(this.pauseMs * (attempt + 2));
    }

    return null;
  }

  // trending_history is a { "YYYY-MM-DD": count } object, which is everything
  // you need to draw a sparkline without a second request per value.
  async history({ field = "industry.id", limit = 12, trendingDays = 14, mincount = 100 } = {}) {
    const payload = await this.get({
      field,
      trending: "true",
      trending_days: String(trendingDays),
      mincount: String(mincount),
      per_page: String(limit)
    });

    if (!payload) return [];

    return (payload.trends || []).map((trend) => {
      const value = trend.value;
      const isObject = value !== null && typeof value === "object";
      const historyObject = trend.trending_history || {};

      const series = Object.keys(historyObject)
        .sort()
        .map((date) => ({ date, count: historyObject[date] }));

      return {
        id: isObject ? value.id : value,
        name: isObject ? value.name : String(value),
        articles: trend.count,
        trendingScore: trend.trending_score,
        growthRate: trend.growth_rate,
        series
      };
    });
  }

  sparkline(series) {
    if (series.length === 0) return "";

    const blocks = "▁▂▃▄▅▆▇█";
    const counts = series.map((point) => point.count);
    const min = Math.min(...counts);
    const max = Math.max(...counts);
    const span = max - min || 1;

    return counts
      .map((count) => blocks[Math.min(Math.floor(((count - min) / span) * (blocks.length - 1)), blocks.length - 1)])
      .join("");
  }

  // Split the series in half and compare the averages. This is a second,
  // independent read on direction that does not rely on trending_score.
  direction(series) {
    if (series.length < 4) return { label: "insufficient history", ratio: null };

    const midpoint = Math.floor(series.length / 2);
    const mean = (points) => points.reduce((sum, p) => sum + p.count, 0) / (points.length || 1);

    const firstHalf = mean(series.slice(0, midpoint));
    const secondHalf = mean(series.slice(midpoint));

    if (firstHalf === 0) return { label: "new", ratio: null };

    const ratio = secondHalf / firstHalf;
    const label = ratio > 1.15 ? "accelerating" : ratio < 0.85 ? "cooling" : "steady";

    return { label, ratio: Number(ratio.toFixed(2)) };
  }

  async report(options = {}) {
    const rows = await this.history(options);

    return rows.map((row) => ({
      ...row,
      spark: this.sparkline(row.series),
      ...this.direction(row.series)
    }));
  }
}

const heatMap = new FundingHeatMap(API_KEY);
const report = await heatMap.report({ limit: 12, trendingDays: 14 });

// Say so out loud. An empty table after retries means the endpoint is down,
// not that no industry has activity — silently printing headers with no rows
// reads like "no data" and hides an outage.
if (report.length === 0) {
  console.error("trends unavailable after retries — /v1/news/trends is returning ER0183");
  process.exit(1);
}

console.log("Industry activity, 14-day history:\n");
console.log(
  `${"Industry".padEnd(40)}${"articles".padStart(10)}${"score".padStart(8)}  ${"trend".padEnd(16)}${"dir".padEnd(14)}`
);

for (const row of report) {
  console.log(
    `${row.name.slice(0, 38).padEnd(40)}${row.articles.toLocaleString().padStart(10)}` +
      `${String(row.trendingScore ?? "—").padStart(8)}  ${row.spark.padEnd(16)}${row.label.padEnd(14)}`
  );
}

const accelerating = report.filter((r) => r.label === "accelerating");
console.log(`\nAccelerating industries: ${accelerating.length}`);
accelerating.forEach((row) => {
  console.log(`  ${row.name} — ratio ${row.ratio}, score ${row.trendingScore ?? "—"}`);
});

const cooling = report.filter((r) => r.label === "cooling");
console.log(`\nCooling industries: ${cooling.length}`);
cooling.forEach((row) => {
  console.log(`  ${row.name} — ratio ${row.ratio}, score ${row.trendingScore ?? "—"}`);
});
```

---

## PHP — IPO and Deal Pipeline Monitor

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

function pipelineGet(string $url, array $params, int $timeout = 60): ?array
{
    $handle = curl_init($url . "?" . http_build_query($params + ["api_key" => API_KEY]));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, $timeout);

    $body = curl_exec($handle);
    $failed = curl_errno($handle) !== 0;
    curl_close($handle);

    usleep(900000);

    if ($failed) {
        return null;
    }

    $payload = json_decode($body, true) ?: [];

    return ($payload["status"] ?? "") === "ok" ? $payload : null;
}

/**
 * Weekly IPO and deal volume, built by walking explicit date windows rather
 * than relative NOW- values so the buckets are reproducible across runs.
 */
function weeklyPipeline(string $event = "ipo", int $weeks = 8): array
{
    $rows = [];

    for ($index = $weeks - 1; $index >= 0; $index--) {
        $end = date("Y-m-d", strtotime("-" . ($index * 7) . " days"));
        $start = date("Y-m-d", strtotime("-" . (($index + 1) * 7) . " days"));

        $payload = pipelineGet(COUNT_URL, [
            "event.type"         => $event,
            "published_at.start" => $start,
            "published_at.end"   => $end,
        ], 45);

        $rows[] = [
            "start"    => $start,
            "end"      => $end,
            "articles" => $payload["count"] ?? 0,
        ];
    }

    return $rows;
}

function pipelineChange(array $rows): ?float
{
    $count = count($rows);

    if ($count < 4) {
        return null;
    }

    $half = intdiv($count, 2);
    $first = array_slice($rows, 0, $half);
    $second = array_slice($rows, $half);

    $meanFirst = array_sum(array_column($first, "articles")) / max(count($first), 1);
    $meanSecond = array_sum(array_column($second, "articles")) / max(count($second), 1);

    if ($meanFirst == 0.0) {
        return null;
    }

    return round(($meanSecond - $meanFirst) / $meanFirst * 100, 1);
}

/**
 * Named companies appearing in IPO coverage, counted from the organization
 * entities on each article. This is the closest the news corpus gets to a
 * pipeline list — it reflects who is being written about, not a filing feed.
 */
function pipelineCompanies(string $event = "ipo", int $days = 30, int $pages = 2): array
{
    $tally = [];

    for ($page = 1; $page <= $pages; $page++) {
        $payload = pipelineGet(BASE_URL, [
            "event.type"         => $event,
            "published_at.start" => "NOW-{$days}DAYS",
            "sort.by"            => "published_at",
            "sort.order"         => "desc",
            "is_duplicate"       => 0,
            "per_page"           => 100,
            "page"               => $page,
            "fl"                 => "id,title,published_at,entities",
        ]);

        if ($payload === null) {
            break;
        }

        foreach ($payload["results"] ?? [] as $article) {
            foreach ($article["entities"] ?? [] as $entity) {
                if (($entity["type"] ?? "") !== "organization") {
                    continue;
                }

                $name = $entity["name"];
                $tally[$name] = ($tally[$name] ?? 0) + 1;
            }
        }

        if (empty($payload["has_next_pages"])) {
            break;
        }
    }

    arsort($tally);

    return $tally;
}

$events = ["ipo", "merger-acquisition"];

foreach ($events as $event) {
    $rows = weeklyPipeline($event, 8);
    $peak = max(array_column($rows, "articles")) ?: 1;

    printf("\n%s — weekly coverage:\n\n", strtoupper($event));

    foreach ($rows as $row) {
        $bar = str_repeat("#", (int) round($row["articles"] / $peak * 40));
        printf("  %s %7d %s\n", $row["start"], $row["articles"], $bar);
    }

    $change = pipelineChange($rows);
    printf("  recent half vs earlier half: %s\n", $change === null ? "n/a" : sprintf("%+.1f%%", $change));
}

printf("\nCompanies most often named in IPO coverage (last 30 days):\n\n");

$companies = pipelineCompanies("ipo", 30, 2);
$shown = 0;

foreach ($companies as $name => $mentions) {
    if ($mentions < 2) {
        continue;
    }

    printf("  %-44s %d mentions\n", substr($name, 0, 42), $mentions);

    if (++$shown >= 15) {
        break;
    }
}
```

---

## Notes on Behaviour

- **`/v1/news/trends` fails intermittently.** The same query returns `500 ER0183` several times in a row and then succeeds — a 12-request sample of one unchanged query came back OK once. This is not parameter-dependent: `compare`, `trending` and plain count queries all show it. Retry with a growing delay, and treat an empty trends result as "retry exhausted" rather than "no data". Every example here retries.
- **`sort=change` and `sort=trending_score` need `compare=true`.** Both are rejected without it, and `compare=true` itself requires `compare_window`. `sort=count`, `sort=value` and `sort=growth_rate` work unconditionally.
- **Trends `field` accepts five values only.** `source.id`, `category.id`, `topic.id`, `industry.id`, `entity.id`. Anything else returns `ER0350` with the allowed list in the message.
- **`trend.value` is an object for resolved fields.** For `industry.id` it comes back as `{"id": 644, "name": "Printed Material"}` rather than a bare id, so handle both shapes if you switch fields.
- **`trending_history` is a date-keyed object.** Sort the keys before plotting — object key order is not guaranteed to be chronological.
- **Several business event codes are empty.** `expansion`, `spin-off` and `closure` are valid values that currently match zero articles and return no error. The populated launch and financing codes are `product-launch`, `merger-acquisition`, `funding-round` and `ipo`.
- **`organization.name` fails on missing entities.** It raises `ER0222 entity organization name not found` rather than returning zero, so a watchlist loop dies on the first unknown name. `title` degrades gracefully and is the safer default; use `organization.name` only for names you have confirmed.
- **Mind the rate limit.** Paid plans allow 50 requests per minute. A four-company scoreboard issues five counts each; the weekly pipeline issues one per week per event. Both examples pace themselves, and dropping the pause turns rate-limited responses into silent zeros.

## See Also

- [README.md](./README.md) — Launch and Funding Radar workflow overview and quick start.
