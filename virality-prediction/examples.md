# Virality Prediction — Examples

Advanced code examples for continuous velocity monitoring, per-category virality benchmarks, and separating controversy-driven spread from broad appeal.

---

## Python — Continuous Virality Monitor

```python
import requests
import time
import statistics
from datetime import datetime, timezone

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

MATURITY_HOURS = 2  # shares aggregate progressively over roughly this window


class ViralityMonitor:
    """Poll on an interval and turn share snapshots into a growth series.

    The API returns shares as a point-in-time value, so a single request cannot
    tell a story that is climbing from one that peaked yesterday. This monitor
    keeps its own history keyed by article id and differences successive polls.
    """

    def __init__(self, api_key, keep_observations=6):
        self.api_key = api_key
        self.keep = keep_observations
        self.series = {}
        self.meta = {}

    def poll(self, hours=12, limit=100, language="en", min_authority=4, **filters):
        """One collection pass. Records an observation for every article seen."""
        params = {
            "api_key": self.api_key,
            "sort.by": "engagement",
            "published_at.start": f"NOW-{hours}HOURS",
            "language.code": language,
            "source.rank.opr.min": min_authority,
            "is_duplicate": 0,
            "per_page": limit,
            "fl": "id,title,href,shares,published_at,source.domain,sentiment",
            **filters,
        }

        try:
            payload = requests.get(BASE_URL, params=params, timeout=60).json()
        except requests.RequestException:
            return 0

        if payload.get("status") != "ok":
            return 0

        stamp = time.time()

        for article in payload.get("results", []):
            article_id = article["id"]
            total = (article.get("shares") or {}).get("total", 0)

            observations = self.series.setdefault(article_id, [])
            observations.append((stamp, total))
            del observations[:-self.keep]

            self.meta[article_id] = {
                "title": article["title"],
                "url": article.get("href"),
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "sentiment": (article.get("sentiment") or {}).get("overall", {}).get("score", 0),
            }

        return len(payload.get("results", []))

    def age_hours(self, published_at):
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - published).total_seconds() / 3600, 0.01)

    def growth(self, article_id):
        """Velocity and acceleration from the recorded observations."""
        observations = self.series.get(article_id, [])

        if len(observations) < 2:
            return None

        (t0, s0), (t1, s1) = observations[-2], observations[-1]
        elapsed = (t1 - t0) / 3600

        if elapsed <= 0:
            return None

        recent_rate = (s1 - s0) / elapsed

        acceleration = None
        if len(observations) >= 3:
            (t_prev, s_prev) = observations[-3]
            prior_elapsed = (t0 - t_prev) / 3600
            if prior_elapsed > 0:
                prior_rate = (s0 - s_prev) / prior_elapsed
                acceleration = recent_rate - prior_rate

        return {
            "current_total": s1,
            "delta": s1 - s0,
            "rate_per_hour": round(recent_rate, 1),
            "acceleration": round(acceleration, 1) if acceleration is not None else None,
            "observations": len(observations),
        }

    def rising(self, min_rate=10.0):
        """Articles gaining shares faster than a floor, mature enough to trust."""
        rows = []

        for article_id, meta in self.meta.items():
            if self.age_hours(meta["published_at"]) < MATURITY_HOURS:
                continue

            growth = self.growth(article_id)
            if growth is None or growth["rate_per_hour"] < min_rate:
                continue

            rows.append({"id": article_id, **meta, **growth})

        return sorted(rows, key=lambda r: -r["rate_per_hour"])

    def cohort_stats(self):
        """Median and spread of the current lifetime rates — your baseline."""
        rates = []

        for article_id, meta in self.meta.items():
            age = self.age_hours(meta["published_at"])
            if age < MATURITY_HOURS:
                continue

            observations = self.series.get(article_id, [])
            if observations:
                rates.append(observations[-1][1] / age)

        if not rates:
            return None

        return {
            "n": len(rates),
            "median_per_hour": round(statistics.median(rates), 1),
            "p90_per_hour": round(sorted(rates)[int(len(rates) * 0.9)], 1) if len(rates) >= 10 else None,
            "max_per_hour": round(max(rates), 1),
        }


monitor = ViralityMonitor(API_KEY)

# In production this is a scheduled job. Two passes 15–30 minutes apart give
# the first real velocity reading; a shorter gap mostly measures rounding.
for pass_number in range(2):
    collected = monitor.poll(hours=12, limit=100)
    print(f"Pass {pass_number + 1}: collected {collected} articles")

    if pass_number == 0:
        time.sleep(5)  # replace with your real interval

stats = monitor.cohort_stats()
if stats:
    print(f"\nCohort: n={stats['n']} median={stats['median_per_hour']}/h max={stats['max_per_hour']}/h")

rising = monitor.rising(min_rate=1.0)
print(f"\nRising articles: {len(rising)}")

if not rising:
    # Expected with the 5-second demo gap above: share counts barely move in
    # that window, so every observed velocity rounds to zero. Space the passes
    # 15-30 minutes apart and this fills in.
    print("  (none — the demo interval is too short for shares to move; use 15-30 min)")
for row in rising[:10]:
    accel = f"accel={row['acceleration']:+.1f}" if row["acceleration"] is not None else "accel=n/a"
    print(f"  {row['rate_per_hour']:>7}/h {accel:<14} total={row['current_total']:<6} "
          f"{row['source'][:20]:<20} {row['title'][:44]}")
```

---

## JavaScript — Per-Category Virality Benchmarks

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

// A share count only means something relative to its beat, so benchmark each
// category against itself rather than against one global threshold.
//
// Use narrow categories. Top-level IDs such as medtop:15000000 (sport, ~21M
// articles) or medtop:01000000 (arts and culture, ~12M) combined with a
// composite sort and a date window are large enough to time out server-side.
const CATEGORIES = {
  climate: "medtop:20000418",
  pollution: "medtop:20000424",
  cinema: "medtop:20000005",
  environment: "medtop:06000000"
};

const MATURITY_HOURS = 2;

class CategoryBenchmarks {
  constructor(apiKey) {
    this.apiKey = apiKey;
  }

  async sample(categoryId, { hours = 24, limit = 100, language = "en" } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "category.id": categoryId,
      "sort.by": "engagement",
      "published_at.start": `NOW-${hours}HOURS`,
      "language.code": language,
      is_duplicate: "0",
      per_page: String(limit),
      fl: "id,title,href,shares,published_at,source.domain"
    });

    let payload;
    try {
      payload = await (await fetch(`${BASE_URL}?${params}`)).json();
    } catch {
      return [];
    }

    if (payload.status !== "ok") return [];

    const now = Date.now();

    return (payload.results || [])
      .map((article) => {
        const ageHours = Math.max((now - new Date(article.published_at).getTime()) / 3600000, 0.01);
        const total = article.shares?.total || 0;

        return {
          id: article.id,
          title: article.title,
          url: article.href,
          source: article.source.domain,
          total,
          ageHours,
          ratePerHour: total / ageHours
        };
      })
      .filter((row) => row.ageHours >= MATURITY_HOURS);
  }

  percentile(sorted, fraction) {
    if (sorted.length === 0) return 0;
    const index = Math.min(Math.floor(sorted.length * fraction), sorted.length - 1);
    return sorted[index];
  }

  summarize(rows) {
    const rates = rows.map((r) => r.ratePerHour).sort((a, b) => a - b);
    const totals = rows.map((r) => r.total).sort((a, b) => a - b);

    return {
      n: rows.length,
      medianRate: Number(this.percentile(rates, 0.5).toFixed(1)),
      p90Rate: Number(this.percentile(rates, 0.9).toFixed(1)),
      medianTotal: this.percentile(totals, 0.5),
      p90Total: this.percentile(totals, 0.9)
    };
  }

  // An article is "viral for its beat" when it clears its own category's p90,
  // not a global number.
  breakout(rows, benchmark) {
    return rows
      .filter((row) => row.ratePerHour > benchmark.p90Rate && benchmark.p90Rate > 0)
      .map((row) => ({
        ...row,
        multiple: Number((row.ratePerHour / Math.max(benchmark.medianRate, 0.1)).toFixed(1))
      }))
      .sort((a, b) => b.multiple - a.multiple);
  }

  async run(options = {}) {
    const report = {};

    for (const [name, categoryId] of Object.entries(CATEGORIES)) {
      const rows = await this.sample(categoryId, options);

      if (rows.length === 0) {
        report[name] = { benchmark: null, breakout: [] };
        continue;
      }

      const benchmark = this.summarize(rows);
      report[name] = { benchmark, breakout: this.breakout(rows, benchmark) };
    }

    return report;
  }
}

const benchmarks = new CategoryBenchmarks(API_KEY);
const report = await benchmarks.run({ hours: 24, limit: 100 });

console.log("Category benchmarks (mature articles only):\n");
for (const [name, data] of Object.entries(report)) {
  if (!data.benchmark) {
    console.log(`${name.padEnd(12)} no mature articles in window`);
    continue;
  }

  const b = data.benchmark;
  console.log(
    `${name.padEnd(12)} n=${String(b.n).padStart(3)} ` +
      `median=${String(b.medianRate).padStart(7)}/h p90=${String(b.p90Rate).padStart(7)}/h ` +
      `medianTotal=${String(b.medianTotal).padStart(5)}`
  );
}

console.log("\nBreakouts relative to their own beat:\n");
for (const [name, data] of Object.entries(report)) {
  data.breakout.slice(0, 3).forEach((row) => {
    console.log(`  [${name}] ${row.multiple}x median | ${row.total} shares | ${row.title.slice(0, 50)}`);
  });
}
```

---

## PHP — Engagement vs Controversy Splitter

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

/**
 * Two things spread on social platforms: broad appeal and disagreement. The
 * `engagement` mode is tuned for the first, `controversy` for the second.
 * Articles that appear in both rankings are the ones that travel furthest —
 * and the ones most likely to damage a brand that amplifies them blindly.
 */
function rankBy(string $mode, int $hours = 24, int $limit = 60, string $language = "en"): array
{
    $params = [
        "api_key"            => API_KEY,
        "sort.by"            => $mode,
        "published_at.start" => "NOW-{$hours}HOURS",
        "language.code"      => $language,
        "is_duplicate"       => 0,
        "per_page"           => $limit,
        "fl"                 => "id,title,href,shares,published_at,source.domain,sentiment",
    ];

    $handle = curl_init(BASE_URL . "?" . http_build_query($params));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, 60);

    $body = curl_exec($handle);
    curl_close($handle);

    $payload = json_decode($body, true) ?: [];

    if (($payload["status"] ?? "") !== "ok") {
        return [];
    }

    $rows = [];
    $rank = 1;

    foreach ($payload["results"] ?? [] as $article) {
        $rows[$article["id"]] = [
            "id"           => $article["id"],
            "rank"         => $rank++,
            "title"        => $article["title"],
            "url"          => $article["href"] ?? "",
            "source"       => $article["source"]["domain"],
            "shares"       => $article["shares"]["total"] ?? 0,
            "sentiment"    => $article["sentiment"]["overall"]["score"] ?? 0,
            "published_at" => $article["published_at"],
        ];
    }

    return $rows;
}

function splitAudiences(int $hours = 24, int $limit = 60): array
{
    $engaging = rankBy("engagement", $hours, $limit);
    $divisive = rankBy("controversy", $hours, $limit);

    $bothIds = array_intersect(array_keys($engaging), array_keys($divisive));

    $both = [];
    foreach ($bothIds as $id) {
        $both[] = $engaging[$id] + [
            "engagement_rank"  => $engaging[$id]["rank"],
            "controversy_rank" => $divisive[$id]["rank"],
            "combined_rank"    => $engaging[$id]["rank"] + $divisive[$id]["rank"],
        ];
    }

    usort($both, fn($a, $b) => $a["combined_rank"] <=> $b["combined_rank"]);

    $engagingOnly = array_values(array_diff_key($engaging, $divisive));
    $divisiveOnly = array_values(array_diff_key($divisive, $engaging));

    return [
        "both"          => $both,
        "engaging_only" => $engagingOnly,
        "divisive_only" => $divisiveOnly,
    ];
}

function sentimentSplit(array $rows): array
{
    $positive = 0;
    $negative = 0;
    $neutral = 0;

    foreach ($rows as $row) {
        if ($row["sentiment"] > 0.05) {
            $positive++;
        } elseif ($row["sentiment"] < -0.05) {
            $negative++;
        } else {
            $neutral++;
        }
    }

    return ["positive" => $positive, "negative" => $negative, "neutral" => $neutral];
}

$split = splitAudiences(24, 60);

printf(
    "Engagement-only: %d | Controversy-only: %d | Both: %d\n\n",
    count($split["engaging_only"]),
    count($split["divisive_only"]),
    count($split["both"])
);

printf("Sentiment mix of engagement-only: %s\n", json_encode(sentimentSplit($split["engaging_only"])));
printf("Sentiment mix of controversy-only: %s\n\n", json_encode(sentimentSplit($split["divisive_only"])));

printf("Highest combined reach (safe to amplify only after review):\n");
foreach (array_slice($split["both"], 0, 8) as $row) {
    printf(
        "  eng#%-3d ctrl#%-3d %6d shares | sent=%+.2f | %-20s %s\n",
        $row["engagement_rank"],
        $row["controversy_rank"],
        $row["shares"],
        $row["sentiment"],
        substr($row["source"], 0, 20),
        substr($row["title"], 0, 44)
    );
}

printf("\nBroad appeal without the argument:\n");
foreach (array_slice($split["engaging_only"], 0, 5) as $row) {
    printf("  %6d shares | sent=%+.2f | %-20s %s\n", $row["shares"], $row["sentiment"], substr($row["source"], 0, 20), substr($row["title"], 0, 44));
}
```

---

## Notes on Behaviour

- **`shares` is a snapshot, not a series.** There is no historical share endpoint. Any velocity number has to come from your own repeated sampling, or from dividing the current total by the article's age.
- **The first two hours are unreliable.** Share scores are aggregated progressively from publication time, so anything younger than about two hours under-reports. Every example filters on a maturity threshold before ranking.
- **`sort.by=engagement` is not "most shared".** It is a forward-looking composite of timeliness, media richness and positive sentiment. In a 20-article English sample its median `shares.total` was around 175, while ranking directly by `shares.facebook.min` produced medians above 1,100 — the latter is what already peaked.
- **`sort.by=published_at` has near-zero share signal.** In the same sample, 19 of 20 newest-first articles had zero shares. Newest-first is the wrong input for any virality ranking.
- **The share sort keys keep their `.min` suffix.** Use `shares.facebook.min`, `shares.twitter.min`, `shares.reddit.min`. The suffix is part of the key name, not a threshold, and direction comes from `sort.order`. Passing `shares.facebook` without the suffix is silently ignored and returns an unsorted page.
- **`media` is a flat array.** Each entry has `url` and `type` (`image` or `video`) — count by type rather than expecting `media.images` and `media.videos` objects. As a filter, `media.images.count` does work as a parameter.
- **Top-level categories can time out under composite sorts.** `category.id=medtop:15000000` (sport, ~21M articles) or `medtop:01000000` (arts and culture, ~12M) combined with `sort.by=engagement` and a `published_at` window returns a gateway error rather than JSON. Narrow subcategories — `medtop:20000418`, `medtop:20000424`, `medtop:20000005` — behave normally. Wrap category calls in a try/catch and fall back to a subcategory.

## See Also

- [README.md](./README.md) — Virality Prediction workflow overview and quick start.
