# Virality Prediction

Workflow for catching stories before they peak — engagement scoring, share velocity tracking, and controversy comparison using the [APITube News API](https://apitube.io).

## Overview

The **Virality Prediction** workflow identifies articles that are likely to spread before their share counts confirm it. It ranks candidates with the `engagement` composite score, samples the same articles repeatedly to measure share velocity rather than share volume, separates positive virality from controversy-driven spread, and flags outliers whose growth rate outpaces their cohort. Uses repeated sampling because the API exposes shares as a snapshot, not a time series. Ideal for social media teams, newsroom traffic desks, content curators, and trend researchers.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/count
```

## Key Parameters

| Parameter             | Type    | Description                                                                        |
|-----------------------|---------|------------------------------------------------------------------------------------|
| `api_key`             | string  | **Required.** Your API key.                                                        |
| `sort.by`             | string  | `engagement` (viral potential), `controversy` (polarization), `shares.facebook.min` / `shares.twitter.min` / `shares.reddit.min` (raw share counts). |
| `sort.order`          | string  | `asc` or `desc`. Applies to every sort mode including the share keys.              |
| `published_at.start`  | string  | Start of the window (ISO 8601, `YYYY-MM-DD`, or `NOW-6HOURS`).                     |
| `published_at.end`    | string  | End of the window.                                                                 |
| `category.id`         | string  | Scope to a topic area, e.g. `medtop:15000000` for sport.                           |
| `language.code`       | string  | Filter by language code.                                                           |
| `source.rank.opr.min` | number  | Minimum source authority. Filters out low-quality domains before scoring.          |
| `is_duplicate`        | integer | `0` drops near-duplicate bodies so one story is not counted many times.            |
| `media.images.count`  | integer | Media presence — a strong input to the engagement score.                           |
| `fl`                  | string  | Field selection. Sampling only needs `id,title,shares,published_at,source.domain`. |
| `per_page`            | integer | Results per page.                                                                  |

## Response Fields That Matter

| Field              | Description                                                          |
|--------------------|-----------------------------------------------------------------------|
| `shares.total`     | Sum across platforms.                                                 |
| `shares.facebook`  | Facebook share count.                                                 |
| `shares.twitter`   | X/Twitter share count.                                                |
| `shares.reddit`    | Reddit share count.                                                   |
| `published_at`     | Publication timestamp — the clock your velocity is measured against.  |
| `media`            | Images and videos attached to the article.                            |
| `sentiment.overall`| `score` and `polarity`, both inputs to `engagement` and `controversy`.|

## Quick Start

### cURL

```bash
# Predicted virality — recent, media-rich, positively framed
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sort.by=engagement&language.code=en&published_at.start=NOW-6HOURS&per_page=20"

# Already viral — highest raw share counts
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sort.by=shares.facebook.min&sort.order=desc&language.code=en&per_page=20"

# Divisive rather than popular
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&sort.by=controversy&language.code=en&published_at.start=NOW-1DAY&per_page=20"
```

### Python

```python
import requests
import time
import statistics
from datetime import datetime, timezone

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

SHARE_SORT_KEYS = {
    "facebook": "shares.facebook.min",
    "twitter": "shares.twitter.min",
    "reddit": "shares.reddit.min",
}


class ViralityTracker:
    """Sample articles repeatedly to turn share snapshots into share velocity."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.history = {}

    def candidates(self, hours=6, limit=50, language="en", min_authority=4, **filters):
        """Rank likely spreaders with the engagement composite score."""
        params = {
            "api_key": self.api_key,
            "sort.by": "engagement",
            "published_at.start": f"NOW-{hours}HOURS",
            "language.code": language,
            "source.rank.opr.min": min_authority,
            "is_duplicate": 0,
            "per_page": limit,
            **filters,
        }

        payload = requests.get(BASE_URL, params=params, timeout=45).json()

        if payload.get("status") != "ok":
            return []

        return [self._shape(a) for a in payload.get("results", [])]

    def already_viral(self, platform="facebook", limit=20, language="en", **filters):
        """Rank by raw share count — what has already spread."""
        params = {
            "api_key": self.api_key,
            "sort.by": SHARE_SORT_KEYS[platform],
            "sort.order": "desc",
            "language.code": language,
            "per_page": limit,
            **filters,
        }

        payload = requests.get(BASE_URL, params=params, timeout=45).json()

        if payload.get("status") != "ok":
            return []

        return [self._shape(a) for a in payload.get("results", [])]

    def _shape(self, article):
        shares = article.get("shares") or {}
        media = article.get("media") or []

        return {
            "id": article["id"],
            "title": article["title"],
            "url": article["href"],
            "source": article["source"]["domain"],
            "published_at": article["published_at"],
            "shares": {
                "total": shares.get("total", 0),
                "facebook": shares.get("facebook", 0),
                "twitter": shares.get("twitter", 0),
                "reddit": shares.get("reddit", 0),
            },
            "sentiment": (article.get("sentiment") or {}).get("overall", {}).get("score", 0),
            "images": sum(1 for m in media if m.get("type") == "image"),
            "videos": sum(1 for m in media if m.get("type") == "video"),
        }

    def age_hours(self, article):
        """Hours since publication. Shares mature over roughly the first two."""
        published = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
        return max((datetime.now(timezone.utc) - published).total_seconds() / 3600, 0.01)

    def sample(self, articles):
        """Record one observation per article and return per-hour velocity."""
        stamp = time.time()
        rows = []

        for article in articles:
            article_id = article["id"]
            total = article["shares"]["total"]
            previous = self.history.get(article_id)

            self.history[article_id] = {"at": stamp, "total": total}

            if previous is None:
                velocity = None
            else:
                elapsed_hours = (stamp - previous["at"]) / 3600
                velocity = (total - previous["total"]) / elapsed_hours if elapsed_hours > 0 else None

            rows.append({
                **article,
                "age_hours": round(self.age_hours(article), 2),
                "shares_per_hour_lifetime": round(total / self.age_hours(article), 1),
                "shares_per_hour_observed": round(velocity, 1) if velocity is not None else None,
                "delta": total - previous["total"] if previous else None,
            })

        return rows

    def outliers(self, rows, z_threshold=1.5):
        """Flag articles whose lifetime share rate is far above their cohort."""
        rates = [r["shares_per_hour_lifetime"] for r in rows if r["age_hours"] >= 2]

        if len(rates) < 4:
            return []

        mean = statistics.mean(rates)
        spread = statistics.pstdev(rates) or 1.0

        flagged = []
        for row in rows:
            if row["age_hours"] < 2:
                continue

            z_score = (row["shares_per_hour_lifetime"] - mean) / spread
            if z_score >= z_threshold:
                flagged.append({**row, "z_score": round(z_score, 2)})

        return sorted(flagged, key=lambda r: -r["z_score"])


tracker = ViralityTracker(API_KEY)

predicted = tracker.candidates(hours=6, limit=40)
rows = tracker.sample(predicted)

print(f"Engagement candidates: {len(rows)}")
for row in rows[:8]:
    print(f"  {row['shares']['total']:>6} shares | {row['shares_per_hour_lifetime']:>7}/h | "
          f"{row['age_hours']:>5}h | {row['source'][:22]:<22} {row['title'][:44]}")

flagged = tracker.outliers(rows)
print(f"\nOutliers above the cohort: {len(flagged)}")
for row in flagged[:5]:
    print(f"  z={row['z_score']:>5} {row['shares_per_hour_lifetime']:>7}/h {row['title'][:56]}")

viral = tracker.already_viral(platform="facebook", limit=5)
print("\nAlready viral (raw share ranking):")
for article in viral:
    print(f"  {article['shares']['total']:>6} shares | {article['source'][:22]:<22} {article['title'][:44]}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const SHARE_SORT_KEYS = {
  facebook: "shares.facebook.min",
  twitter: "shares.twitter.min",
  reddit: "shares.reddit.min"
};

class ViralityTracker {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.history = new Map();
  }

  async candidates({ hours = 6, limit = 50, language = "en", minAuthority = 4, ...filters } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "sort.by": "engagement",
      "published_at.start": `NOW-${hours}HOURS`,
      "language.code": language,
      "source.rank.opr.min": String(minAuthority),
      is_duplicate: "0",
      per_page: String(limit),
      ...filters
    });

    const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
    if (payload.status !== "ok") return [];

    return (payload.results || []).map((article) => this.shape(article));
  }

  async alreadyViral({ platform = "facebook", limit = 20, language = "en", ...filters } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "sort.by": SHARE_SORT_KEYS[platform],
      "sort.order": "desc",
      "language.code": language,
      per_page: String(limit),
      ...filters
    });

    const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
    if (payload.status !== "ok") return [];

    return (payload.results || []).map((article) => this.shape(article));
  }

  shape(article) {
    const shares = article.shares || {};
    const media = article.media || [];

    return {
      id: article.id,
      title: article.title,
      url: article.href,
      source: article.source.domain,
      publishedAt: article.published_at,
      shares: {
        total: shares.total || 0,
        facebook: shares.facebook || 0,
        twitter: shares.twitter || 0,
        reddit: shares.reddit || 0
      },
      sentiment: article.sentiment?.overall?.score || 0,
      images: media.filter((m) => m.type === "image").length,
      videos: media.filter((m) => m.type === "video").length
    };
  }

  ageHours(article) {
    const published = new Date(article.publishedAt).getTime();
    return Math.max((Date.now() - published) / 3600000, 0.01);
  }

  sample(articles) {
    const stamp = Date.now();

    return articles.map((article) => {
      const previous = this.history.get(article.id);
      const total = article.shares.total;

      this.history.set(article.id, { at: stamp, total });

      let velocity = null;
      if (previous) {
        const elapsedHours = (stamp - previous.at) / 3600000;
        if (elapsedHours > 0) velocity = (total - previous.total) / elapsedHours;
      }

      const age = this.ageHours(article);

      return {
        ...article,
        ageHours: Number(age.toFixed(2)),
        sharesPerHourLifetime: Number((total / age).toFixed(1)),
        sharesPerHourObserved: velocity === null ? null : Number(velocity.toFixed(1)),
        delta: previous ? total - previous.total : null
      };
    });
  }

  outliers(rows, zThreshold = 1.5) {
    const mature = rows.filter((r) => r.ageHours >= 2);
    if (mature.length < 4) return [];

    const rates = mature.map((r) => r.sharesPerHourLifetime);
    const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
    const variance = rates.reduce((sum, r) => sum + (r - mean) ** 2, 0) / rates.length;
    const spread = Math.sqrt(variance) || 1;

    return mature
      .map((row) => ({ ...row, zScore: Number(((row.sharesPerHourLifetime - mean) / spread).toFixed(2)) }))
      .filter((row) => row.zScore >= zThreshold)
      .sort((a, b) => b.zScore - a.zScore);
  }
}

const tracker = new ViralityTracker(API_KEY);

const predicted = await tracker.candidates({ hours: 6, limit: 40 });
const rows = tracker.sample(predicted);

console.log(`Engagement candidates: ${rows.length}`);
rows.slice(0, 8).forEach((row) => {
  console.log(
    `  ${String(row.shares.total).padStart(6)} shares | ` +
      `${String(row.sharesPerHourLifetime).padStart(7)}/h | ` +
      `${String(row.ageHours).padStart(5)}h | ${row.source.slice(0, 22).padEnd(22)} ${row.title.slice(0, 44)}`
  );
});

const flagged = tracker.outliers(rows);
console.log(`\nOutliers above the cohort: ${flagged.length}`);
flagged.slice(0, 5).forEach((row) => {
  console.log(`  z=${String(row.zScore).padStart(5)} ${String(row.sharesPerHourLifetime).padStart(7)}/h ${row.title.slice(0, 56)}`);
});

const viral = await tracker.alreadyViral({ platform: "facebook", limit: 5 });
console.log("\nAlready viral (raw share ranking):");
viral.forEach((article) => {
  console.log(`  ${String(article.shares.total).padStart(6)} shares | ${article.source.slice(0, 22).padEnd(22)} ${article.title.slice(0, 44)}`);
});
```

### PHP

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const SHARE_SORT_KEYS = [
    "facebook" => "shares.facebook.min",
    "twitter"  => "shares.twitter.min",
    "reddit"   => "shares.reddit.min",
];

function fetchArticles(array $params): array
{
    $handle = curl_init(BASE_URL . "?" . http_build_query($params + ["api_key" => API_KEY]));
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_TIMEOUT, 45);

    $body = curl_exec($handle);
    curl_close($handle);

    $payload = json_decode($body, true) ?: [];

    if (($payload["status"] ?? "") !== "ok") {
        return [];
    }

    return array_map("shapeArticle", $payload["results"] ?? []);
}

function shapeArticle(array $article): array
{
    $shares = $article["shares"] ?? [];
    $media = $article["media"] ?? [];
    $imageCount = count(array_filter($media, fn($m) => ($m["type"] ?? "") === "image"));
    $videoCount = count(array_filter($media, fn($m) => ($m["type"] ?? "") === "video"));

    return [
        "id"           => $article["id"],
        "title"        => $article["title"],
        "url"          => $article["href"],
        "source"       => $article["source"]["domain"],
        "published_at" => $article["published_at"],
        "shares"       => [
            "total"    => $shares["total"] ?? 0,
            "facebook" => $shares["facebook"] ?? 0,
            "twitter"  => $shares["twitter"] ?? 0,
            "reddit"   => $shares["reddit"] ?? 0,
        ],
        "sentiment"    => $article["sentiment"]["overall"]["score"] ?? 0,
        "images"       => $imageCount,
        "videos"       => $videoCount,
    ];
}

function ageHours(array $article): float
{
    $published = strtotime($article["published_at"]);

    return max((time() - $published) / 3600, 0.01);
}

function engagementCandidates(int $hours = 6, int $limit = 50, string $language = "en"): array
{
    return fetchArticles([
        "sort.by"             => "engagement",
        "published_at.start"  => "NOW-{$hours}HOURS",
        "language.code"       => $language,
        "source.rank.opr.min" => 4,
        "is_duplicate"        => 0,
        "per_page"            => $limit,
    ]);
}

function alreadyViral(string $platform = "facebook", int $limit = 20, string $language = "en"): array
{
    return fetchArticles([
        "sort.by"       => SHARE_SORT_KEYS[$platform],
        "sort.order"    => "desc",
        "language.code" => $language,
        "per_page"      => $limit,
    ]);
}

function withVelocity(array $articles): array
{
    $rows = [];

    foreach ($articles as $article) {
        $age = ageHours($article);

        $rows[] = $article + [
            "age_hours"                => round($age, 2),
            "shares_per_hour_lifetime" => round($article["shares"]["total"] / $age, 1),
        ];
    }

    return $rows;
}

function outliers(array $rows, float $zThreshold = 1.5): array
{
    $mature = array_values(array_filter($rows, fn($r) => $r["age_hours"] >= 2));

    if (count($mature) < 4) {
        return [];
    }

    $rates = array_column($mature, "shares_per_hour_lifetime");
    $mean = array_sum($rates) / count($rates);

    $variance = 0.0;
    foreach ($rates as $rate) {
        $variance += ($rate - $mean) ** 2;
    }
    $spread = sqrt($variance / count($rates)) ?: 1.0;

    $flagged = [];
    foreach ($mature as $row) {
        $z = ($row["shares_per_hour_lifetime"] - $mean) / $spread;

        if ($z >= $zThreshold) {
            $flagged[] = $row + ["z_score" => round($z, 2)];
        }
    }

    usort($flagged, fn($a, $b) => $b["z_score"] <=> $a["z_score"]);

    return $flagged;
}

$rows = withVelocity(engagementCandidates(6, 40));

printf("Engagement candidates: %d\n", count($rows));
foreach (array_slice($rows, 0, 8) as $row) {
    printf(
        "  %6d shares | %7.1f/h | %5.2fh | %-22s %s\n",
        $row["shares"]["total"],
        $row["shares_per_hour_lifetime"],
        $row["age_hours"],
        substr($row["source"], 0, 22),
        substr($row["title"], 0, 44)
    );
}

$flagged = outliers($rows);
printf("\nOutliers above the cohort: %d\n", count($flagged));
foreach (array_slice($flagged, 0, 5) as $row) {
    printf("  z=%5.2f %7.1f/h %s\n", $row["z_score"], $row["shares_per_hour_lifetime"], substr($row["title"], 0, 56));
}

printf("\nAlready viral (raw share ranking):\n");
foreach (alreadyViral("facebook", 5) as $article) {
    printf("  %6d shares | %-22s %s\n", $article["shares"]["total"], substr($article["source"], 0, 22), substr($article["title"], 0, 44));
}
```

## Prediction vs. Measurement

The three ranking strategies answer different questions, and mixing them up is the most common mistake in this workflow. Measured over a 20-article sample of English coverage:

| Ranking                     | Median `shares.total` | What it surfaces                                    |
|-----------------------------|-----------------------|------------------------------------------------------|
| `sort.by=published_at`      | 0                     | Newest articles — shares have not accumulated yet.   |
| `sort.by=engagement`        | ~175                  | Articles with viral characteristics, mid-lifecycle.  |
| `sort.by=shares.facebook.min` | ~1150–2250          | Articles that already spread. Acting on them is late.|

`engagement` is a **predictor** built from timeliness, media richness and positive sentiment — not a measurement of spread. That is exactly why it is useful here: the highest-share ranking tells you what already won, and the newest-first ranking tells you what has no signal yet. Engagement sits between them.

Because shares are a snapshot rather than a time series, velocity has to be computed by sampling. Two approaches, both in the code above:

- **Lifetime rate** — `shares.total / hours_since_publication`. Available from a single request, but biased against articles under two hours old, since share counts are aggregated progressively over roughly that window.
- **Observed rate** — poll the same article ids on an interval and difference the totals. Accurate, but needs state and at least two passes.

Filter out articles younger than two hours before ranking on either rate, or fresh articles will always look flat.

## Common Use Cases

- **Social desk triage** — surface stories worth amplifying before the peak.
- **Trending sections** — rank a homepage module by predicted rather than past engagement.
- **Newsletter curation** — pick stories that are rising, not stories that already saturated.
- **Competitive content analysis** — see which publishers consistently produce high-velocity articles.
- **Topic-level virality** — compare median share rates across categories to find spreadable beats.
- **Controversy separation** — split divisive traction from broadly positive traction.
- **Outlier alerting** — notify when an article's share rate exceeds its cohort by a z-score threshold.

## See Also

- [examples.md](./examples.md) — detailed code examples for virality prediction.
