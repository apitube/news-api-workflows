# Layoffs Tracker — Examples

Advanced code examples for normalised distress scoring, period-over-period sector comparison, and regional impact reporting.

---

## Python — Normalised Distress Index

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
COUNT_URL = "https://api.apitube.io/v1/news/count"

DISTRESS_EVENTS = ["layoffs", "bankruptcy"]


class DistressIndex:
    """Score companies by the share of their coverage that is distress coverage.

    A raw layoff article count ranks companies by press attention as much as by
    distress: a household name generates thousands of articles for the same cut
    that earns a regional firm three. Dividing distress coverage by the
    company's total coverage over the same window removes that bias.
    """

    def __init__(self, api_key, pause=1.2):
        self.api_key = api_key
        self.pause = pause  # stay under the per-minute request limit

    def count(self, **filters):
        params = {"api_key": self.api_key, **filters}

        try:
            payload = requests.get(COUNT_URL, params=params, timeout=45).json()
        except requests.RequestException:
            return None

        if payload.get("status") != "ok":
            return None

        time.sleep(self.pause)
        return payload.get("count", 0)

    def score(self, company, days=90, headline_only=True):
        """Distress share plus the raw components behind it."""
        key = "title" if headline_only else "organization.name"
        window = {"published_at.start": f"NOW-{days}DAYS"}

        baseline = self.count(**{key: company}, **window)
        if baseline is None:
            return None

        per_event = {}
        for event in DISTRESS_EVENTS:
            value = self.count(**{key: company, "event.type": event}, **window)
            if value is None:
                return None
            per_event[event] = value

        distress = sum(per_event.values())

        return {
            "company": company,
            "baseline_articles": baseline,
            **per_event,
            "distress_articles": distress,
            "distress_share": round(distress / baseline, 4) if baseline else 0.0,
        }

    def rank(self, companies, days=90, min_baseline=50, **kwargs):
        """Score a watchlist, dropping companies with too little coverage.

        A company with 4 total articles and 2 distress articles scores 50% —
        statistically meaningless. min_baseline filters that out.
        """
        rows = []

        for company in companies:
            row = self.score(company, days=days, **kwargs)

            if row is None:
                continue
            if row["baseline_articles"] < min_baseline:
                row["note"] = "insufficient coverage"
                rows.append(row)
                continue

            rows.append(row)

        scored = [r for r in rows if "note" not in r]
        skipped = [r for r in rows if "note" in r]

        return sorted(scored, key=lambda r: -r["distress_share"]) + skipped

    def trend(self, company, recent_days=30, prior_days=90, headline_only=True):
        """Is distress coverage accelerating relative to the company's norm?"""
        recent = self.score(company, days=recent_days, headline_only=headline_only)
        prior = self.score(company, days=prior_days, headline_only=headline_only)

        if not recent or not prior or prior["distress_share"] == 0:
            return None

        return {
            "company": company,
            "recent_share": recent["distress_share"],
            "prior_share": prior["distress_share"],
            "ratio": round(recent["distress_share"] / prior["distress_share"], 2),
        }


index = DistressIndex(API_KEY)

WATCHLIST = ["Microsoft", "Intel", "Boeing", "Nissan"]

print("Distress index (90 days, headline matches):\n")
print(f"{'Company':<14}{'total':>9}{'layoffs':>9}{'bankr':>8}{'share':>9}")
for row in index.rank(WATCHLIST, days=90):
    if row.get("note"):
        print(f"{row['company']:<14}{row['baseline_articles']:>9}  {row['note']}")
        continue

    print(f"{row['company']:<14}{row['baseline_articles']:>9}{row['layoffs']:>9}"
          f"{row['bankruptcy']:>8}{row['distress_share'] * 100:>8.2f}%")

print("\nAcceleration (30d share vs 90d share):\n")
for company in WATCHLIST:
    trend = index.trend(company)
    if not trend:
        print(f"  {company:<14} no prior distress coverage")
        continue

    direction = "rising" if trend["ratio"] > 1.15 else "falling" if trend["ratio"] < 0.85 else "flat"
    print(f"  {company:<14} {trend['recent_share'] * 100:>6.2f}% vs {trend['prior_share'] * 100:>6.2f}% "
          f"ratio={trend['ratio']:<5} {direction}")
```

---

## JavaScript — Sector Rotation Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class SectorRotation {
  constructor(apiKey, pauseMs = 1200) {
    this.apiKey = apiKey;
    this.pauseMs = pauseMs;
  }

  async count(filters) {
    const params = new URLSearchParams({ api_key: this.apiKey, ...filters });

    try {
      const payload = await (await fetch(`${COUNT_URL}?${params}`)).json();
      await sleep(this.pauseMs);
      return payload.status === "ok" ? payload.count : null;
    } catch {
      return null;
    }
  }

  // Facet buckets return numeric industry IDs. Any article in the bucket
  // carries an `industries` array with both id and name, so one small fetch
  // per id resolves the labels.
  async topIndustries({ event = "layoffs", limit = 10, days = 30 } = {}) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "event.type": event,
      "published_at.start": `NOW-${days}DAYS`,
      facet: "true",
      "facet.field": "industry.id",
      "facet.limit": String(limit),
      per_page: "1"
    });

    try {
      const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
      if (payload.status !== "ok") return [];

      return (payload.facets?.["industry.id"] || []).map((b) => ({
        industryId: b.value,
        articles: b.count
      }));
    } catch {
      return [];
    }
  }

  async resolveIndustryName(industryId) {
    const params = new URLSearchParams({
      api_key: this.apiKey,
      "industry.id": String(industryId),
      per_page: "1",
      fl: "id,industries"
    });

    try {
      const payload = await (await fetch(`${BASE_URL}?${params}`)).json();
      if (payload.status !== "ok") return String(industryId);

      const match = (payload.results?.[0]?.industries || []).find((i) => i.id === industryId);
      return match?.name || String(industryId);
    } catch {
      return String(industryId);
    }
  }

  // Compare two equal-length windows. The API has no built-in period compare
  // on /count, so run it twice with explicit bounds.
  async compareWindows({ industryId, event = "layoffs", days = 30 } = {}) {
    const now = new Date();
    const msPerDay = 86400000;

    const fmt = (date) => date.toISOString().slice(0, 10);
    const recentStart = fmt(new Date(now.getTime() - days * msPerDay));
    const priorStart = fmt(new Date(now.getTime() - 2 * days * msPerDay));

    const recent = await this.count({
      "event.type": event,
      "industry.id": String(industryId),
      "published_at.start": recentStart
    });

    const prior = await this.count({
      "event.type": event,
      "industry.id": String(industryId),
      "published_at.start": priorStart,
      "published_at.end": recentStart
    });

    if (recent === null || prior === null) return null;

    return {
      industryId,
      recent,
      prior,
      changeAbsolute: recent - prior,
      changePercent: prior > 0 ? Number((((recent - prior) / prior) * 100).toFixed(1)) : null
    };
  }

  async report({ event = "layoffs", days = 30, top = 6 } = {}) {
    const industries = await this.topIndustries({ event, days, limit: top });
    const rows = [];

    for (const industry of industries) {
      const comparison = await this.compareWindows({ industryId: industry.industryId, event, days });
      if (!comparison) continue;

      const name = await this.resolveIndustryName(industry.industryId);
      rows.push({ ...comparison, name });
    }

    return rows.sort((a, b) => (b.changePercent ?? -Infinity) - (a.changePercent ?? -Infinity));
  }
}

const monitor = new SectorRotation(API_KEY);
const rows = await monitor.report({ event: "layoffs", days: 30, top: 6 });

console.log("Layoff coverage by industry — last 30 days vs the 30 before:\n");
console.log(`${"Industry".padEnd(38)}${"recent".padStart(9)}${"prior".padStart(9)}${"change".padStart(10)}`);

for (const row of rows) {
  const change = row.changePercent === null ? "n/a" : `${row.changePercent > 0 ? "+" : ""}${row.changePercent}%`;
  console.log(
    `${row.name.slice(0, 36).padEnd(38)}${String(row.recent).padStart(9)}` +
      `${String(row.prior).padStart(9)}${change.padStart(10)}`
  );
}

const rising = rows.filter((r) => (r.changePercent ?? 0) > 20);
console.log(`\nIndustries with distress coverage up more than 20%: ${rising.length}`);
rising.forEach((r) => console.log(`  ${r.name} (+${r.changePercent}%)`));
```

---

## PHP — Regional Impact Report

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

const DISTRESS_EVENTS = ["layoffs", "bankruptcy"];

function callApi(string $url, array $params, int $timeout = 60): array
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

/**
 * source.country.* filters by where the OUTLET is based, not where the job
 * cuts happened. It is a good proxy for local trade press and a poor one for
 * international coverage of the same event — state that in any report you ship.
 */
function countryBreakdown(string $event = "layoffs", int $days = 30, int $limit = 15): array
{
    $payload = callApi(BASE_URL, [
        "event.type"         => $event,
        "published_at.start" => "NOW-{$days}DAYS",
        "facet"              => "true",
        "facet.field"        => "source.country.id",
        "facet.limit"        => $limit,
        "per_page"           => 1,
    ]);

    $buckets = $payload["facets"]["source.country.id"] ?? [];

    return array_map(
        fn($b) => ["country_id" => $b["value"], "articles" => $b["count"]],
        $buckets
    );
}

function countForCountry(string $countryCode, string $event, int $days = 30): int
{
    $payload = callApi(COUNT_URL, [
        "event.type"          => $event,
        "source.country.code" => $countryCode,
        "published_at.start"  => "NOW-{$days}DAYS",
    ], 45);

    usleep(1200000);

    return $payload["count"] ?? 0;
}

function baselineForCountry(string $countryCode, int $days = 30): int
{
    $payload = callApi(COUNT_URL, [
        "source.country.code" => $countryCode,
        "published_at.start"  => "NOW-{$days}DAYS",
    ], 45);

    usleep(1200000);

    return $payload["count"] ?? 0;
}

/**
 * Distress intensity per market: distress articles as a share of that market's
 * total output. Comparing raw counts across countries just ranks them by how
 * much press each country produces.
 */
function marketReport(array $countryCodes, int $days = 30): array
{
    $rows = [];

    foreach ($countryCodes as $code) {
        $baseline = baselineForCountry($code, $days);

        if ($baseline === 0) {
            continue;
        }

        $perEvent = [];
        foreach (DISTRESS_EVENTS as $event) {
            $perEvent[$event] = countForCountry($code, $event, $days);
        }

        $distress = array_sum($perEvent);

        $rows[] = [
            "country"       => strtoupper($code),
            "baseline"      => $baseline,
            "layoffs"       => $perEvent["layoffs"],
            "bankruptcy"    => $perEvent["bankruptcy"],
            "distress"      => $distress,
            "per_10k"       => round($distress / $baseline * 10000, 1),
        ];
    }

    usort($rows, fn($a, $b) => $b["per_10k"] <=> $a["per_10k"]);

    return $rows;
}

function breakingDistress(int $limit = 10, int $days = 3): array
{
    $payload = callApi(BASE_URL, [
        "event.type"         => implode(",", DISTRESS_EVENTS),
        "is_breaking"        => 1,
        "published_at.start" => "NOW-{$days}DAYS",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "is_duplicate"       => 0,
        "per_page"           => $limit,
        "fl"                 => "id,title,href,published_at,source.domain,source.location",
    ], 45);

    $rows = [];

    foreach ($payload["results"] ?? [] as $article) {
        $rows[] = [
            "title"        => $article["title"],
            "source"       => $article["source"]["domain"],
            "country"      => strtoupper($article["source"]["location"]["country_code"] ?? "??"),
            "published_at" => $article["published_at"],
        ];
    }

    return $rows;
}

printf("Countries producing the most layoff coverage (facet on source country):\n\n");
foreach (array_slice(countryBreakdown("layoffs", 30, 10), 0, 10) as $row) {
    printf("  country %-6s %8d articles\n", $row["country_id"], $row["articles"]);
}

printf("\nDistress intensity per market (articles per 10,000 published):\n\n");
printf("  %-10s%10s%10s%10s%12s\n", "Country", "baseline", "layoffs", "bankr", "per 10k");

foreach (marketReport(["us", "gb", "de", "in"], 30) as $row) {
    printf(
        "  %-10s%10d%10d%10d%12.1f\n",
        $row["country"],
        $row["baseline"],
        $row["layoffs"],
        $row["bankruptcy"],
        $row["per_10k"]
    );
}

printf("\nBreaking distress coverage, last 3 days:\n\n");
foreach (breakingDistress(8, 3) as $row) {
    printf("  %s [%s] %-22s %s\n", substr($row["published_at"], 0, 10), $row["country"], substr($row["source"], 0, 22), substr($row["title"], 0, 44));
}
```

---

## Notes on Behaviour

- **Coverage is not headcount.** Every number here counts articles. Press attention scales with brand recognition, not with the size of the cut. Normalise against each company's or market's baseline coverage before comparing.
- **Only some event codes carry data.** `/v1/news/event-types` returns 44 codes, but `closure`, `expansion` and `spin-off` currently match zero articles. They are accepted without error. Confirm any code with `/v1/news/count` before it becomes a dashboard panel. The populated distress codes are `layoffs` (~272k) and `bankruptcy` (~45k).
- **Mind the rate limit.** Paid plans allow 50 requests per minute. A watchlist loop that issues three counts per company hits that ceiling at roughly 17 companies. Every example above paces itself; without the pause you get partial results that look like zeros.
- **`event.type` is not returned on the article.** The classifier assigns it, but the response body has no event field. Use the filter as the signal and parse `title` and `body` for headcount and dates.
- **Facets return IDs.** `facet.field=industry.id` and `facet.field=source.country.id` produce numeric values. Resolve names from the `industries` or `source.country` object on any article in that bucket.
- **`source.country.code` describes the publisher.** It filters by where the outlet is based, not where the layoffs happened. For a national trade-press view it works well; for the location of the event itself, read `locations_mentioned` or use the geo filters.
- **The publisher country is read back from `source.location`.** The filter parameter is `source.country.code`, but in the response the value lives at `source.location.country_code` (with `country_name` beside it) — there is no `source.country` object.

## See Also

- [README.md](./README.md) — Layoffs Tracker workflow overview and quick start.
