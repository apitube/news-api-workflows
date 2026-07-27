# Sports Intelligence

Workflow for tracking sports coverage by discipline — transfer rumours, injury reports, fan sentiment, and per-sport engagement benchmarks using the [APITube News API](https://apitube.io).

## Overview

The **Sports Intelligence** workflow builds per-discipline news feeds from the IPTC sport taxonomy, tracks named athletes and clubs through the entity system, separates transfer and injury coverage with headline filters, and benchmarks engagement within each sport rather than across all news. Uses sport subcategories instead of the top-level sport category, which is large enough to time out under composite sorts. Ideal for sports media teams, fantasy and betting products, club communications staff, and fan-engagement platforms.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/count
GET https://api.apitube.io/v1/people
```

## Key Parameters

| Parameter                    | Type    | Description                                                                      |
|------------------------------|---------|----------------------------------------------------------------------------------|
| `api_key`                    | string  | **Required.** Your API key.                                                      |
| `category.id`                | string  | Sport discipline. Use a subcategory, not the top-level `medtop:15000000`.        |
| `person.name`                | string  | Athlete or manager by extracted person entity name.                              |
| `entity.id`                  | integer | Resolved entity id — unambiguous where names collide.                            |
| `organization.name`          | string  | Club, league or federation.                                                      |
| `title`                      | string  | Headline keywords. Multiple words are AND-matched regardless of order.           |
| `sort.by`                    | string  | `engagement` for viral potential, `controversy` for divisive coverage.           |
| `sentiment.overall.polarity` | string  | `positive`, `neutral`, `negative`.                                               |
| `published_at.start`         | string  | Window start (ISO 8601, `YYYY-MM-DD`, or `NOW-24HOURS`).                         |
| `language.code`              | string  | Filter by language code.                                                         |
| `source.country.code`        | string  | Filter by the outlet's country — useful for national fan press.                  |
| `facet`                      | boolean | Enable faceting for source, country and sentiment breakdowns.                    |
| `is_duplicate`               | integer | `0` collapses syndicated re-runs of the same match report.                        |
| `per_page`                   | integer | Results per page.                                                                |

## Discipline Category IDs

The sport branch is deep. These are the high-volume disciplines, with corpus sizes measured at the time of writing:

| Discipline  | `category.id`       | Approx. articles |
|-------------|---------------------|------------------|
| Baseball    | `medtop:20000849`   | 501,000          |
| Golf        | `medtop:20000940`   | 467,000          |
| Football    | `medtop:20001065`   | 302,000          |
| Ice hockey  | `medtop:20000965`   | 269,000          |
| Basketball  | `medtop:20000851`   | 150,000          |
| Cricket     | `medtop:20000888`   | 17,000           |
| Tennis      | `medtop:20001085`   | 16,000           |
| Boxing      | `medtop:20000856`   | 16,000           |

There is also `medtop:20000822` (competition discipline) as an intermediate level. Avoid the top-level `medtop:15000000` — at roughly 21 million articles it times out when combined with a composite sort and a date window.

> Note: there is a `sport.name` filter in the parameter reference, but it currently rejects every value with `ER0250 entity sport name not found`. Scope by `category.id` instead.

## Quick Start

### cURL

```bash
# Most shareable football coverage in the last day
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&category.id=medtop:20001065&sort.by=engagement&language.code=en&published_at.start=NOW-24HOURS&per_page=20"

# Transfer rumours — headline keywords inside the discipline
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&category.id=medtop:20001065&title=transfer&published_at.start=NOW-7DAYS&per_page=20"

# Negative coverage of one athlete
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&person.name=Lionel%20Messi&sentiment.overall.polarity=negative&per_page=20"

# Resolve an athlete to a stable entity id
curl -s "https://api.apitube.io/v1/people?api_key=YOUR_API_KEY&name=Lionel%20Messi"
```

### Python

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
COUNT_URL = "https://api.apitube.io/v1/news/count"
PEOPLE_URL = "https://api.apitube.io/v1/people"

DISCIPLINES = {
    "football": "medtop:20001065",
    "basketball": "medtop:20000851",
    "ice_hockey": "medtop:20000965",
    "tennis": "medtop:20001085",
    "cricket": "medtop:20000888",
}

# Headline vocabularies. `title` is AND-matched, so keep each pattern to a
# single word or a phrase that really appears in headlines together.
STORY_PATTERNS = {
    "transfer": ["transfer", "signing", "loan"],
    "injury": ["injury", "injured", "sidelined"],
    "contract": ["contract", "extension"],
    "manager": ["sacked", "appointed"],
}


class SportsIntelligence:
    """Per-discipline sports news with entity resolution and story typing."""

    def __init__(self, api_key, pause=0.8):
        self.api_key = api_key
        self.pause = pause

    def _get(self, url, params, timeout=60):
        params = {"api_key": self.api_key, **params}

        try:
            payload = requests.get(url, params=params, timeout=timeout).json()
        except (requests.RequestException, ValueError):
            return None

        time.sleep(self.pause)

        # /v1/people/{id} returns the profile with no `status` field, unlike
        # the search and news endpoints. Accept either shape.
        if payload.get("status") == "ok" or "id" in payload:
            return payload

        return None

    def resolve_athlete(self, name):
        """Turn a name into a stable entity id plus its coverage summary.

        Two calls are required: the directory search returns id and profile
        only, while the `coverage` block lives on the individual profile.
        """
        listing = self._get(PEOPLE_URL, {"name": name})

        if not listing or not listing.get("results"):
            return None

        person_id = listing["results"][0]["id"]
        payload = self._get(f"{PEOPLE_URL}/{person_id}", {})

        if not payload:
            return None

        person = payload
        coverage = person.get("coverage") or {}

        return {
            "id": person["id"],
            "name": person["name"],
            "article_count": coverage.get("article_count"),
            "sentiment": coverage.get("sentiment"),
            "momentum": coverage.get("momentum"),
            "top_sources": [s.get("name") or s.get("domain") for s in (coverage.get("top_sources") or [])[:3]],
        }

    def discipline_feed(self, discipline, hours=24, limit=25, mode="engagement", language="en"):
        """Top coverage for one sport, ranked by a composite score."""
        payload = self._get(BASE_URL, {
            "category.id": DISCIPLINES[discipline],
            "sort.by": mode,
            "published_at.start": f"NOW-{hours}HOURS",
            "language.code": language,
            "is_duplicate": 0,
            "per_page": limit,
        })

        if not payload:
            return []

        return [self._shape(a) for a in payload.get("results", [])]

    def _shape(self, article):
        shares = article.get("shares") or {}
        sentiment = (article.get("sentiment") or {}).get("overall", {})

        return {
            "id": article["id"],
            "title": article["title"],
            "url": article.get("href"),
            "source": article["source"]["domain"],
            "country": (article["source"].get("location") or {}).get("country_code", "??"),
            "published_at": article["published_at"],
            "shares": shares.get("total", 0),
            "sentiment_score": sentiment.get("score", 0),
            "sentiment": sentiment.get("polarity", "neutral"),
            "people": [e["name"] for e in (article.get("entities") or []) if e.get("type") == "person"][:4],
        }

    def story_mix(self, discipline, days=7):
        """How much of a discipline's coverage is transfers, injuries, contracts."""
        category = DISCIPLINES[discipline]
        window = {"published_at.start": f"NOW-{days}DAYS"}

        total_payload = self._get(COUNT_URL, {"category.id": category, **window}, timeout=45)
        total = total_payload.get("count", 0) if total_payload else 0

        mix = {}
        for story_type, keywords in STORY_PATTERNS.items():
            best = 0
            for keyword in keywords:
                payload = self._get(
                    COUNT_URL,
                    {"category.id": category, "title": keyword, **window},
                    timeout=45,
                )
                best = max(best, payload.get("count", 0) if payload else 0)

            mix[story_type] = {
                "articles": best,
                "share": round(best / total, 4) if total else 0.0,
            }

        return {"discipline": discipline, "total": total, "mix": mix}

    def athlete_feed(self, entity_id, limit=20, mode="engagement"):
        """Coverage of one athlete by resolved entity id."""
        payload = self._get(BASE_URL, {
            "entity.id": entity_id,
            "sort.by": mode,
            "is_duplicate": 0,
            "per_page": limit,
        })

        if not payload:
            return []

        return [self._shape(a) for a in payload.get("results", [])]

    def sentiment_split(self, discipline, days=7):
        """Tone breakdown for a discipline, as shares rather than raw counts."""
        category = DISCIPLINES[discipline]
        window = {"published_at.start": f"NOW-{days}DAYS"}

        total_payload = self._get(COUNT_URL, {"category.id": category, **window}, timeout=45)
        total = total_payload.get("count", 0) if total_payload else 0

        out = {}
        for polarity in ("positive", "negative", "neutral"):
            payload = self._get(
                COUNT_URL,
                {"category.id": category, "sentiment.overall.polarity": polarity, **window},
                timeout=45,
            )
            count = payload.get("count", 0) if payload else 0
            out[polarity] = {"articles": count, "share": round(count / total, 3) if total else 0.0}

        return out


sports = SportsIntelligence(API_KEY)

print("Top football coverage by engagement:\n")
for row in sports.discipline_feed("football", hours=24, limit=8):
    people = ", ".join(row["people"]) or "—"
    print(f"  {row['shares']:>6} shares [{row['country']}] {row['source'][:22]:<22} {row['title'][:44]}")
    print(f"    people: {people[:70]}")

print("\nStory mix for football, last 7 days:\n")
mix = sports.story_mix("football", days=7)
print(f"  total articles: {mix['total']:,}")
for story_type, data in mix["mix"].items():
    print(f"    {story_type:<10} {data['articles']:>7} ({data['share'] * 100:.2f}%)")

print("\nSentiment split for football:\n")
for polarity, data in sports.sentiment_split("football", days=7).items():
    print(f"    {polarity:<9} {data['articles']:>7} ({data['share'] * 100:.1f}%)")

athlete = sports.resolve_athlete("Lionel Messi")
if athlete:
    print(f"\nResolved: {athlete['name']} (id {athlete['id']}) — {athlete['article_count']} articles")
    print(f"  sentiment: {athlete['sentiment']}")
    print(f"  top sources: {', '.join(str(s) for s in athlete['top_sources'])}")

    print("\n  Most shareable coverage:")
    for row in sports.athlete_feed(athlete["id"], limit=5):
        print(f"    {row['shares']:>6} shares | {row['sentiment']:<8} | {row['title'][:52]}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";
const PEOPLE_URL = "https://api.apitube.io/v1/people";

const DISCIPLINES = {
  football: "medtop:20001065",
  basketball: "medtop:20000851",
  iceHockey: "medtop:20000965",
  tennis: "medtop:20001085",
  cricket: "medtop:20000888"
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class SportsIntelligence {
  constructor(apiKey, pauseMs = 800) {
    this.apiKey = apiKey;
    this.pauseMs = pauseMs;
  }

  async get(url, params) {
    const query = new URLSearchParams({ api_key: this.apiKey, ...params });

    try {
      const payload = await (await fetch(`${url}?${query}`)).json();
      await sleep(this.pauseMs);

      // /v1/people/{id} returns the profile without a `status` field, unlike
      // the search and news endpoints. Accept either shape.
      if (payload.status === "ok" || payload.id !== undefined) return payload;

      return null;
    } catch {
      return null;
    }
  }

  // Two calls: the directory returns id and profile, while the coverage block
  // is only present on the individual profile endpoint.
  async resolveAthlete(name) {
    const listing = await this.get(PEOPLE_URL, { name });
    if (!listing?.results?.length) return null;

    const person = await this.get(`${PEOPLE_URL}/${listing.results[0].id}`, {});
    if (!person) return null;

    const coverage = person.coverage || {};

    return {
      id: person.id,
      name: person.name,
      articleCount: coverage.article_count,
      sentiment: coverage.sentiment,
      momentum: coverage.momentum,
      topSources: (coverage.top_sources || []).slice(0, 3).map((s) => s.name || s.domain)
    };
  }

  shape(article) {
    const shares = article.shares || {};
    const sentiment = article.sentiment?.overall || {};

    return {
      id: article.id,
      title: article.title,
      url: article.href,
      source: article.source.domain,
      country: article.source.location?.country_code || "??",
      publishedAt: article.published_at,
      shares: shares.total || 0,
      sentimentScore: sentiment.score || 0,
      sentiment: sentiment.polarity || "neutral",
      people: (article.entities || [])
        .filter((e) => e.type === "person")
        .slice(0, 4)
        .map((e) => e.name)
    };
  }

  async disciplineFeed(discipline, { hours = 24, limit = 25, mode = "engagement", language = "en" } = {}) {
    const payload = await this.get(BASE_URL, {
      "category.id": DISCIPLINES[discipline],
      "sort.by": mode,
      "published_at.start": `NOW-${hours}HOURS`,
      "language.code": language,
      is_duplicate: "0",
      per_page: String(limit)
    });

    if (!payload) return [];
    return (payload.results || []).map((article) => this.shape(article));
  }

  // Benchmark within a sport, never across sports: baseball and golf carry
  // an order of magnitude more coverage than tennis or boxing, so a global
  // share threshold would only ever surface the big two.
  async engagementBenchmark(discipline, { hours = 24, limit = 60 } = {}) {
    const rows = await this.disciplineFeed(discipline, { hours, limit });
    if (rows.length === 0) return null;

    const shares = rows.map((r) => r.shares).sort((a, b) => a - b);
    const at = (fraction) => shares[Math.min(Math.floor(shares.length * fraction), shares.length - 1)];

    return {
      discipline,
      n: rows.length,
      median: at(0.5),
      p90: at(0.9),
      max: shares[shares.length - 1],
      breakouts: rows.filter((r) => r.shares > at(0.9)).sort((a, b) => b.shares - a.shares)
    };
  }

  async compareDisciplines({ hours = 24, limit = 60 } = {}) {
    const report = [];

    for (const discipline of Object.keys(DISCIPLINES)) {
      const benchmark = await this.engagementBenchmark(discipline, { hours, limit });
      if (benchmark) report.push(benchmark);
    }

    return report.sort((a, b) => b.median - a.median);
  }
}

const sports = new SportsIntelligence(API_KEY);

console.log("Engagement benchmarks by discipline (last 24h):\n");
console.log(`${"Discipline".padEnd(14)}${"n".padStart(5)}${"median".padStart(9)}${"p90".padStart(9)}${"max".padStart(9)}`);

const report = await sports.compareDisciplines({ hours: 24, limit: 60 });
for (const row of report) {
  console.log(
    `${row.discipline.padEnd(14)}${String(row.n).padStart(5)}${String(row.median).padStart(9)}` +
      `${String(row.p90).padStart(9)}${String(row.max).padStart(9)}`
  );
}

console.log("\nBreakout stories within each discipline:\n");
for (const row of report) {
  row.breakouts.slice(0, 2).forEach((story) => {
    console.log(`  [${row.discipline}] ${story.shares} shares | ${story.title.slice(0, 54)}`);
  });
}

const athlete = await sports.resolveAthlete("Lionel Messi");
if (athlete) {
  console.log(`\nResolved: ${athlete.name} (id ${athlete.id}) — ${athlete.articleCount} articles`);
  console.log(`  sentiment: ${JSON.stringify(athlete.sentiment)}`);
  console.log(`  momentum: ${JSON.stringify(athlete.momentum)}`);
}
```

### PHP

```php
<?php

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";
const PEOPLE_URL = "https://api.apitube.io/v1/people";

const DISCIPLINES = [
    "football"   => "medtop:20001065",
    "basketball" => "medtop:20000851",
    "ice_hockey" => "medtop:20000965",
    "tennis"     => "medtop:20001085",
    "cricket"    => "medtop:20000888",
];

function sportsGet(string $url, array $params, int $timeout = 60): ?array
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

    // /v1/people/{id} returns the profile with no `status` field, unlike the
    // search and news endpoints. Accept either shape.
    if (($payload["status"] ?? "") === "ok" || isset($payload["id"])) {
        return $payload;
    }

    return null;
}

function shapeArticle(array $article): array
{
    $people = [];

    foreach ($article["entities"] ?? [] as $entity) {
        if (($entity["type"] ?? "") === "person") {
            $people[] = $entity["name"];
        }
    }

    return [
        "title"        => $article["title"],
        "url"          => $article["href"] ?? "",
        "source"       => $article["source"]["domain"],
        "country"      => strtoupper($article["source"]["location"]["country_code"] ?? "??"),
        "published_at" => $article["published_at"],
        "shares"       => $article["shares"]["total"] ?? 0,
        "sentiment"    => $article["sentiment"]["overall"]["polarity"] ?? "neutral",
        "people"       => array_slice($people, 0, 4),
    ];
}

function disciplineFeed(string $discipline, int $hours = 24, int $limit = 25, string $mode = "engagement"): array
{
    $payload = sportsGet(BASE_URL, [
        "category.id"        => DISCIPLINES[$discipline],
        "sort.by"            => $mode,
        "published_at.start" => "NOW-{$hours}HOURS",
        "language.code"      => "en",
        "is_duplicate"       => 0,
        "per_page"           => $limit,
    ]);

    if ($payload === null) {
        return [];
    }

    return array_map("shapeArticle", $payload["results"] ?? []);
}

function disciplineCount(string $discipline, array $extra = [], int $days = 7): int
{
    $payload = sportsGet(COUNT_URL, [
        "category.id"        => DISCIPLINES[$discipline],
        "published_at.start" => "NOW-{$days}DAYS",
    ] + $extra, 45);

    return $payload["count"] ?? 0;
}

/**
 * Share of a discipline's coverage that mentions a keyword in the headline.
 * Raw counts are useless across disciplines — baseball carries roughly 30x
 * the volume of tennis — so always report the share.
 */
function storyShare(string $discipline, string $keyword, int $days = 7): array
{
    $total = disciplineCount($discipline, [], $days);
    $matching = disciplineCount($discipline, ["title" => $keyword], $days);

    return [
        "discipline" => $discipline,
        "keyword"    => $keyword,
        "total"      => $total,
        "matching"   => $matching,
        "share"      => $total > 0 ? round($matching / $total * 100, 2) : 0.0,
    ];
}

/**
 * Two calls are needed: the directory search returns id and profile only, and
 * the coverage block is exposed on the individual profile endpoint.
 */
function resolveAthlete(string $name): ?array
{
    $listing = sportsGet(PEOPLE_URL, ["name" => $name]);

    if ($listing === null || empty($listing["results"])) {
        return null;
    }

    $person = sportsGet(PEOPLE_URL . "/" . $listing["results"][0]["id"], []);

    if ($person === null) {
        return null;
    }

    $coverage = $person["coverage"] ?? [];

    return [
        "id"            => $person["id"],
        "name"          => $person["name"],
        "article_count" => $coverage["article_count"] ?? null,
        "sentiment"     => $coverage["sentiment"] ?? null,
        "momentum"      => $coverage["momentum"] ?? null,
    ];
}

printf("Top football coverage by engagement:\n\n");
foreach (array_slice(disciplineFeed("football", 24, 8), 0, 8) as $row) {
    printf(
        "  %6d shares [%s] %-22s %s\n",
        $row["shares"],
        $row["country"],
        substr($row["source"], 0, 22),
        substr($row["title"], 0, 44)
    );

    if (!empty($row["people"])) {
        printf("    people: %s\n", substr(implode(", ", $row["people"]), 0, 70));
    }
}

printf("\nStory shares in football, last 7 days:\n\n");
foreach (["transfer", "injury", "contract"] as $keyword) {
    $row = storyShare("football", $keyword, 7);
    printf("  %-10s %7d / %-9d %5.2f%%\n", $row["keyword"], $row["matching"], $row["total"], $row["share"]);
}

$athlete = resolveAthlete("Lionel Messi");
if ($athlete) {
    printf("\nResolved: %s (id %d) — %s articles\n", $athlete["name"], $athlete["id"], $athlete["article_count"] ?? "?");
    printf("  sentiment: %s\n", json_encode($athlete["sentiment"]));
    printf("  momentum: %s\n", json_encode($athlete["momentum"]));
}
```

## Why Not `sport.name`

The parameter reference documents a `sport.name` filter that takes values such as `Football` or `Cricket`. In practice every value currently returns:

```json
{
  "status": "not_ok",
  "errors": [{ "status": 400, "code": "ER0250", "message": "entity sport name 'Football' not found." }]
}
```

`/v1/news/trends` also rejects `field=sport.name` with `ER0350` — its allowed fields are `source.id`, `category.id`, `topic.id`, `industry.id` and `entity.id`. Scope disciplines with `category.id` and resolve athletes and clubs through `entity.id`; both are populated and stable.

## Common Use Cases

- **Fantasy and betting products** — injury and lineup coverage per discipline.
- **Transfer window monitoring** — headline-scoped rumour tracking with source ranking.
- **Athlete reputation** — sentiment split and momentum for a resolved entity.
- **Fan sentiment by market** — facet a discipline by `source.country.id`.
- **Club communications** — negative coverage alerts scoped to one organisation.
- **Editorial planning** — per-discipline engagement benchmarks to size a story.
- **Rights and sponsorship research** — coverage volume trends per league.

## See Also

- [examples.md](./examples.md) — detailed code examples for sports intelligence.
