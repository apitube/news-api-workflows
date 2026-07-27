# Sports Intelligence — Examples

Advanced code examples for transfer rumour credibility scoring, club reputation dashboards, and athlete comparison.

---

## Python — Transfer Rumour Credibility Monitor

```python
import requests
import time
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

FOOTBALL = "medtop:20001065"

# Rumour vocabulary. `title` AND-matches every word, so each entry is a single
# term — running them separately and merging beats one impossible multi-word query.
RUMOUR_TERMS = ["transfer", "signing", "bid", "medical", "agreement", "talks"]


class TransferMonitor:
    """Score transfer rumours by how much authority is behind them.

    A rumour carried only by aggregators means much less than the same rumour
    in high-ranking outlets. `source.rank.opr` (0–10) is the authority signal,
    and counting distinct domains separates one story echoed everywhere from
    several outlets reporting independently.
    """

    def __init__(self, api_key, pause=0.8):
        self.api_key = api_key
        self.pause = pause

    def _get(self, params, timeout=60):
        try:
            payload = requests.get(
                BASE_URL, params={"api_key": self.api_key, **params}, timeout=timeout
            ).json()
        except (requests.RequestException, ValueError):
            return None

        time.sleep(self.pause)
        return payload if payload.get("status") == "ok" else None

    def rumours(self, category=FOOTBALL, days=7, per_term=50, language="en"):
        """Collect rumour coverage across the vocabulary, merged on article id."""
        by_id = {}

        for term in RUMOUR_TERMS:
            payload = self._get({
                "category.id": category,
                "title": term,
                "published_at.start": f"NOW-{days}DAYS",
                "language.code": language,
                "is_duplicate": 0,
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": per_term,
            })

            if not payload:
                continue

            for article in payload.get("results", []):
                article_id = article["id"]

                if article_id not in by_id:
                    by_id[article_id] = self._shape(article, term)
                else:
                    by_id[article_id]["terms"].add(term)

        return list(by_id.values())

    def _shape(self, article, term):
        source = article["source"]

        return {
            "id": article["id"],
            "title": article["title"],
            "url": article.get("href"),
            "domain": source["domain"],
            "authority": (source.get("rankings") or {}).get("opr") or 0,
            "published_at": article["published_at"],
            "shares": (article.get("shares") or {}).get("total", 0),
            "people": [e["name"] for e in (article.get("entities") or []) if e.get("type") == "person"],
            "orgs": [e["name"] for e in (article.get("entities") or []) if e.get("type") == "organization"],
            "terms": {term},
        }

    def cluster_by_player(self, rumours, min_articles=2):
        """Group rumours by the player they name and score each cluster."""
        clusters = defaultdict(list)

        for rumour in rumours:
            for person in rumour["people"][:3]:
                clusters[person].append(rumour)

        scored = []

        for player, items in clusters.items():
            if len(items) < min_articles:
                continue

            domains = {item["domain"] for item in items}
            authorities = [item["authority"] for item in items]
            top_authority = max(authorities)

            # Independent corroboration matters more than raw volume: five
            # articles from five domains beats fifteen from one aggregator.
            credibility = (
                len(domains) * 2
                + top_authority
                + (sum(authorities) / len(authorities))
            )

            scored.append({
                "player": player,
                "articles": len(items),
                "distinct_domains": len(domains),
                "top_authority": top_authority,
                "avg_authority": round(sum(authorities) / len(authorities), 1),
                "credibility": round(credibility, 1),
                "clubs": sorted({org for item in items for org in item["orgs"][:2]})[:4],
                "latest": max(item["published_at"] for item in items),
                "headline": max(items, key=lambda i: i["authority"])["title"],
            })

        return sorted(scored, key=lambda c: -c["credibility"])


monitor = TransferMonitor(API_KEY)

rumours = monitor.rumours(days=7, per_term=50)
print(f"Collected {len(rumours)} distinct rumour articles\n")

clusters = monitor.cluster_by_player(rumours, min_articles=2)
print(f"Player clusters with corroboration: {len(clusters)}\n")

print(f"{'Player':<26}{'arts':>5}{'doms':>6}{'topOPR':>8}{'cred':>7}")
for cluster in clusters[:12]:
    print(f"{cluster['player'][:24]:<26}{cluster['articles']:>5}{cluster['distinct_domains']:>6}"
          f"{cluster['top_authority']:>8}{cluster['credibility']:>7}")

print("\nTop cluster detail:\n")
for cluster in clusters[:3]:
    print(f"  {cluster['player']}")
    print(f"    {cluster['headline'][:70]}")
    print(f"    clubs: {', '.join(cluster['clubs']) or '—'}")
    print(f"    {cluster['articles']} articles across {cluster['distinct_domains']} domains, "
          f"latest {cluster['latest'][:10]}")
```

---

## JavaScript — Club Reputation Dashboard

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const COUNT_URL = "https://api.apitube.io/v1/news/count";

const FOOTBALL = "medtop:20001065";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class ClubDashboard {
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

  async count(filters) {
    const payload = await this.get(COUNT_URL, filters);
    return payload?.count ?? 0;
  }

  // Clubs are matched on the headline, not with organization.name: most club
  // names are absent from the organization entity list, and even exact forms
  // such as "Manchester United F.C." return zero. Scoping by discipline keeps
  // the headline match from picking up unrelated uses of the city name.
  //
  // Sentiment shares, not raw counts. A club with ten times the coverage will
  // always have more negative articles in absolute terms.
  async sentimentProfile(club, { days = 30, category = FOOTBALL } = {}) {
    const window = {
      "category.id": category,
      title: club,
      "published_at.start": `NOW-${days}DAYS`
    };
    const total = await this.count(window);

    if (total === 0) return { club, total: 0 };

    const positive = await this.count({ ...window, "sentiment.overall.polarity": "positive" });
    const negative = await this.count({ ...window, "sentiment.overall.polarity": "negative" });

    return {
      club,
      total,
      positive,
      negative,
      positiveShare: Number(((positive / total) * 100).toFixed(1)),
      negativeShare: Number(((negative / total) * 100).toFixed(1)),
      netTone: Number((((positive - negative) / total) * 100).toFixed(1))
    };
  }

  // One facet request replaces one count request per country.
  async geography(club, { days = 30, limit = 10, category = FOOTBALL } = {}) {
    const payload = await this.get(BASE_URL, {
      "category.id": category,
      title: club,
      "published_at.start": `NOW-${days}DAYS`,
      facet: "true",
      "facet.field": "source.country.id,language.id",
      "facet.limit": String(limit),
      per_page: "1"
    });

    if (!payload?.facets) return { countries: [], languages: [] };

    return {
      countries: (payload.facets["source.country.id"] || []).map((b) => ({
        countryId: b.value,
        articles: b.count
      })),
      languages: (payload.facets["language.id"] || []).map((b) => ({
        languageId: b.value,
        articles: b.count
      }))
    };
  }

  async worstCoverage(club, { days = 30, limit = 5, category = FOOTBALL } = {}) {
    const payload = await this.get(BASE_URL, {
      "category.id": category,
      title: club,
      "sentiment.overall.polarity": "negative",
      "published_at.start": `NOW-${days}DAYS`,
      "sort.by": "engagement",
      is_duplicate: "0",
      per_page: String(limit)
    });

    if (!payload) return [];

    return (payload.results || []).map((article) => ({
      title: article.title,
      domain: article.source.domain,
      authority: article.source.rankings?.opr || 0,
      shares: article.shares?.total || 0,
      score: article.sentiment?.overall?.score ?? 0
    }));
  }

  async report(clubs, options = {}) {
    const rows = [];

    for (const club of clubs) {
      const profile = await this.sentimentProfile(club, options);
      if (profile.total === 0) continue;

      rows.push(profile);
    }

    return rows.sort((a, b) => b.netTone - a.netTone);
  }
}

const dashboard = new ClubDashboard(API_KEY);

const CLUBS = ["Real Madrid", "Manchester United", "Bayern Munich", "Juventus"];

console.log("Club reputation, last 30 days:\n");
console.log(`${"Club".padEnd(20)}${"total".padStart(8)}${"pos%".padStart(8)}${"neg%".padStart(8)}${"net".padStart(8)}`);

const report = await dashboard.report(CLUBS, { days: 30 });
for (const row of report) {
  console.log(
    `${row.club.padEnd(20)}${String(row.total).padStart(8)}` +
      `${String(row.positiveShare).padStart(8)}${String(row.negativeShare).padStart(8)}` +
      `${String(row.netTone).padStart(8)}`
  );
}

if (report.length > 0) {
  const worst = report[report.length - 1];
  console.log(`\nMost negative coverage for ${worst.club}:\n`);

  for (const article of await dashboard.worstCoverage(worst.club, { days: 30, limit: 5 })) {
    console.log(
      `  score=${article.score.toFixed(2)} opr=${article.authority} ` +
        `${String(article.shares).padStart(5)} shares | ${article.domain.slice(0, 20).padEnd(20)} ${article.title.slice(0, 44)}`
    );
  }

  const geography = await dashboard.geography(worst.club, { days: 30, limit: 6 });
  console.log(`\nCoverage geography for ${worst.club} (country ids):`);
  geography.countries.forEach((c) => console.log(`  country ${c.countryId}: ${c.articles} articles`));
}
```

---

## PHP — Athlete Comparison

```php
<?php

const API_KEY = "YOUR_API_KEY";
const PEOPLE_URL = "https://api.apitube.io/v1/people";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

function athleteGet(string $url, array $params, int $timeout = 60): ?array
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

    // The individual profile endpoint omits `status`; search and news include it.
    if (($payload["status"] ?? "") === "ok" || isset($payload["id"])) {
        return $payload;
    }

    return null;
}

/**
 * The people directory returns id and profile; the coverage block with
 * article_count, sentiment, momentum, top_sources and top_topics only appears
 * on /v1/people/{id}. Two calls per athlete.
 */
function athleteProfile(string $name): ?array
{
    $listing = athleteGet(PEOPLE_URL, ["name" => $name]);

    if ($listing === null || empty($listing["results"])) {
        return null;
    }

    $profile = athleteGet(PEOPLE_URL . "/" . $listing["results"][0]["id"], []);

    if ($profile === null) {
        return null;
    }

    $coverage = $profile["coverage"] ?? [];
    $sentiment = $coverage["sentiment"] ?? [];
    $momentum = $coverage["momentum"] ?? [];

    $positive = $sentiment["positive"] ?? 0;
    $negative = $sentiment["negative"] ?? 0;
    $neutral = $sentiment["neutral"] ?? 0;
    $rated = $positive + $negative + $neutral;

    return [
        "id"            => $profile["id"],
        "name"          => $profile["name"],
        "article_count" => $coverage["article_count"] ?? 0,
        "first_seen"    => $coverage["first_seen"] ?? null,
        "last_seen"     => $coverage["last_seen"] ?? null,
        "positive"      => $positive,
        "negative"      => $negative,
        "net_tone"      => $rated > 0 ? round(($positive - $negative) / $rated * 100, 1) : 0.0,
        "last_30"       => $momentum["last_30_days"] ?? 0,
        "prev_30"       => $momentum["previous_30_days"] ?? 0,
        "change_pct"    => $momentum["change_pct"] ?? 0,
        "top_sources"   => array_slice(array_column($coverage["top_sources"] ?? [], "domain"), 0, 3),
        "top_topics"    => array_slice(array_column($coverage["top_topics"] ?? [], "name"), 0, 3),
    ];
}

function mostShared(int $entityId, int $limit = 3): array
{
    $payload = athleteGet(BASE_URL, [
        "entity.id"    => $entityId,
        "sort.by"      => "engagement",
        "is_duplicate" => 0,
        "per_page"     => $limit,
        "fl"           => "id,title,shares,source.domain,sentiment",
    ]);

    if ($payload === null) {
        return [];
    }

    $rows = [];

    foreach ($payload["results"] ?? [] as $article) {
        $rows[] = [
            "title"     => $article["title"],
            "domain"    => $article["source"]["domain"],
            "shares"    => $article["shares"]["total"] ?? 0,
            "sentiment" => $article["sentiment"]["overall"]["polarity"] ?? "neutral",
        ];
    }

    return $rows;
}

function compareAthletes(array $names): array
{
    $rows = [];

    foreach ($names as $name) {
        $profile = athleteProfile($name);

        if ($profile !== null) {
            $rows[] = $profile;
        }
    }

    usort($rows, fn($a, $b) => $b["article_count"] <=> $a["article_count"]);

    return $rows;
}

$athletes = compareAthletes(["Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappe"]);

printf("Athlete coverage comparison:\n\n");
printf("  %-22s%12s%10s%10s%12s\n", "Athlete", "articles", "net tone", "last 30d", "change %");

foreach ($athletes as $row) {
    printf(
        "  %-22s%12s%9.1f%%%10s%11s%%\n",
        substr($row["name"], 0, 20),
        number_format($row["article_count"]),
        $row["net_tone"],
        number_format($row["last_30"]),
        $row["change_pct"]
    );
}

printf("\nProfile detail:\n");
foreach ($athletes as $row) {
    printf("\n  %s (id %d)\n", $row["name"], $row["id"]);
    printf("    first seen %s, last seen %s\n", substr((string) $row["first_seen"], 0, 10), substr((string) $row["last_seen"], 0, 10));
    printf("    top sources: %s\n", implode(", ", $row["top_sources"]) ?: "—");
    printf("    top topics:  %s\n", implode(", ", $row["top_topics"]) ?: "—");

    foreach (mostShared($row["id"], 2) as $article) {
        printf("      %5d shares [%s] %-20s %s\n", $article["shares"], substr($article["sentiment"], 0, 3), substr($article["domain"], 0, 20), substr($article["title"], 0, 40));
    }
}
```

---

## Notes on Behaviour

- **`sport.name` does not work.** Every value returns `ER0250 entity sport name not found`, and `/v1/news/trends?field=sport.name` returns `ER0350`. Use `category.id` with a sport subcategory instead.
- **Avoid the top-level sport category.** `medtop:15000000` holds roughly 21 million articles. Combined with a composite sort and a date window it returns a gateway error rather than JSON. Discipline subcategories such as `medtop:20001065` (football) behave normally.
- **`/v1/people/{id}` has no `status` field.** The search endpoint returns `{"status":"ok","results":[...]}`, but the individual profile returns the person object directly. A guard that requires `status === "ok"` silently discards a valid profile — accept either shape.
- **`coverage` is only on the profile endpoint.** The directory listing carries `id`, `name`, `type`, `links` and `profile`. `article_count`, `sentiment`, `momentum`, `top_sources`, `top_topics` and `recent_articles` require the second call to `/v1/people/{id}`.
- **Club names are mostly missing from `organization.name`.** `Real Madrid`, `Manchester United` and `Bayern Munich` all return `ER0222 entity organization name not found`; `Juventus FC` resolves but `Manchester United F.C.` matches zero articles. Match clubs on `title` inside a discipline category instead.
- **Entity extraction is noisy on sports copy.** Person entities pulled from match reports frequently include historical figures and unrelated namesakes — stadium names resolve to the person they commemorate, for example. Filter clusters by article count and require corroboration across distinct domains before trusting a rumour.
- **Neutral sentiment is rare in article-level counts.** Filtering a discipline by `sentiment.overall.polarity=neutral` can return zero even when the positive and negative shares do not add up to the total. Compute the neutral remainder yourself rather than querying for it.
- **Mind the rate limit.** Paid plans allow 50 requests per minute. The rumour monitor issues one request per vocabulary term and the athlete comparison two per athlete; both pace themselves.

## See Also

- [README.md](./README.md) — Sports Intelligence workflow overview and quick start.
