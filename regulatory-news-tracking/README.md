# Regulatory News Tracking

Workflow for monitoring regulatory changes, tracking compliance news, analyzing policy developments, and detecting enforcement actions using the [APITube News API](https://apitube.io).

## Overview

The **Regulatory News Tracking** workflow combines topic filtering, entity tracking, keyword analysis, and multi-jurisdictional monitoring to build comprehensive regulatory intelligence systems. Track new legislation, monitor enforcement actions, analyze policy changes, detect compliance risks, and stay ahead of regulatory developments across industries and regions. Ideal for compliance teams, legal departments, financial institutions, and policy analysts.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/entity
GET https://api.apitube.io/v1/news/category
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by regulatory body, company, or jurisdiction.                |
| `entity.type`                 | string  | Filter by type: `organization`, `location`, `person`.               |
| `category.id`                 | string  | Filter by IPTC category (e.g., `medtop:11000000` for politics).    |
| `topic.id`                    | string  | Filter by topic (regulation, compliance, policy).                   |
| `title`                       | string  | Filter by keywords (fine, penalty, regulation, compliance, etc.).   |
| `industry.id`                 | string  | Filter by industry for sector-specific regulations.                 |
| `source.country.code`         | string  | Filter by jurisdiction (source country).                            |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Monitor regulatory enforcement actions
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=fine,penalty,violation,enforcement,settlement&source.rank.opr.min=0.6&language=en&per_page=20"

# Track data privacy regulations
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=GDPR,privacy,data protection,CCPA&language=en&per_page=20"

# Monitor financial regulations
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=SEC,FCA,FINRA&entity.type=organization&title=regulation,rule,enforcement&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

REGULATORY_BODIES = {
    "US": ["SEC", "FTC", "FDA", "EPA", "DOJ", "CFPB", "CFTC"],
    "EU": ["European Commission", "ECB", "EBA", "ESMA"],
    "UK": ["FCA", "CMA", "ICO", "Ofcom"],
}

ENFORCEMENT_KEYWORDS = [
    "fine", "penalty", "violation", "enforcement", "settlement",
    "investigation", "lawsuit", "compliance", "sanction", "ban"
]

def track_enforcement_actions(regulator, days=7):
    """Track enforcement actions by a regulatory body."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": regulator,
        "entity.type": "organization",
        "title": ",".join(ENFORCEMENT_KEYWORDS),
        "published_at.start": start,
        "language": "en",
        "source.rank.opr.min": 0.5,
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 20,
    })

    data = response.json()
    actions = []

    for article in data.get("results", []):
        # Extract affected entities
        entities = article.get("entities", [])
        affected = [e["name"] for e in entities
                   if e.get("type") == "organization" and e["name"] != regulator]

        actions.append({
            "title": article["title"],
            "source": article["source"]["domain"],
            "published_at": article["published_at"],
            "affected_entities": affected[:3],
            "sentiment": article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
        })

    return {
        "regulator": regulator,
        "total_actions": data.get("total_results", 0),
        "actions": actions,
    }

print("REGULATORY ENFORCEMENT TRACKER")
print("=" * 60)
print(f"Period: Last 7 days")
print(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

for jurisdiction, regulators in REGULATORY_BODIES.items():
    print(f"\n{jurisdiction} REGULATORS:")
    print("-" * 50)

    for regulator in regulators:
        report = track_enforcement_actions(regulator, days=7)
        print(f"\n  {regulator}: {report['total_actions']} enforcement-related articles")

        for action in report["actions"][:2]:
            affected = ", ".join(action["affected_entities"]) if action["affected_entities"] else "N/A"
            print(f"    • {action['title'][:55]}...")
            print(f"      Affected: {affected}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const REGULATORY_KEYWORDS = [
  "regulation", "compliance", "fine", "penalty", "enforcement",
  "investigation", "ruling", "law", "policy", "mandate"
];

async function trackRegulatoryNews(regulator, days = 7) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const params = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": regulator,
    "entity.type": "organization",
    title: REGULATORY_KEYWORDS.join(","),
    "published_at.start": start,
    language: "en",
    "source.rank.opr.min": "0.5",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "15",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  return {
    regulator,
    total: data.total_results || 0,
    articles: (data.results || []).map(article => ({
      title: article.title,
      source: article.source.domain,
      publishedAt: article.published_at,
      sentiment: article.sentiment?.overall?.polarity || "neutral",
    })),
  };
}

async function generateReport() {
  const regulators = ["SEC", "FTC", "FDA", "European Commission", "FCA"];

  console.log("REGULATORY NEWS MONITOR");
  console.log("=".repeat(50));

  for (const regulator of regulators) {
    const report = await trackRegulatoryNews(regulator, 7);
    console.log(`\n${regulator}: ${report.total} articles`);

    for (const article of report.articles.slice(0, 3)) {
      const icon = article.sentiment === "negative" ? "📉" : "📰";
      console.log(`  ${icon} ${article.title.slice(0, 50)}...`);
      console.log(`     [${article.source}] ${article.publishedAt.slice(0, 10)}`);
    }
  }
}

generateReport();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$regulators = ["SEC", "FTC", "FDA", "European Commission", "FCA"];
$keywords   = ["regulation", "compliance", "fine", "penalty", "enforcement", "investigation"];

function trackRegulator(string $regulator, int $days = 7): array
{
    global $apiKey, $baseUrl, $keywords;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    $query = http_build_query([
        "api_key"             => $apiKey,
        "entity.name"         => $regulator,
        "entity.type"         => "organization",
        "title"               => implode(",", $keywords),
        "published_at.start"  => $start,
        "language"            => "en",
        "source.rank.opr.min" => 0.5,
        "sort.by"             => "published_at",
        "sort.order"          => "desc",
        "per_page"            => 15,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

    return [
        "regulator" => $regulator,
        "total"     => $data["total_results"] ?? 0,
        "articles"  => array_map(fn($a) => [
            "title"        => $a["title"],
            "source"       => $a["source"]["domain"],
            "published_at" => $a["published_at"],
            "sentiment"    => $a["sentiment"]["overall"]["polarity"] ?? "neutral",
        ], $data["results"] ?? []),
    ];
}

echo "REGULATORY NEWS TRACKER\n";
echo str_repeat("=", 60) . "\n\n";

foreach ($regulators as $regulator) {
    $report = trackRegulator($regulator, 7);
    echo "{$regulator}: {$report['total']} articles\n";

    foreach (array_slice($report["articles"], 0, 3) as $article) {
        $icon = $article["sentiment"] === "negative" ? "📉" : "📰";
        echo "  {$icon} " . substr($article["title"], 0, 50) . "...\n";
        echo "     [{$article['source']}] " . substr($article["published_at"], 0, 10) . "\n";
    }
    echo "\n";
}
```

## Common Use Cases

- **Enforcement action monitoring** — track fines, penalties, and regulatory actions across jurisdictions.
- **New regulation alerts** — detect new laws, rules, and policy changes affecting your industry.
- **Compliance risk assessment** — monitor news about compliance failures and violations.
- **Policy change tracking** — analyze regulatory policy developments and proposed rules.
- **Industry-specific regulation** — track sector-specific regulations (fintech, healthcare, energy, etc.).
- **Multi-jurisdictional monitoring** — monitor regulatory developments across US, EU, UK, and Asia.
- **ESG regulation tracking** — monitor environmental, social, and governance regulatory developments.
- **Data privacy compliance** — track GDPR, CCPA, and other privacy regulation news.

## See Also

- [examples.md](./examples.md) — detailed code examples for regulatory news tracking workflows.
