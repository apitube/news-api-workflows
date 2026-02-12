# Geopolitical Risk Monitoring

Workflow for tracking geopolitical events, analyzing regional instability, monitoring sanctions and trade policies, and assessing country-specific risks using the [APITube News API](https://apitube.io).

## Overview

The **Geopolitical Risk Monitoring** workflow combines location-based filtering, topic analysis, sentiment tracking, and multi-language coverage to build comprehensive geopolitical intelligence systems. Monitor conflicts, track sanctions, analyze trade disputes, assess election impacts, and identify emerging risks across regions. Ideal for multinational corporations, investment firms, supply chain managers, and policy analysts.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/trends
GET https://api.apitube.io/v1/news/entity
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by country, region, or political figure.                     |
| `entity.type`                 | string  | Filter by type: `location`, `person`, `organization`.               |
| `topic.id`                    | string  | Filter by topic (e.g., `war`, `sanctions`, `trade`).               |
| `category.id`                 | string  | Filter by IPTC category (e.g., `medtop:11000000` for politics).    |
| `source.country.code`         | string  | Filter by source country (ISO 3166-1).                              |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `title`                       | string  | Filter by keywords (sanctions, conflict, military, etc.).           |
| `language`                    | string  | Filter by language code (comma-separated for multi-lang).           |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `is_breaking`                 | boolean | Filter for breaking news.                                            |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `sort.by`                     | string  | Sort field: `published_at`, `sentiment.overall.score`.              |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Monitor conflict-related news for a region
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Ukraine&entity.type=location&title=war,conflict,military,missile&language=en&per_page=20"

# Track sanctions news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=sanctions,embargo,tariff&source.rank.opr.min=0.6&language=en&per_page=20"

# Monitor political instability across multiple countries
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Russia,China,Iran,North Korea&entity.type=location&category.id=medtop:11000000&per_page=30"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

RISK_KEYWORDS = [
    "war", "conflict", "military", "invasion", "missile",
    "sanctions", "embargo", "tariff", "trade war",
    "coup", "protest", "unrest", "instability"
]

REGIONS = {
    "Eastern Europe": ["Ukraine", "Russia", "Belarus", "Moldova"],
    "Middle East": ["Iran", "Israel", "Saudi Arabia", "Iraq", "Syria"],
    "East Asia": ["China", "Taiwan", "North Korea", "South Korea"],
    "South Asia": ["India", "Pakistan", "Afghanistan"],
}

def assess_regional_risk(region_name, countries, hours=24):
    """Assess geopolitical risk for a region."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    risk_metrics = {
        "total_coverage": 0,
        "negative_coverage": 0,
        "conflict_mentions": 0,
        "breaking_news": 0,
    }

    for country in countries:
        # Total coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "entity.type": "location",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        risk_metrics["total_coverage"] += resp.json().get("total_results", 0)

        # Negative coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "entity.type": "location",
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        risk_metrics["negative_coverage"] += resp.json().get("total_results", 0)

        # Conflict-related mentions
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "title": ",".join(RISK_KEYWORDS),
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        risk_metrics["conflict_mentions"] += resp.json().get("total_results", 0)

        # Breaking news
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "entity.type": "location",
            "is_breaking": "true",
            "published_at.start": start,
            "per_page": 1,
        })
        risk_metrics["breaking_news"] += resp.json().get("total_results", 0)

    # Calculate risk score
    total = risk_metrics["total_coverage"] or 1
    neg_ratio = risk_metrics["negative_coverage"] / total
    conflict_ratio = risk_metrics["conflict_mentions"] / total

    risk_score = min(100, (
        neg_ratio * 30 +
        conflict_ratio * 40 +
        min(20, risk_metrics["breaking_news"] * 5) +
        min(10, risk_metrics["conflict_mentions"] * 0.5)
    ))

    return {
        "region": region_name,
        "countries": countries,
        "metrics": risk_metrics,
        "risk_score": risk_score,
        "risk_level": "CRITICAL" if risk_score >= 70 else "HIGH" if risk_score >= 50 else "ELEVATED" if risk_score >= 30 else "LOW",
    }

print("GEOPOLITICAL RISK ASSESSMENT")
print("=" * 60)
print(f"Assessment Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()

results = []
for region, countries in REGIONS.items():
    assessment = assess_regional_risk(region, countries, hours=24)
    results.append(assessment)

# Sort by risk score
results.sort(key=lambda x: x["risk_score"], reverse=True)

for r in results:
    emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "ELEVATED": "🟡", "LOW": "🟢"}[r["risk_level"]]
    print(f"\n{emoji} {r['region']}: {r['risk_score']:.0f}/100 ({r['risk_level']})")
    print(f"   Countries: {', '.join(r['countries'])}")
    print(f"   Coverage: {r['metrics']['total_coverage']} | Negative: {r['metrics']['negative_coverage']}")
    print(f"   Conflict Keywords: {r['metrics']['conflict_mentions']} | Breaking: {r['metrics']['breaking_news']}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const RISK_KEYWORDS = [
  "war", "conflict", "military", "invasion", "sanctions",
  "embargo", "coup", "protest", "missile", "nuclear"
];

async function monitorCountryRisk(country, hours = 24) {
  const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

  const metrics = {
    totalCoverage: 0,
    negativeCoverage: 0,
    conflictMentions: 0,
    breakingNews: 0,
  };

  // Parallel fetch all metrics
  const requests = [
    fetch(`${BASE_URL}?${new URLSearchParams({
      api_key: API_KEY,
      "entity.name": country,
      "entity.type": "location",
      "published_at.start": start,
      language: "en",
      per_page: "1",
    })}`),
    fetch(`${BASE_URL}?${new URLSearchParams({
      api_key: API_KEY,
      "entity.name": country,
      "entity.type": "location",
      "sentiment.overall.polarity": "negative",
      "published_at.start": start,
      language: "en",
      per_page: "1",
    })}`),
    fetch(`${BASE_URL}?${new URLSearchParams({
      api_key: API_KEY,
      "entity.name": country,
      title: RISK_KEYWORDS.join(","),
      "published_at.start": start,
      per_page: "1",
    })}`),
    fetch(`${BASE_URL}?${new URLSearchParams({
      api_key: API_KEY,
      "entity.name": country,
      is_breaking: "true",
      "published_at.start": start,
      per_page: "1",
    })}`),
  ];

  const responses = await Promise.all(requests);
  const data = await Promise.all(responses.map(r => r.json()));

  metrics.totalCoverage = data[0].total_results || 0;
  metrics.negativeCoverage = data[1].total_results || 0;
  metrics.conflictMentions = data[2].total_results || 0;
  metrics.breakingNews = data[3].total_results || 0;

  const total = metrics.totalCoverage || 1;
  const riskScore = Math.min(100,
    (metrics.negativeCoverage / total) * 30 +
    (metrics.conflictMentions / total) * 40 +
    Math.min(20, metrics.breakingNews * 5)
  );

  return { country, metrics, riskScore };
}

// Monitor multiple countries
const countries = ["Ukraine", "Taiwan", "Iran", "North Korea", "Venezuela"];

console.log("COUNTRY RISK MONITOR");
console.log("=".repeat(50));

Promise.all(countries.map(c => monitorCountryRisk(c))).then(results => {
  results.sort((a, b) => b.riskScore - a.riskScore);

  for (const r of results) {
    const level = r.riskScore >= 70 ? "🔴 CRITICAL" :
                  r.riskScore >= 50 ? "🟠 HIGH" :
                  r.riskScore >= 30 ? "🟡 ELEVATED" : "🟢 LOW";

    console.log(`\n${r.country}: ${r.riskScore.toFixed(0)}/100 ${level}`);
    console.log(`  Coverage: ${r.metrics.totalCoverage} | Negative: ${r.metrics.negativeCoverage}`);
    console.log(`  Conflict: ${r.metrics.conflictMentions} | Breaking: ${r.metrics.breakingNews}`);
  }
});
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$riskKeywords = ["war", "conflict", "military", "sanctions", "embargo", "coup", "protest"];

$countries = [
    "Ukraine"     => "Eastern Europe",
    "Taiwan"      => "East Asia",
    "Iran"        => "Middle East",
    "North Korea" => "East Asia",
    "Venezuela"   => "South America",
];

function assessCountryRisk(string $country, int $hours = 24): array
{
    global $apiKey, $baseUrl, $riskKeywords;

    $start = (new DateTime("-{$hours} hours"))->format("c");

    $metrics = [
        "total"    => 0,
        "negative" => 0,
        "conflict" => 0,
        "breaking" => 0,
    ];

    // Total coverage
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $country,
        "entity.type"        => "location",
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["total"] = $data["total_results"] ?? 0;

    // Negative coverage
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $country,
        "entity.type"                => "location",
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "language"                   => "en",
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["negative"] = $data["total_results"] ?? 0;

    // Conflict keywords
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $country,
        "title"              => implode(",", $riskKeywords),
        "published_at.start" => $start,
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["conflict"] = $data["total_results"] ?? 0;

    // Breaking news
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $country,
        "is_breaking"        => "true",
        "published_at.start" => $start,
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["breaking"] = $data["total_results"] ?? 0;

    $total = $metrics["total"] ?: 1;
    $riskScore = min(100,
        ($metrics["negative"] / $total) * 30 +
        ($metrics["conflict"] / $total) * 40 +
        min(20, $metrics["breaking"] * 5)
    );

    return [
        "country"    => $country,
        "metrics"    => $metrics,
        "risk_score" => $riskScore,
        "risk_level" => $riskScore >= 70 ? "CRITICAL" :
                       ($riskScore >= 50 ? "HIGH" :
                       ($riskScore >= 30 ? "ELEVATED" : "LOW")),
    ];
}

echo "GEOPOLITICAL RISK DASHBOARD\n";
echo str_repeat("=", 60) . "\n";

$results = [];
foreach ($countries as $country => $region) {
    $assessment = assessCountryRisk($country, 24);
    $assessment["region"] = $region;
    $results[] = $assessment;
}

usort($results, fn($a, $b) => $b["risk_score"] <=> $a["risk_score"]);

foreach ($results as $r) {
    $emoji = match ($r["risk_level"]) {
        "CRITICAL" => "🔴",
        "HIGH"     => "🟠",
        "ELEVATED" => "🟡",
        default    => "🟢",
    };

    echo "\n{$emoji} {$r['country']} ({$r['region']}): {$r['risk_score']:.0f}/100 [{$r['risk_level']}]\n";
    echo "   Total: {$r['metrics']['total']} | Negative: {$r['metrics']['negative']}\n";
    echo "   Conflict: {$r['metrics']['conflict']} | Breaking: {$r['metrics']['breaking']}\n";
}
```

## Common Use Cases

- **Conflict monitoring** — track military actions, territorial disputes, and armed conflicts.
- **Sanctions tracking** — monitor new sanctions, embargo announcements, and trade restrictions.
- **Election risk assessment** — analyze coverage and sentiment around elections in key markets.
- **Trade dispute analysis** — track tariff announcements, trade negotiations, and policy changes.
- **Country risk scoring** — build quantitative risk scores for country exposure management.
- **Supply chain geopolitical risk** — identify risks in sourcing regions and shipping routes.
- **Diplomatic relations tracking** — monitor bilateral relations and diplomatic incidents.
- **Energy security monitoring** — track geopolitical events affecting energy supply.

## See Also

- [examples.md](./examples.md) — detailed code examples for geopolitical risk monitoring workflows.
