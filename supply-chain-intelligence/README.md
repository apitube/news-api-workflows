# Supply Chain Intelligence

Workflow for monitoring supply chain disruptions, tracking logistics news, analyzing commodity markets, and identifying sourcing risks using the [APITube News API](https://apitube.io).

## Overview

The **Supply Chain Intelligence** workflow combines location tracking, industry filtering, entity monitoring, and sentiment analysis to build comprehensive supply chain visibility systems. Monitor port congestion, track shipping disruptions, analyze commodity price drivers, identify supplier risks, and detect emerging bottlenecks before they impact operations. Ideal for procurement teams, logistics managers, supply chain analysts, and operations leaders.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
GET https://api.apitube.io/v1/news/industry
GET https://api.apitube.io/v1/news/entity
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `organization.name`           | string  | Filter by company.                                                   |
| `location.name`               | string  | Filter by port or shipping route.                                   |
| `industry.id`                 | string  | Filter by industry (logistics, manufacturing, agriculture, etc.).   |
| `topic.id`                    | string  | Filter by topic (commodities, shipping, trade).                     |
| `title`                       | string  | Filter by keywords (shortage, delay, disruption, etc.).             |
| `category.id`                 | string  | Filter by IPTC category.                                             |
| `source.country.code`         | string  | Filter by source country.                                            |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `is_breaking`                 | integer | Filter for breaking news (1 or 0).                                   |
| `language.code`               | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Monitor supply chain disruption news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=supply chain,shortage,disruption,delay,bottleneck&language.code=en&per_page=20"

# Track shipping and logistics news
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&industry.id=logistics&title=shipping,freight,port,container&per_page=20"

# Monitor semiconductor supply chain
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&industry.id=semiconductors&title=chip shortage,supply,production,fab&per_page=20"
```

### Python

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

DISRUPTION_KEYWORDS = [
    "shortage", "disruption", "delay", "bottleneck", "backlog",
    "congestion", "strike", "closure", "shutdown", "recall"
]

CRITICAL_COMMODITIES = ["semiconductors", "lithium", "rare earth", "oil", "natural gas", "wheat", "fertilizer"]

def monitor_supply_chain_disruptions(hours=24):
    """Monitor for supply chain disruption signals."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    # Get disruption news
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(DISRUPTION_KEYWORDS),
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        "language.code": "en",
        "source.rank.opr.min": 3,
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 30,
    })

    data = response.json()
    disruptions = []

    for article in data.get("results", []):
        # Extract entities to identify affected areas
        entities = article.get("entities", [])
        locations = [e["name"] for e in entities if e.get("type") == "location"]
        organizations = [e["name"] for e in entities if e.get("type") == "organization"]

        disruptions.append({
            "title": article["title"],
            "source": article["source"]["domain"],
            "published_at": article["published_at"],
            "locations": locations[:3],
            "organizations": organizations[:3],
            "industries": [i["name"] for i in article.get("industries", [])[:2]],
        })

    return {
        "total_disruptions": len(data.get("results", [])),
        "disruptions": disruptions,
    }

def monitor_commodity_supply(commodity, hours=24):
    """Monitor supply news for a specific commodity."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    # Get commodity news
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": f"{commodity},supply,production,shortage,price",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 20,
    })

    data = response.json()
    articles = data.get("results", [])

    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
        sentiments[polarity] += 1

    total = sum(sentiments.values()) or 1
    supply_risk = sentiments["negative"] / total

    return {
        "commodity": commodity,
        "total_articles": len(articles),
        "sentiments": sentiments,
        "supply_risk_score": supply_risk,
        "risk_level": "HIGH" if supply_risk > 0.5 else "MEDIUM" if supply_risk > 0.3 else "LOW",
    }

print("SUPPLY CHAIN INTELLIGENCE DASHBOARD")
print("=" * 60)
print(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

# Monitor disruptions
print("ACTIVE DISRUPTIONS (Last 24h):")
print("-" * 60)
disruption_report = monitor_supply_chain_disruptions(hours=24)
print(f"Total disruption-related articles: {disruption_report['total_disruptions']}\n")

for d in disruption_report["disruptions"][:5]:
    print(f"• {d['title'][:70]}...")
    print(f"  Source: {d['source']} | {d['published_at'][:10]}")
    if d["locations"]:
        print(f"  Locations: {', '.join(d['locations'])}")
    print()

# Monitor commodities
print("\nCOMMODITY SUPPLY RISK:")
print("-" * 60)
print(f"{'Commodity':<20} {'Articles':>10} {'Risk':>8} {'Level'}")
print("-" * 50)

for commodity in CRITICAL_COMMODITIES:
    report = monitor_commodity_supply(commodity, hours=48)
    emoji = "🔴" if report["risk_level"] == "HIGH" else "🟡" if report["risk_level"] == "MEDIUM" else "🟢"
    print(f"{commodity:<20} {report['total_articles']:>10} {report['supply_risk_score']:>7.0%} {emoji} {report['risk_level']}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const DISRUPTION_KEYWORDS = [
  "shortage", "disruption", "delay", "bottleneck",
  "congestion", "strike", "closure", "shutdown"
];

async function getSupplyChainAlerts(hours = 24) {
  const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

  const params = new URLSearchParams({
    api_key: API_KEY,
    title: DISRUPTION_KEYWORDS.join(","),
    "sentiment.overall.polarity": "negative",
    "published_at.start": start,
    "language.code": "en",
    "source.rank.opr.min": "3",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "20",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  return {
    total: (data.results || []).length,
    alerts: (data.results || []).map(article => ({
      title: article.title,
      source: article.source.domain,
      publishedAt: article.published_at,
      entities: article.entities?.filter(e =>
        ["location", "organization"].includes(e.type)
      ).map(e => e.name).slice(0, 5) || [],
    })),
  };
}

async function displayDashboard() {
  console.log("SUPPLY CHAIN ALERTS");
  console.log("=".repeat(50));

  const alerts = await getSupplyChainAlerts(24);
  console.log(`\nFound ${alerts.total} disruption alerts (24h)\n`);

  for (const alert of alerts.alerts.slice(0, 10)) {
    console.log(`• ${alert.title.slice(0, 60)}...`);
    console.log(`  Source: ${alert.source} | ${alert.publishedAt.slice(0, 10)}`);
    if (alert.entities.length > 0) {
      console.log(`  Entities: ${alert.entities.join(", ")}`);
    }
    console.log();
  }
}

displayDashboard();
```

### PHP

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$disruptionKeywords = ["shortage", "disruption", "delay", "bottleneck", "congestion", "strike"];

function getSupplyChainAlerts(int $hours = 24): array
{
    global $apiKey, $baseUrl, $disruptionKeywords;

    $start = (new DateTime("-{$hours} hours"))->format("c");

    $query = http_build_query([
        "api_key"                    => $apiKey,
        "title"                      => implode(",", $disruptionKeywords),
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "language.code"              => "en",
        "source.rank.opr.min"        => 3,
        "sort.by"                    => "published_at",
        "sort.order"                 => "desc",
        "per_page"                   => 20,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

    return [
        "total"  => count($data["results"] ?? []),
        "alerts" => array_map(fn($a) => [
            "title"        => $a["title"],
            "source"       => $a["source"]["domain"],
            "published_at" => $a["published_at"],
            "entities"     => array_column(
                array_filter($a["entities"] ?? [], fn($e) =>
                    in_array($e["type"], ["location", "organization"])
                ),
                "name"
            ),
        ], $data["results"] ?? []),
    ];
}

echo "SUPPLY CHAIN ALERT DASHBOARD\n";
echo str_repeat("=", 60) . "\n\n";

$alerts = getSupplyChainAlerts(24);
echo "Found {$alerts['total']} disruption alerts (24h)\n\n";

foreach (array_slice($alerts["alerts"], 0, 10) as $alert) {
    echo "• " . substr($alert["title"], 0, 60) . "...\n";
    echo "  Source: {$alert['source']} | " . substr($alert["published_at"], 0, 10) . "\n";
    if (!empty($alert["entities"])) {
        echo "  Entities: " . implode(", ", array_slice($alert["entities"], 0, 5)) . "\n";
    }
    echo "\n";
}
```

## Common Use Cases

- **Disruption early warning** — detect supply chain disruptions before they impact operations.
- **Port congestion monitoring** — track shipping delays and port backlog status.
- **Supplier risk assessment** — monitor news about key suppliers for risk signals.
- **Commodity price drivers** — analyze news affecting commodity supply and pricing.
- **Logistics network monitoring** — track freight rates, shipping routes, and carrier news.
- **Manufacturing disruption alerts** — monitor factory closures, strikes, and production issues.
- **Inventory planning intelligence** — use news signals to inform demand forecasting.
- **Geopolitical supply risk** — identify supply chain exposure to geopolitical events.

## See Also

- [examples.md](./examples.md) — detailed code examples for supply chain intelligence workflows.
