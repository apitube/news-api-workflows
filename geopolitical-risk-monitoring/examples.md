# Geopolitical Risk Monitoring — Code Examples

Detailed examples for building geopolitical intelligence and risk assessment systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Global Hotspot Monitor

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

HOTSPOTS = {
    "Ukraine-Russia": {
        "locations": ["Ukraine", "Russia"],
        "keywords": ["war", "military", "missile", "offensive", "frontline", "Crimea", "Donbas"],
    },
    "Taiwan Strait": {
        "locations": ["Taiwan", "China"],
        "keywords": ["military", "invasion", "blockade", "drills", "strait", "independence"],
    },
    "Middle East": {
        "locations": ["Israel", "Iran", "Gaza", "Lebanon"],
        "keywords": ["conflict", "strike", "attack", "Hamas", "Hezbollah", "nuclear"],
    },
    "Korean Peninsula": {
        "locations": ["North Korea", "South Korea"],
        "keywords": ["missile", "nuclear", "test", "military", "demilitarized"],
    },
    "South China Sea": {
        "locations": ["Philippines", "Vietnam", "China"],
        "keywords": ["territorial", "dispute", "navy", "reef", "island", "maritime"],
    },
}

def analyze_hotspot(name, config, hours=24):
    """Analyze activity level for a geopolitical hotspot."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    locations = config["locations"]
    keywords = config["keywords"]

    metrics = {
        "total_coverage": 0,
        "negative_coverage": 0,
        "keyword_hits": 0,
        "breaking_news": 0,
        "top_headlines": [],
    }

    # Get coverage for all locations
    for location in locations:
        # Total
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": location,
            "entity.type": "location",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        metrics["total_coverage"] += resp.json().get("total_results", 0)

        # Negative
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": location,
            "entity.type": "location",
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        metrics["negative_coverage"] += resp.json().get("total_results", 0)

    # Keyword hits (combined locations)
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": ",".join(locations),
        "title": ",".join(keywords),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["keyword_hits"] = resp.json().get("total_results", 0)

    # Breaking news
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": ",".join(locations),
        "is_breaking": "true",
        "published_at.start": start,
        "per_page": 5,
    })
    data = resp.json()
    metrics["breaking_news"] = data.get("total_results", 0)
    metrics["top_headlines"] = [
        {"title": a["title"], "source": a["source"]["domain"]}
        for a in data.get("results", [])[:3]
    ]

    # Calculate activity score
    total = metrics["total_coverage"] or 1
    activity_score = min(100, (
        (metrics["negative_coverage"] / total) * 25 +
        (metrics["keyword_hits"] / total) * 35 +
        min(25, metrics["breaking_news"] * 5) +
        min(15, metrics["total_coverage"] * 0.1)
    ))

    status = "🔴 CRITICAL" if activity_score >= 70 else \
             "🟠 HIGH" if activity_score >= 50 else \
             "🟡 ELEVATED" if activity_score >= 30 else "🟢 LOW"

    return {
        "hotspot": name,
        "locations": locations,
        "metrics": metrics,
        "activity_score": activity_score,
        "status": status,
    }

print("=" * 70)
print("GLOBAL HOTSPOT MONITOR")
print(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 70)

results = []
for name, config in HOTSPOTS.items():
    result = analyze_hotspot(name, config, hours=24)
    results.append(result)

# Sort by activity score
results.sort(key=lambda x: x["activity_score"], reverse=True)

for r in results:
    print(f"\n{r['status']} {r['hotspot']}: {r['activity_score']:.0f}/100")
    print(f"   Locations: {', '.join(r['locations'])}")
    print(f"   Coverage: {r['metrics']['total_coverage']} total, "
          f"{r['metrics']['negative_coverage']} negative")
    print(f"   Conflict Keywords: {r['metrics']['keyword_hits']} | "
          f"Breaking: {r['metrics']['breaking_news']}")

    if r["metrics"]["top_headlines"]:
        print("   Recent Headlines:")
        for h in r["metrics"]["top_headlines"]:
            print(f"     • [{h['source']}] {h['title'][:55]}...")
```

### Sanctions Tracker

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

SANCTIONED_ENTITIES = {
    "countries": ["Russia", "Iran", "North Korea", "Venezuela", "Cuba", "Syria", "Belarus"],
    "organizations": ["Wagner Group", "Huawei", "ZTE", "Kaspersky"],
}

SANCTION_KEYWORDS = [
    "sanctions", "embargo", "blacklist", "OFAC", "SDN",
    "asset freeze", "trade ban", "export control", "restricted"
]

def track_sanctions_news(hours=72):
    """Track recent sanctions-related news."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    sanctions_news = []

    # Track country sanctions
    for country in SANCTIONED_ENTITIES["countries"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "entity.type": "location",
            "title": ",".join(SANCTION_KEYWORDS),
            "published_at.start": start,
            "language": "en",
            "source.rank.opr.min": 0.5,
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 10,
        })

        for article in resp.json().get("results", []):
            sanctions_news.append({
                "entity": country,
                "entity_type": "country",
                "title": article["title"],
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "sentiment": article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
                "opr": article["source"].get("rank", {}).get("opr", 0),
            })

    # Track organization sanctions
    for org in SANCTIONED_ENTITIES["organizations"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": org,
            "entity.type": "organization",
            "title": ",".join(SANCTION_KEYWORDS),
            "published_at.start": start,
            "language": "en",
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 5,
        })

        for article in resp.json().get("results", []):
            sanctions_news.append({
                "entity": org,
                "entity_type": "organization",
                "title": article["title"],
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "sentiment": article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
                "opr": article["source"].get("rank", {}).get("opr", 0),
            })

    # Sort by date and authority
    sanctions_news.sort(key=lambda x: (x["published_at"], x["opr"]), reverse=True)

    return sanctions_news

def generate_sanctions_report():
    """Generate a sanctions activity report."""

    print("SANCTIONS TRACKER REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Lookback: 72 hours")
    print()

    news = track_sanctions_news(hours=72)

    # Group by entity
    by_entity = defaultdict(list)
    for item in news:
        by_entity[item["entity"]].append(item)

    print(f"Found {len(news)} sanctions-related articles\n")

    # Summary by entity
    print("MENTIONS BY ENTITY:")
    print("-" * 40)

    sorted_entities = sorted(by_entity.items(), key=lambda x: len(x[1]), reverse=True)

    for entity, articles in sorted_entities[:10]:
        entity_type = articles[0]["entity_type"]
        print(f"  {entity} ({entity_type}): {len(articles)} articles")

    # Recent headlines
    print("\nRECENT SANCTIONS NEWS:")
    print("-" * 70)

    for item in news[:15]:
        sentiment_icon = "📈" if item["sentiment"] == "positive" else \
                        "📉" if item["sentiment"] == "negative" else "➡️"

        print(f"\n[{item['entity']}] {sentiment_icon}")
        print(f"  {item['title'][:65]}...")
        print(f"  Source: {item['source']} | {item['published_at'][:10]}")

generate_sanctions_report()
```

### Diplomatic Relations Monitor

```python
import requests
from datetime import datetime, timedelta
from itertools import combinations

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

KEY_BILATERAL_RELATIONS = [
    ("United States", "China"),
    ("United States", "Russia"),
    ("China", "Russia"),
    ("United States", "Iran"),
    ("India", "China"),
    ("Israel", "Iran"),
    ("Saudi Arabia", "Iran"),
    ("Japan", "China"),
]

DIPLOMATIC_KEYWORDS = [
    "diplomatic", "relations", "summit", "talks", "agreement",
    "tension", "dispute", "ambassador", "embassy", "bilateral"
]

def analyze_bilateral_relations(country1, country2, days=7):
    """Analyze news coverage of bilateral relations."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Search for articles mentioning both countries
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": f"{country1},{country2}",
        "entity.type": "location",
        "published_at.start": start,
        "language": "en",
        "per_page": 50,
    })

    data = resp.json()
    total = data.get("total_results", 0)
    articles = data.get("results", [])

    # Analyze sentiment distribution
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
        sentiments[polarity] += 1

    # Get diplomatic-specific coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": f"{country1},{country2}",
        "title": ",".join(DIPLOMATIC_KEYWORDS),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    diplomatic_mentions = resp.json().get("total_results", 0)

    # Calculate relation score (-1 to +1)
    total_sentiment = sum(sentiments.values()) or 1
    relation_score = (sentiments["positive"] - sentiments["negative"]) / total_sentiment

    # Determine trend
    if relation_score > 0.2:
        trend = "🟢 IMPROVING"
    elif relation_score < -0.2:
        trend = "🔴 DETERIORATING"
    else:
        trend = "🟡 STABLE"

    return {
        "pair": f"{country1} - {country2}",
        "total_coverage": total,
        "sentiments": sentiments,
        "diplomatic_mentions": diplomatic_mentions,
        "relation_score": relation_score,
        "trend": trend,
    }

print("DIPLOMATIC RELATIONS MONITOR")
print("=" * 70)
print(f"Analysis Period: Last 7 days")
print()

results = []
for country1, country2 in KEY_BILATERAL_RELATIONS:
    result = analyze_bilateral_relations(country1, country2, days=7)
    results.append(result)

# Sort by coverage
results.sort(key=lambda x: x["total_coverage"], reverse=True)

print(f"{'Bilateral Pair':<30} {'Coverage':>10} {'Score':>8} {'Trend'}")
print("-" * 70)

for r in results:
    print(f"{r['pair']:<30} {r['total_coverage']:>10} {r['relation_score']:>+7.3f} {r['trend']}")

print("\n" + "=" * 70)
print("RELATIONS REQUIRING ATTENTION (Score < -0.1):")
print("-" * 70)

for r in sorted(results, key=lambda x: x["relation_score"]):
    if r["relation_score"] < -0.1:
        print(f"\n{r['trend']} {r['pair']}")
        print(f"   Score: {r['relation_score']:+.3f}")
        print(f"   Coverage: {r['total_coverage']} articles "
              f"(+{r['sentiments']['positive']} / -{r['sentiments']['negative']})")
```

### Regional Stability Index

```python
import requests
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

REGIONS = {
    "Western Europe": {
        "countries": ["France", "Germany", "United Kingdom", "Italy", "Spain"],
        "weight": 1.0,
    },
    "Eastern Europe": {
        "countries": ["Poland", "Ukraine", "Romania", "Czech Republic", "Hungary"],
        "weight": 1.2,
    },
    "Middle East": {
        "countries": ["Saudi Arabia", "Iran", "Israel", "UAE", "Turkey"],
        "weight": 1.5,
    },
    "East Asia": {
        "countries": ["China", "Japan", "South Korea", "Taiwan"],
        "weight": 1.3,
    },
    "South Asia": {
        "countries": ["India", "Pakistan", "Bangladesh", "Sri Lanka"],
        "weight": 1.2,
    },
    "Southeast Asia": {
        "countries": ["Indonesia", "Vietnam", "Philippines", "Thailand", "Malaysia"],
        "weight": 1.1,
    },
    "Latin America": {
        "countries": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile"],
        "weight": 1.0,
    },
    "Africa": {
        "countries": ["Nigeria", "South Africa", "Egypt", "Kenya", "Ethiopia"],
        "weight": 1.1,
    },
}

INSTABILITY_KEYWORDS = [
    "conflict", "protest", "coup", "violence", "unrest",
    "crisis", "emergency", "martial law", "curfew"
]

def calculate_stability_index(region_name, config, days=7):
    """Calculate stability index for a region (0-100, higher = more stable)."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    countries = config["countries"]
    weight = config["weight"]

    total_coverage = 0
    negative_coverage = 0
    instability_mentions = 0

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
        total_coverage += resp.json().get("total_results", 0)

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
        negative_coverage += resp.json().get("total_results", 0)

        # Instability keywords
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": country,
            "title": ",".join(INSTABILITY_KEYWORDS),
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        instability_mentions += resp.json().get("total_results", 0)

    # Calculate raw instability score (0-100)
    total = total_coverage or 1
    neg_ratio = negative_coverage / total
    instability_ratio = instability_mentions / total

    raw_instability = (neg_ratio * 40 + instability_ratio * 60) * weight

    # Convert to stability index (invert and normalize)
    stability_index = max(0, min(100, 100 - raw_instability * 100))

    return {
        "region": region_name,
        "countries": countries,
        "total_coverage": total_coverage,
        "negative_coverage": negative_coverage,
        "instability_mentions": instability_mentions,
        "stability_index": stability_index,
        "risk_level": "LOW" if stability_index >= 70 else \
                     "MODERATE" if stability_index >= 50 else \
                     "ELEVATED" if stability_index >= 30 else "HIGH",
    }

print("REGIONAL STABILITY INDEX")
print("=" * 70)
print(f"Assessment Date: {datetime.utcnow().strftime('%Y-%m-%d')}")
print(f"Analysis Period: 7 days")
print()

indices = []
for region_name, config in REGIONS.items():
    index = calculate_stability_index(region_name, config, days=7)
    indices.append(index)

# Sort by stability (most stable first)
indices.sort(key=lambda x: x["stability_index"], reverse=True)

print(f"{'Region':<20} {'Index':>8} {'Risk Level':<12} {'Coverage':>10} {'Instability':>12}")
print("-" * 70)

for idx in indices:
    emoji = "🟢" if idx["risk_level"] == "LOW" else \
            "🟡" if idx["risk_level"] == "MODERATE" else \
            "🟠" if idx["risk_level"] == "ELEVATED" else "🔴"

    print(f"{idx['region']:<20} {idx['stability_index']:>7.1f} {emoji} {idx['risk_level']:<10} "
          f"{idx['total_coverage']:>10} {idx['instability_mentions']:>12}")

# Summary
print("\n" + "=" * 70)
avg_stability = sum(i["stability_index"] for i in indices) / len(indices)
print(f"Global Average Stability Index: {avg_stability:.1f}/100")

high_risk = [i for i in indices if i["risk_level"] in ["HIGH", "ELEVATED"]]
if high_risk:
    print(f"\nRegions Requiring Attention: {', '.join(i['region'] for i in high_risk)}")

# Export as JSON
print("\nJSON Export:")
print(json.dumps(indices, indent=2, default=str))
```

---

## JavaScript

### Real-Time Geopolitical Alert System

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const WATCHED_REGIONS = {
  "Ukraine-Russia": ["Ukraine", "Russia"],
  "Taiwan Strait": ["Taiwan", "China"],
  "Middle East": ["Israel", "Iran", "Gaza"],
  "Korean Peninsula": ["North Korea", "South Korea"],
};

const ALERT_KEYWORDS = [
  "attack", "strike", "invasion", "missile", "nuclear",
  "war", "escalation", "troops", "bombing", "casualties"
];

class GeopoliticalAlertSystem {
  constructor(regions, pollInterval = 300000) {
    this.regions = regions;
    this.pollInterval = pollInterval;
    this.seenArticles = new Set();
    this.alertHandlers = [];
  }

  onAlert(handler) {
    this.alertHandlers.push(handler);
  }

  async checkRegion(regionName, countries) {
    const thirtyMinutesAgo = new Date(Date.now() - 30 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": countries.join(","),
      "entity.type": "location",
      title: ALERT_KEYWORDS.join(","),
      is_breaking: "true",
      "published_at.start": thirtyMinutesAgo,
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "10",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const alerts = [];

    for (const article of data.results || []) {
      if (!this.seenArticles.has(article.id)) {
        this.seenArticles.add(article.id);

        alerts.push({
          level: "CRITICAL",
          region: regionName,
          title: article.title,
          source: article.source.domain,
          url: article.href,
          publishedAt: article.published_at,
          sentiment: article.sentiment?.overall?.polarity,
        });
      }
    }

    return alerts;
  }

  async poll() {
    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] Scanning for geopolitical alerts...`);

    for (const [regionName, countries] of Object.entries(this.regions)) {
      const alerts = await this.checkRegion(regionName, countries);

      for (const alert of alerts) {
        this.alertHandlers.forEach(handler => handler(alert));
      }
    }
  }

  async start() {
    console.log("GEOPOLITICAL ALERT SYSTEM");
    console.log("=".repeat(50));
    console.log(`Monitoring regions: ${Object.keys(this.regions).join(", ")}`);
    console.log(`Poll interval: ${this.pollInterval / 1000}s`);

    await this.poll();
    setInterval(() => this.poll(), this.pollInterval);
  }
}

// Initialize and run
const alertSystem = new GeopoliticalAlertSystem(WATCHED_REGIONS, 300000);

alertSystem.onAlert((alert) => {
  console.log("\n" + "!".repeat(60));
  console.log(`🚨 GEOPOLITICAL ALERT [${alert.level}]`);
  console.log(`Region: ${alert.region}`);
  console.log(`Headline: ${alert.title}`);
  console.log(`Source: ${alert.source}`);
  console.log(`Time: ${alert.publishedAt}`);
  console.log("!".repeat(60));
});

alertSystem.start();
```

### Trade War Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const TRADE_PAIRS = [
  { countries: ["United States", "China"], name: "US-China" },
  { countries: ["European Union", "China"], name: "EU-China" },
  { countries: ["United States", "European Union"], name: "US-EU" },
];

const TRADE_KEYWORDS = [
  "tariff", "trade war", "import duty", "export ban",
  "trade deficit", "sanctions", "protectionism", "subsidy"
];

async function analyzeTradeRelations(pair, days = 14) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const params = new URLSearchParams({
    api_key: API_KEY,
    "entity.name": pair.countries.join(","),
    title: TRADE_KEYWORDS.join(","),
    "published_at.start": start,
    language: "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "50",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  const articles = data.results || [];

  const sentiments = { positive: 0, negative: 0, neutral: 0 };
  for (const article of articles) {
    const polarity = article.sentiment?.overall?.polarity || "neutral";
    sentiments[polarity]++;
  }

  const total = data.total_results || 0;
  const sentimentTotal = Object.values(sentiments).reduce((a, b) => a + b, 0) || 1;
  const tensionScore = sentiments.negative / sentimentTotal;

  return {
    pair: pair.name,
    countries: pair.countries,
    totalArticles: total,
    sentiments,
    tensionScore,
    status: tensionScore > 0.5 ? "🔴 HIGH TENSION" :
            tensionScore > 0.3 ? "🟠 ELEVATED" :
            tensionScore > 0.15 ? "🟡 MODERATE" : "🟢 STABLE",
    recentHeadlines: articles.slice(0, 3).map(a => ({
      title: a.title,
      source: a.source.domain,
    })),
  };
}

async function generateTradeReport() {
  console.log("TRADE WAR MONITOR");
  console.log("=".repeat(60));
  console.log(`Analysis Period: Last 14 days\n`);

  const results = [];

  for (const pair of TRADE_PAIRS) {
    const analysis = await analyzeTradeRelations(pair, 14);
    results.push(analysis);
  }

  results.sort((a, b) => b.tensionScore - a.tensionScore);

  console.log(`${"Trade Pair".padEnd(15)} ${"Articles".padStart(10)} ${"Tension".padStart(10)} Status`);
  console.log("-".repeat(55));

  for (const r of results) {
    console.log(
      `${r.pair.padEnd(15)} ${String(r.totalArticles).padStart(10)} ` +
      `${(r.tensionScore * 100).toFixed(1).padStart(9)}% ${r.status}`
    );
  }

  console.log("\nRecent Headlines by Pair:");
  for (const r of results) {
    console.log(`\n${r.pair}:`);
    for (const h of r.recentHeadlines) {
      console.log(`  • [${h.source}] ${h.title.slice(0, 50)}...`);
    }
  }

  return results;
}

generateTradeReport();
```

---

## PHP

### Country Risk Dashboard

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$watchedCountries = [
    "Ukraine"     => ["region" => "Europe", "risk_weight" => 1.5],
    "Taiwan"      => ["region" => "Asia", "risk_weight" => 1.4],
    "Iran"        => ["region" => "Middle East", "risk_weight" => 1.3],
    "North Korea" => ["region" => "Asia", "risk_weight" => 1.5],
    "Venezuela"   => ["region" => "Americas", "risk_weight" => 1.2],
    "Myanmar"     => ["region" => "Asia", "risk_weight" => 1.1],
    "Ethiopia"    => ["region" => "Africa", "risk_weight" => 1.1],
    "Sudan"       => ["region" => "Africa", "risk_weight" => 1.2],
];

$riskKeywords = ["conflict", "war", "crisis", "sanctions", "coup", "violence", "protest"];

function assessCountry(string $country, array $config, int $days = 7): array
{
    global $apiKey, $baseUrl, $riskKeywords;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    $metrics = [];

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

    // Risk keywords
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $country,
        "title"              => implode(",", $riskKeywords),
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["risk_keywords"] = $data["total_results"] ?? 0;

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

    // Calculate risk score
    $total = $metrics["total"] ?: 1;
    $rawScore = (
        ($metrics["negative"] / $total) * 30 +
        ($metrics["risk_keywords"] / $total) * 40 +
        min(20, $metrics["breaking"] * 3)
    ) * $config["risk_weight"];

    $riskScore = min(100, $rawScore * 100);

    return [
        "country"    => $country,
        "region"     => $config["region"],
        "metrics"    => $metrics,
        "risk_score" => $riskScore,
        "risk_level" => match (true) {
            $riskScore >= 70 => "CRITICAL",
            $riskScore >= 50 => "HIGH",
            $riskScore >= 30 => "ELEVATED",
            default          => "LOW",
        },
    ];
}

echo "COUNTRY RISK DASHBOARD\n";
echo str_repeat("=", 70) . "\n";
echo "Analysis Period: 7 days\n\n";

$results = [];
foreach ($watchedCountries as $country => $config) {
    $results[] = assessCountry($country, $config, 7);
}

usort($results, fn($a, $b) => $b["risk_score"] <=> $a["risk_score"]);

printf("%-15s %-15s %10s %12s %s\n",
    "Country", "Region", "Coverage", "Risk Score", "Level");
echo str_repeat("-", 65) . "\n";

foreach ($results as $r) {
    $emoji = match ($r["risk_level"]) {
        "CRITICAL" => "🔴",
        "HIGH"     => "🟠",
        "ELEVATED" => "🟡",
        default    => "🟢",
    };

    printf("%-15s %-15s %10d %11.0f %s %s\n",
        $r["country"],
        $r["region"],
        $r["metrics"]["total"],
        $r["risk_score"],
        $emoji,
        $r["risk_level"]
    );
}

// Summary by region
echo "\nRISK BY REGION:\n";
echo str_repeat("-", 40) . "\n";

$byRegion = [];
foreach ($results as $r) {
    $byRegion[$r["region"]][] = $r["risk_score"];
}

foreach ($byRegion as $region => $scores) {
    $avgScore = array_sum($scores) / count($scores);
    $emoji = $avgScore >= 50 ? "🔴" : ($avgScore >= 30 ? "🟡" : "🟢");
    printf("  %s %-15s: %.0f avg risk\n", $emoji, $region, $avgScore);
}
```
