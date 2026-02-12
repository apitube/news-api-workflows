# Executive Intelligence — Code Examples

Advanced examples for building executive monitoring and reputation analysis systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Executive Reputation Dashboard

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

EXECUTIVE_ROSTER = {
    "Apple": [
        {"name": "Tim Cook", "title": "CEO"},
        {"name": "Luca Maestri", "title": "CFO"},
    ],
    "Microsoft": [
        {"name": "Satya Nadella", "title": "CEO"},
        {"name": "Amy Hood", "title": "CFO"},
    ],
    "Tesla": [
        {"name": "Elon Musk", "title": "CEO"},
    ],
    "Amazon": [
        {"name": "Andy Jassy", "title": "CEO"},
    ],
    "Google": [
        {"name": "Sundar Pichai", "title": "CEO"},
        {"name": "Ruth Porat", "title": "CFO"},
    ],
}

REPUTATION_FACTORS = {
    "positive_coverage": {"weight": 0.25, "keywords": []},
    "negative_coverage": {"weight": -0.30, "keywords": []},
    "tier1_mentions": {"weight": 0.20, "sources": "reuters.com,bloomberg.com,wsj.com,ft.com"},
    "leadership_mentions": {"weight": 0.10, "keywords": ["leadership", "vision", "strategy", "innovation"]},
    "controversy_mentions": {"weight": -0.35, "keywords": ["scandal", "investigation", "lawsuit", "controversy", "misconduct"]},
}

def calculate_executive_reputation(name, title, company, days=30):
    """Calculate comprehensive reputation score for an executive."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    metrics = {
        "total_coverage": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "tier1_coverage": 0,
        "leadership_mentions": 0,
        "controversy_mentions": 0,
        "recent_articles": [],
    }

    # Total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "entity.type": "person",
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["total_coverage"] = resp.json().get("total_results", 0)

    # Sentiment breakdown
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": name,
            "entity.type": "person",
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        metrics[polarity] = resp.json().get("total_results", 0)

    # Tier-1 coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "entity.type": "person",
        "source.domain": REPUTATION_FACTORS["tier1_mentions"]["sources"],
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["tier1_coverage"] = resp.json().get("total_results", 0)

    # Leadership mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "title": ",".join(REPUTATION_FACTORS["leadership_mentions"]["keywords"]),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["leadership_mentions"] = resp.json().get("total_results", 0)

    # Controversy mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "title": ",".join(REPUTATION_FACTORS["controversy_mentions"]["keywords"]),
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    metrics["controversy_mentions"] = resp.json().get("total_results", 0)

    # Get recent articles
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": name,
        "entity.type": "person",
        "source.rank.opr.min": 0.5,
        "published_at.start": start,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 5,
    })
    metrics["recent_articles"] = [
        {
            "title": a["title"],
            "source": a["source"]["domain"],
            "sentiment": a.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
            "date": a["published_at"][:10],
        }
        for a in resp.json().get("results", [])
    ]

    # Calculate reputation score (0-100)
    total = metrics["total_coverage"] or 1

    # Components
    sentiment_score = (metrics["positive"] - metrics["negative"]) / total  # -1 to 1
    tier1_ratio = metrics["tier1_coverage"] / total
    leadership_ratio = metrics["leadership_mentions"] / total
    controversy_ratio = metrics["controversy_mentions"] / total

    # Weighted score
    reputation_score = 50 + (  # Start at 50 (neutral)
        sentiment_score * 25 +
        tier1_ratio * 15 +
        leadership_ratio * 10 -
        controversy_ratio * 30
    )

    reputation_score = max(0, min(100, reputation_score))

    return {
        "name": name,
        "title": title,
        "company": company,
        "metrics": metrics,
        "reputation_score": reputation_score,
        "reputation_grade": "A" if reputation_score >= 80 else \
                           "B" if reputation_score >= 65 else \
                           "C" if reputation_score >= 50 else \
                           "D" if reputation_score >= 35 else "F",
        "risk_flags": {
            "high_controversy": controversy_ratio > 0.1,
            "negative_sentiment": metrics["negative"] > metrics["positive"],
            "low_tier1": tier1_ratio < 0.1,
        },
    }

def generate_executive_dashboard():
    """Generate comprehensive executive reputation dashboard."""

    print("=" * 80)
    print("EXECUTIVE REPUTATION DASHBOARD")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Analysis Period: Last 30 days")
    print("=" * 80)

    all_executives = []

    for company, executives in EXECUTIVE_ROSTER.items():
        print(f"\n{company}")
        print("-" * 50)

        for exec_info in executives:
            result = calculate_executive_reputation(
                exec_info["name"],
                exec_info["title"],
                company,
                days=30
            )
            all_executives.append(result)

            grade_emoji = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}[result["reputation_grade"]]

            print(f"\n  {result['name']} ({result['title']})")
            print(f"    Reputation Score: {result['reputation_score']:.0f}/100 {grade_emoji} Grade {result['reputation_grade']}")
            print(f"    Coverage: {result['metrics']['total_coverage']} articles")
            print(f"    Sentiment: +{result['metrics']['positive']} / -{result['metrics']['negative']} / ={result['metrics']['neutral']}")

            # Risk flags
            flags = [k for k, v in result["risk_flags"].items() if v]
            if flags:
                print(f"    ⚠️  Flags: {', '.join(flags)}")

            # Top headlines
            if result["metrics"]["recent_articles"]:
                print("    Recent:")
                for article in result["metrics"]["recent_articles"][:2]:
                    icon = "📈" if article["sentiment"] == "positive" else \
                          "📉" if article["sentiment"] == "negative" else "📰"
                    print(f"      {icon} [{article['source']}] {article['title'][:40]}...")

    # Summary rankings
    print("\n" + "=" * 80)
    print("REPUTATION RANKINGS")
    print("-" * 50)

    all_executives.sort(key=lambda x: x["reputation_score"], reverse=True)

    print(f"\n{'Rank':<6} {'Executive':<25} {'Company':<15} {'Score':>8} {'Grade'}")
    print("-" * 65)

    for i, exec_data in enumerate(all_executives, 1):
        print(f"{i:<6} {exec_data['name']:<25} {exec_data['company']:<15} "
              f"{exec_data['reputation_score']:>7.0f} {exec_data['reputation_grade']}")

    return all_executives

generate_executive_dashboard()
```

### Leadership Change Detector

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

LEADERSHIP_KEYWORDS = {
    "appointments": ["appointed", "named", "hired", "promoted", "joined as"],
    "departures": ["resigned", "steps down", "stepping down", "departure", "leaving", "exit"],
    "transitions": ["succession", "successor", "replaced", "interim", "acting"],
}

C_SUITE_TITLES = ["CEO", "CFO", "CTO", "COO", "CMO", "CHRO", "CIO", "CSO", "President", "Chairman"]

def detect_leadership_changes(hours=24):
    """Detect recent leadership changes across companies."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    changes = []

    for change_type, keywords in LEADERSHIP_KEYWORDS.items():
        for title in C_SUITE_TITLES:
            search_terms = [f"{title} {kw}" for kw in keywords]

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(search_terms),
                "source.rank.opr.min": 0.5,
                "published_at.start": start,
                "language": "en",
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 20,
            })

            for article in resp.json().get("results", []):
                # Extract entities
                entities = article.get("entities", [])
                people = [e["name"] for e in entities if e.get("type") == "person"]
                organizations = [e["name"] for e in entities if e.get("type") == "organization"]

                changes.append({
                    "type": change_type,
                    "title": title,
                    "headline": article["title"],
                    "source": article["source"]["domain"],
                    "published_at": article["published_at"],
                    "people": people[:3],
                    "organizations": organizations[:3],
                    "url": article["href"],
                    "source_authority": article["source"].get("rank", {}).get("opr", 0),
                })

    # Deduplicate by URL
    seen_urls = set()
    unique_changes = []
    for change in changes:
        if change["url"] not in seen_urls:
            seen_urls.add(change["url"])
            unique_changes.append(change)

    # Sort by authority and recency
    unique_changes.sort(key=lambda x: (x["source_authority"], x["published_at"]), reverse=True)

    return unique_changes

print("LEADERSHIP CHANGE DETECTOR")
print("=" * 70)
print(f"Scan Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Lookback: 24 hours\n")

changes = detect_leadership_changes(hours=24)

if not changes:
    print("No leadership changes detected.")
else:
    # Group by type
    by_type = defaultdict(list)
    for change in changes:
        by_type[change["type"]].append(change)

    for change_type, items in by_type.items():
        emoji = {"appointments": "🟢", "departures": "🔴", "transitions": "🟡"}[change_type]
        print(f"\n{emoji} {change_type.upper()} ({len(items)} detected)")
        print("-" * 50)

        for item in items[:5]:
            print(f"\n  [{item['title']}] {item['headline'][:60]}...")
            print(f"    Source: {item['source']} | {item['published_at'][:10]}")
            if item["people"]:
                print(f"    People: {', '.join(item['people'])}")
            if item["organizations"]:
                print(f"    Companies: {', '.join(item['organizations'])}")
```

### Executive-Company Correlation Analyzer

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

EXECUTIVE_COMPANY_PAIRS = [
    {"executive": "Tim Cook", "company": "Apple"},
    {"executive": "Satya Nadella", "company": "Microsoft"},
    {"executive": "Elon Musk", "company": "Tesla"},
    {"executive": "Sundar Pichai", "company": "Google"},
    {"executive": "Andy Jassy", "company": "Amazon"},
]

def analyze_executive_company_correlation(executive, company, days=30):
    """Analyze correlation between executive and company sentiment."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Executive sentiment
    exec_sentiment = {}
    for polarity in ["positive", "negative"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": executive,
            "entity.type": "person",
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        exec_sentiment[polarity] = resp.json().get("total_results", 0)

    # Company sentiment
    company_sentiment = {}
    for polarity in ["positive", "negative"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": company,
            "entity.type": "organization",
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        company_sentiment[polarity] = resp.json().get("total_results", 0)

    # Co-mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": f"{executive},{company}",
        "published_at.start": start,
        "language": "en",
        "per_page": 1,
    })
    co_mentions = resp.json().get("total_results", 0)

    # Calculate scores
    exec_total = sum(exec_sentiment.values()) or 1
    exec_score = (exec_sentiment["positive"] - exec_sentiment["negative"]) / exec_total

    company_total = sum(company_sentiment.values()) or 1
    company_score = (company_sentiment["positive"] - company_sentiment["negative"]) / company_total

    # Correlation (simple comparison)
    sentiment_gap = abs(exec_score - company_score)
    correlation = 1 - min(1, sentiment_gap)

    return {
        "executive": executive,
        "company": company,
        "exec_sentiment": exec_score,
        "company_sentiment": company_score,
        "co_mentions": co_mentions,
        "sentiment_gap": sentiment_gap,
        "correlation": correlation,
        "alignment": "ALIGNED" if sentiment_gap < 0.1 else \
                    "DIVERGENT" if sentiment_gap > 0.3 else "MODERATE",
    }

print("EXECUTIVE-COMPANY SENTIMENT CORRELATION")
print("=" * 70)
print(f"Analysis Period: Last 30 days\n")

print(f"{'Executive':<20} {'Company':<15} {'Exec':>8} {'Company':>8} {'Gap':>8} {'Alignment'}")
print("-" * 70)

for pair in EXECUTIVE_COMPANY_PAIRS:
    result = analyze_executive_company_correlation(pair["executive"], pair["company"], days=30)

    alignment_emoji = {"ALIGNED": "🟢", "MODERATE": "🟡", "DIVERGENT": "🔴"}[result["alignment"]]

    print(f"{result['executive']:<20} {result['company']:<15} "
          f"{result['exec_sentiment']:>+7.3f} {result['company_sentiment']:>+7.3f} "
          f"{result['sentiment_gap']:>7.3f} {alignment_emoji} {result['alignment']}")
```

---

## JavaScript

### Executive Alert System

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const WATCHED_EXECUTIVES = [
  { name: "Tim Cook", company: "Apple", title: "CEO" },
  { name: "Satya Nadella", company: "Microsoft", title: "CEO" },
  { name: "Elon Musk", company: "Tesla", title: "CEO" },
];

const ALERT_KEYWORDS = {
  controversy: ["scandal", "investigation", "lawsuit", "allegation", "misconduct"],
  departure: ["resigned", "steps down", "departure", "leaving", "exit"],
  achievement: ["award", "recognition", "honored", "ranked", "best"],
};

class ExecutiveAlertSystem {
  constructor(executives, pollInterval = 300000) {
    this.executives = executives;
    this.pollInterval = pollInterval;
    this.seenAlerts = new Set();
    this.handlers = [];
  }

  onAlert(handler) {
    this.handlers.push(handler);
  }

  async checkExecutive(exec) {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const alerts = [];

    for (const [alertType, keywords] of Object.entries(ALERT_KEYWORDS)) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "entity.name": exec.name,
        "entity.type": "person",
        title: keywords.join(","),
        "published_at.start": oneHourAgo,
        "source.rank.opr.min": "0.5",
        per_page: "5",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();

      for (const article of data.results || []) {
        const alertId = `${exec.name}-${article.id}`;

        if (!this.seenAlerts.has(alertId)) {
          this.seenAlerts.add(alertId);

          alerts.push({
            type: alertType,
            executive: exec.name,
            company: exec.company,
            title: exec.title,
            headline: article.title,
            source: article.source.domain,
            publishedAt: article.published_at,
            sentiment: article.sentiment?.overall?.polarity || "neutral",
            priority: alertType === "controversy" ? "HIGH" :
                      alertType === "departure" ? "HIGH" : "NORMAL",
          });
        }
      }
    }

    return alerts;
  }

  async poll() {
    console.log(`\n[${new Date().toISOString()}] Scanning executives...`);

    for (const exec of this.executives) {
      const alerts = await this.checkExecutive(exec);

      for (const alert of alerts) {
        this.handlers.forEach(h => h(alert));
      }
    }
  }

  async start() {
    console.log("EXECUTIVE ALERT SYSTEM");
    console.log("=".repeat(50));
    console.log(`Monitoring: ${this.executives.map(e => e.name).join(", ")}`);

    await this.poll();
    setInterval(() => this.poll(), this.pollInterval);
  }
}

const alertSystem = new ExecutiveAlertSystem(WATCHED_EXECUTIVES, 300000);

alertSystem.onAlert((alert) => {
  const emoji = alert.priority === "HIGH" ? "🚨" : "📰";
  const typeEmoji = {
    controversy: "⚠️",
    departure: "🚪",
    achievement: "🏆",
  }[alert.type];

  console.log(`\n${emoji} EXECUTIVE ALERT [${alert.priority}]`);
  console.log(`${typeEmoji} Type: ${alert.type.toUpperCase()}`);
  console.log(`   ${alert.executive} (${alert.title}, ${alert.company})`);
  console.log(`   "${alert.headline}"`);
  console.log(`   Source: ${alert.source}`);
});

alertSystem.start();
```

---

## PHP

### Executive Media Report Generator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$executives = [
    ["name" => "Tim Cook", "company" => "Apple", "title" => "CEO"],
    ["name" => "Satya Nadella", "company" => "Microsoft", "title" => "CEO"],
    ["name" => "Elon Musk", "company" => "Tesla", "title" => "CEO"],
];

function generateExecutiveReport(array $exec, int $days = 30): array
{
    global $apiKey, $baseUrl;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Total coverage
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $exec["name"],
        "entity.type"        => "person",
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $totalCoverage = $data["total_results"] ?? 0;

    // Sentiment
    $sentiments = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "entity.name"                => $exec["name"],
            "entity.type"                => "person",
            "sentiment.overall.polarity" => $polarity,
            "published_at.start"         => $start,
            "language"                   => "en",
            "per_page"                   => 1,
        ]);
        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $sentiments[$polarity] = $data["total_results"] ?? 0;
    }

    // Recent headlines
    $query = http_build_query([
        "api_key"             => $apiKey,
        "entity.name"         => $exec["name"],
        "entity.type"         => "person",
        "source.rank.opr.min" => 0.5,
        "published_at.start"  => $start,
        "language"            => "en",
        "sort.by"             => "published_at",
        "sort.order"          => "desc",
        "per_page"            => 5,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $headlines = array_map(fn($a) => [
        "title"     => $a["title"],
        "source"    => $a["source"]["domain"],
        "sentiment" => $a["sentiment"]["overall"]["polarity"] ?? "neutral",
    ], $data["results"] ?? []);

    // Calculate reputation
    $total = array_sum($sentiments) ?: 1;
    $reputationScore = 50 + (($sentiments["positive"] - $sentiments["negative"]) / $total) * 50;

    return [
        "executive"        => $exec,
        "total_coverage"   => $totalCoverage,
        "sentiments"       => $sentiments,
        "reputation_score" => $reputationScore,
        "headlines"        => $headlines,
    ];
}

echo "EXECUTIVE MEDIA REPORT\n";
echo str_repeat("=", 70) . "\n";
echo "Period: Last 30 days\n\n";

foreach ($executives as $exec) {
    $report = generateExecutiveReport($exec, 30);

    echo "{$exec['name']} ({$exec['title']}, {$exec['company']})\n";
    echo str_repeat("-", 50) . "\n";
    echo "  Coverage: {$report['total_coverage']} articles\n";
    printf("  Reputation Score: %.0f/100\n", $report["reputation_score"]);
    echo "  Sentiment: +{$report['sentiments']['positive']} / -{$report['sentiments']['negative']}\n";

    echo "  Recent Headlines:\n";
    foreach (array_slice($report["headlines"], 0, 3) as $h) {
        $icon = $h["sentiment"] === "positive" ? "📈" :
               ($h["sentiment"] === "negative" ? "📉" : "📰");
        echo "    {$icon} [{$h['source']}] " . substr($h["title"], 0, 40) . "...\n";
    }

    echo "\n";
}
```
