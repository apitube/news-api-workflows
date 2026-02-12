# Regulatory News Tracking — Code Examples

Detailed examples for building regulatory intelligence and compliance monitoring systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Multi-Jurisdictional Compliance Dashboard

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

JURISDICTIONS = {
    "United States": {
        "regulators": ["SEC", "FTC", "FDA", "EPA", "DOJ", "CFPB", "FINRA", "CFTC"],
        "language": "en",
    },
    "European Union": {
        "regulators": ["European Commission", "ECB", "ESMA", "EBA", "EDPB"],
        "language": "en",
    },
    "United Kingdom": {
        "regulators": ["FCA", "CMA", "ICO", "Ofcom", "Bank of England"],
        "language": "en",
    },
    "Asia Pacific": {
        "regulators": ["MAS", "HKMA", "FSA Japan", "ASIC", "SEBI"],
        "language": "en",
    },
}

REGULATORY_CATEGORIES = {
    "enforcement": ["fine", "penalty", "violation", "enforcement", "sanction"],
    "new_rules": ["regulation", "rule", "proposal", "amendment", "legislation"],
    "investigation": ["investigation", "probe", "inquiry", "scrutiny", "review"],
    "compliance": ["compliance", "requirement", "mandate", "obligation", "deadline"],
}

def analyze_jurisdiction(name, config, days=14):
    """Analyze regulatory activity for a jurisdiction."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    results = {
        "jurisdiction": name,
        "regulators": {},
        "total_activity": 0,
        "category_breakdown": defaultdict(int),
    }

    for regulator in config["regulators"]:
        regulator_data = {
            "total": 0,
            "by_category": {},
            "top_stories": [],
        }

        # Get total regulatory news
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": regulator,
            "entity.type": "organization",
            "published_at.start": start,
            "language": config["language"],
            "per_page": 1,
        })
        regulator_data["total"] = resp.json().get("total_results", 0)
        results["total_activity"] += regulator_data["total"]

        # Analyze by category
        for category, keywords in REGULATORY_CATEGORIES.items():
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": regulator,
                "entity.type": "organization",
                "title": ",".join(keywords),
                "published_at.start": start,
                "language": config["language"],
                "per_page": 1,
            })
            count = resp.json().get("total_results", 0)
            regulator_data["by_category"][category] = count
            results["category_breakdown"][category] += count

        # Get top stories
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": regulator,
            "entity.type": "organization",
            "source.rank.opr.min": 0.6,
            "published_at.start": start,
            "language": config["language"],
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 3,
        })
        regulator_data["top_stories"] = [
            {"title": a["title"], "source": a["source"]["domain"], "date": a["published_at"][:10]}
            for a in resp.json().get("results", [])
        ]

        results["regulators"][regulator] = regulator_data

    return results

def generate_compliance_dashboard():
    """Generate multi-jurisdictional compliance dashboard."""

    print("=" * 80)
    print("MULTI-JURISDICTIONAL COMPLIANCE DASHBOARD")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Analysis Period: Last 14 days")
    print("=" * 80)

    all_results = []

    for jurisdiction, config in JURISDICTIONS.items():
        result = analyze_jurisdiction(jurisdiction, config, days=14)
        all_results.append(result)

    # Sort by activity
    all_results.sort(key=lambda x: x["total_activity"], reverse=True)

    for result in all_results:
        print(f"\n{'='*60}")
        print(f"{result['jurisdiction']}: {result['total_activity']} regulatory articles")
        print(f"{'='*60}")

        # Category breakdown
        print("\nActivity by Category:")
        for category, count in result["category_breakdown"].items():
            bar = "█" * min(20, count // 5)
            print(f"  {category:<15}: {count:>5} {bar}")

        # Top regulators
        print("\nMost Active Regulators:")
        sorted_regs = sorted(result["regulators"].items(),
                           key=lambda x: x[1]["total"], reverse=True)

        for reg_name, reg_data in sorted_regs[:5]:
            if reg_data["total"] > 0:
                print(f"\n  {reg_name}: {reg_data['total']} articles")
                print(f"    Enforcement: {reg_data['by_category'].get('enforcement', 0)} | "
                      f"New Rules: {reg_data['by_category'].get('new_rules', 0)} | "
                      f"Investigations: {reg_data['by_category'].get('investigation', 0)}")

                if reg_data["top_stories"]:
                    print("    Recent:")
                    for story in reg_data["top_stories"][:2]:
                        print(f"      • [{story['date']}] {story['title'][:50]}...")

    return all_results

generate_compliance_dashboard()
```

### Enforcement Action Tracker

```python
import requests
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ENFORCEMENT_KEYWORDS = [
    "fine", "fined", "penalty", "penalized", "settlement",
    "violation", "violated", "enforcement action", "consent order"
]

MAJOR_REGULATORS = ["SEC", "FTC", "DOJ", "CFPB", "European Commission", "FCA", "CMA"]

def extract_enforcement_details(article):
    """Extract details from an enforcement article."""

    entities = article.get("entities", [])
    organizations = [e["name"] for e in entities if e.get("type") == "organization"]
    people = [e["name"] for e in entities if e.get("type") == "person"]

    return {
        "title": article["title"],
        "source": article["source"]["domain"],
        "published_at": article["published_at"],
        "url": article["href"],
        "organizations_mentioned": organizations[:5],
        "people_mentioned": people[:3],
        "sentiment": article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
        "industries": [i["name"] for i in article.get("industries", [])[:3]],
    }

def track_enforcement_by_regulator(regulator, days=30):
    """Track all enforcement actions by a specific regulator."""

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
        "per_page": 50,
    })

    data = response.json()
    actions = [extract_enforcement_details(a) for a in data.get("results", [])]

    # Group by affected organization
    by_company = defaultdict(list)
    for action in actions:
        for org in action["organizations_mentioned"]:
            if org != regulator:
                by_company[org].append(action)

    return {
        "regulator": regulator,
        "total_actions": data.get("total_results", 0),
        "actions": actions,
        "by_company": dict(by_company),
        "industries_affected": list(set(
            ind for action in actions for ind in action["industries"]
        )),
    }

print("ENFORCEMENT ACTION TRACKER")
print("=" * 70)
print(f"Analysis Period: Last 30 days")
print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

enforcement_data = []

for regulator in MAJOR_REGULATORS:
    result = track_enforcement_by_regulator(regulator, days=30)
    enforcement_data.append(result)

# Sort by activity
enforcement_data.sort(key=lambda x: x["total_actions"], reverse=True)

# Summary
print("ENFORCEMENT ACTIVITY SUMMARY:")
print("-" * 50)
print(f"{'Regulator':<25} {'Actions':>10} {'Companies':>12}")
print("-" * 50)

for data in enforcement_data:
    companies = len(data["by_company"])
    print(f"{data['regulator']:<25} {data['total_actions']:>10} {companies:>12}")

# Detailed view
print("\n" + "=" * 70)
print("DETAILED ENFORCEMENT ACTIONS:")
print("=" * 70)

for data in enforcement_data[:3]:
    if data["total_actions"] > 0:
        print(f"\n{data['regulator']} ({data['total_actions']} actions)")
        print("-" * 50)

        for action in data["actions"][:5]:
            print(f"\n  📋 {action['title'][:65]}...")
            print(f"     Source: {action['source']} | {action['published_at'][:10]}")

            affected = [o for o in action["organizations_mentioned"]
                       if o != data["regulator"]]
            if affected:
                print(f"     Affected: {', '.join(affected[:3])}")

            if action["industries"]:
                print(f"     Industries: {', '.join(action['industries'])}")

# Companies with multiple enforcement mentions
print("\n" + "=" * 70)
print("COMPANIES WITH MULTIPLE ENFORCEMENT MENTIONS:")
print("-" * 50)

all_companies = defaultdict(list)
for data in enforcement_data:
    for company, actions in data["by_company"].items():
        all_companies[company].extend([(data["regulator"], a) for a in actions])

multi_mention = {k: v for k, v in all_companies.items() if len(v) >= 2}
sorted_companies = sorted(multi_mention.items(), key=lambda x: len(x[1]), reverse=True)

for company, mentions in sorted_companies[:10]:
    regulators = list(set(m[0] for m in mentions))
    print(f"  {company}: {len(mentions)} mentions by {', '.join(regulators)}")
```

### Industry-Specific Regulation Monitor

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

INDUSTRY_REGULATIONS = {
    "Financial Services": {
        "topics": ["banking regulation", "financial compliance", "AML", "KYC", "Basel"],
        "regulators": ["SEC", "FINRA", "OCC", "CFPB", "FCA", "ECB"],
    },
    "Healthcare": {
        "topics": ["FDA regulation", "HIPAA", "drug approval", "medical device", "clinical trial"],
        "regulators": ["FDA", "HHS", "CMS", "EMA"],
    },
    "Technology": {
        "topics": ["data privacy", "GDPR", "antitrust tech", "AI regulation", "content moderation"],
        "regulators": ["FTC", "European Commission", "ICO", "CNIL"],
    },
    "Energy": {
        "topics": ["energy regulation", "emissions", "carbon", "renewable mandate", "EPA"],
        "regulators": ["EPA", "FERC", "DOE", "European Commission"],
    },
    "Cryptocurrency": {
        "topics": ["crypto regulation", "digital asset", "stablecoin", "DeFi", "token"],
        "regulators": ["SEC", "CFTC", "FinCEN", "MiCA"],
    },
}

def analyze_industry_regulation(industry, config, days=14):
    """Analyze regulatory landscape for an industry."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    results = {
        "industry": industry,
        "total_coverage": 0,
        "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
        "by_topic": {},
        "by_regulator": {},
        "key_developments": [],
    }

    # Analyze by topic
    for topic in config["topics"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": topic,
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        results["by_topic"][topic] = resp.json().get("total_results", 0)
        results["total_coverage"] += results["by_topic"][topic]

    # Analyze by regulator
    for regulator in config["regulators"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": regulator,
            "entity.type": "organization",
            "title": ",".join(config["topics"]),
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        results["by_regulator"][regulator] = resp.json().get("total_results", 0)

    # Get key developments
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(config["topics"]),
        "source.rank.opr.min": 0.6,
        "published_at.start": start,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 20,
    })

    for article in resp.json().get("results", []):
        polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
        results["sentiment_breakdown"][polarity] += 1

        results["key_developments"].append({
            "title": article["title"],
            "source": article["source"]["domain"],
            "date": article["published_at"][:10],
            "sentiment": polarity,
        })

    # Calculate regulatory pressure
    total_sentiment = sum(results["sentiment_breakdown"].values()) or 1
    results["regulatory_pressure"] = results["sentiment_breakdown"]["negative"] / total_sentiment

    return results

print("INDUSTRY REGULATION MONITOR")
print("=" * 70)
print(f"Analysis Period: Last 14 days")
print()

all_results = []

for industry, config in INDUSTRY_REGULATIONS.items():
    result = analyze_industry_regulation(industry, config, days=14)
    all_results.append(result)

# Sort by regulatory pressure
all_results.sort(key=lambda x: x["regulatory_pressure"], reverse=True)

# Summary table
print(f"{'Industry':<20} {'Coverage':>10} {'Pressure':>10} {'Outlook'}")
print("-" * 55)

for result in all_results:
    pressure = result["regulatory_pressure"]
    outlook = "🔴 High" if pressure > 0.5 else "🟡 Medium" if pressure > 0.3 else "🟢 Low"
    print(f"{result['industry']:<20} {result['total_coverage']:>10} "
          f"{pressure:>9.0%} {outlook}")

# Detailed view
for result in all_results[:3]:
    print(f"\n{'='*60}")
    print(f"{result['industry'].upper()}")
    print(f"{'='*60}")

    print("\nTop Topics:")
    sorted_topics = sorted(result["by_topic"].items(), key=lambda x: x[1], reverse=True)
    for topic, count in sorted_topics[:5]:
        print(f"  • {topic}: {count} articles")

    print("\nActive Regulators:")
    sorted_regs = sorted(result["by_regulator"].items(), key=lambda x: x[1], reverse=True)
    for reg, count in sorted_regs[:5]:
        if count > 0:
            print(f"  • {reg}: {count} articles")

    print("\nKey Developments:")
    for dev in result["key_developments"][:3]:
        sentiment_icon = "📉" if dev["sentiment"] == "negative" else \
                        "📈" if dev["sentiment"] == "positive" else "📰"
        print(f"  {sentiment_icon} [{dev['date']}] {dev['title'][:50]}...")
```

---

## JavaScript

### Real-Time Regulatory Alert System

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const ALERT_TRIGGERS = {
  enforcement: ["fine", "penalty", "enforcement", "violation", "settlement"],
  newRule: ["new regulation", "proposed rule", "final rule", "amendment"],
  deadline: ["deadline", "effective date", "compliance date", "implementation"],
  investigation: ["investigation", "probe", "inquiry", "subpoena"],
};

const WATCHED_REGULATORS = [
  "SEC", "FTC", "FDA", "DOJ", "CFPB",
  "European Commission", "FCA", "CMA", "ICO"
];

class RegulatoryAlertSystem {
  constructor(pollInterval = 300000) {
    this.pollInterval = pollInterval;
    this.seenArticles = new Set();
    this.handlers = [];
  }

  onAlert(handler) {
    this.handlers.push(handler);
  }

  async checkForAlerts() {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const alerts = [];

    for (const regulator of WATCHED_REGULATORS) {
      for (const [alertType, keywords] of Object.entries(ALERT_TRIGGERS)) {
        const params = new URLSearchParams({
          api_key: API_KEY,
          "entity.name": regulator,
          "entity.type": "organization",
          title: keywords.join(","),
          "published_at.start": oneHourAgo,
          "source.rank.opr.min": "0.5",
          "sort.by": "published_at",
          "sort.order": "desc",
          per_page: "5",
        });

        const response = await fetch(`${BASE_URL}?${params}`);
        const data = await response.json();

        for (const article of data.results || []) {
          const alertId = `${regulator}-${article.id}`;

          if (!this.seenArticles.has(alertId)) {
            this.seenArticles.add(alertId);

            alerts.push({
              type: alertType,
              regulator,
              title: article.title,
              source: article.source.domain,
              publishedAt: article.published_at,
              url: article.href,
              priority: alertType === "enforcement" ? "HIGH" :
                       alertType === "investigation" ? "HIGH" :
                       alertType === "deadline" ? "MEDIUM" : "NORMAL",
            });
          }
        }
      }
    }

    return alerts;
  }

  async poll() {
    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] Scanning for regulatory alerts...`);

    const alerts = await this.checkForAlerts();

    for (const alert of alerts) {
      this.handlers.forEach(h => h(alert));
    }

    if (alerts.length === 0) {
      console.log("  No new alerts");
    }
  }

  async start() {
    console.log("REGULATORY ALERT SYSTEM");
    console.log("=".repeat(50));
    console.log(`Monitoring ${WATCHED_REGULATORS.length} regulators`);
    console.log(`Alert types: ${Object.keys(ALERT_TRIGGERS).join(", ")}`);
    console.log(`Poll interval: ${this.pollInterval / 1000}s`);

    await this.poll();
    setInterval(() => this.poll(), this.pollInterval);
  }
}

// Initialize
const alertSystem = new RegulatoryAlertSystem(300000);

alertSystem.onAlert((alert) => {
  const priorityEmoji = alert.priority === "HIGH" ? "🚨" :
                        alert.priority === "MEDIUM" ? "⚠️" : "📋";

  console.log("\n" + "=".repeat(60));
  console.log(`${priorityEmoji} REGULATORY ALERT [${alert.priority}]`);
  console.log(`Type: ${alert.type.toUpperCase()}`);
  console.log(`Regulator: ${alert.regulator}`);
  console.log(`Headline: ${alert.title}`);
  console.log(`Source: ${alert.source}`);
  console.log(`Time: ${alert.publishedAt}`);
  console.log("=".repeat(60));
});

alertSystem.start();
```

### Compliance Calendar Builder

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const DEADLINE_KEYWORDS = [
  "deadline", "effective date", "compliance date", "implementation date",
  "must comply", "required by", "takes effect", "goes into effect"
];

const INDUSTRIES = ["banking", "healthcare", "technology", "energy", "crypto"];

async function findUpcomingDeadlines(industry, days = 30) {
  const start = new Date().toISOString().split("T")[0];

  const params = new URLSearchParams({
    api_key: API_KEY,
    title: [...DEADLINE_KEYWORDS, industry].join(","),
    "published_at.start": new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    language: "en",
    "source.rank.opr.min": "0.5",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "30",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  const deadlines = [];

  for (const article of data.results || []) {
    // Extract potential dates from title (simplified)
    const dateMatches = article.title.match(/\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/gi);

    deadlines.push({
      title: article.title,
      source: article.source.domain,
      publishedAt: article.published_at,
      potentialDates: dateMatches || [],
      industry,
    });
  }

  return {
    industry,
    totalArticles: data.total_results || 0,
    deadlines,
  };
}

async function buildComplianceCalendar() {
  console.log("COMPLIANCE CALENDAR BUILDER");
  console.log("=".repeat(60));
  console.log(`Scanning for regulatory deadlines...\n`);

  const allDeadlines = [];

  for (const industry of INDUSTRIES) {
    const result = await findUpcomingDeadlines(industry, 30);
    allDeadlines.push(result);
  }

  for (const result of allDeadlines) {
    if (result.totalArticles > 0) {
      console.log(`\n${result.industry.toUpperCase()} (${result.totalArticles} deadline-related articles)`);
      console.log("-".repeat(50));

      for (const deadline of result.deadlines.slice(0, 5)) {
        console.log(`\n📅 ${deadline.title.slice(0, 60)}...`);
        console.log(`   Source: ${deadline.source}`);
        if (deadline.potentialDates.length > 0) {
          console.log(`   Dates mentioned: ${deadline.potentialDates.join(", ")}`);
        }
      }
    }
  }

  return allDeadlines;
}

buildComplianceCalendar();
```

---

## PHP

### Regulatory Compliance Report Generator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$jurisdictions = [
    "US"  => ["SEC", "FTC", "FDA", "DOJ", "CFPB"],
    "EU"  => ["European Commission", "ECB", "ESMA"],
    "UK"  => ["FCA", "CMA", "ICO"],
];

$categories = [
    "enforcement"   => ["fine", "penalty", "violation", "enforcement"],
    "new_rules"     => ["regulation", "rule", "proposal", "legislation"],
    "investigation" => ["investigation", "probe", "inquiry"],
];

function analyzeRegulator(string $regulator, int $days = 14): array
{
    global $apiKey, $baseUrl, $categories;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    $result = [
        "regulator" => $regulator,
        "total"     => 0,
        "by_category" => [],
        "recent"    => [],
    ];

    // Total coverage
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $regulator,
        "entity.type"        => "organization",
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $result["total"] = $data["total_results"] ?? 0;

    // By category
    foreach ($categories as $category => $keywords) {
        $query = http_build_query([
            "api_key"            => $apiKey,
            "entity.name"        => $regulator,
            "entity.type"        => "organization",
            "title"              => implode(",", $keywords),
            "published_at.start" => $start,
            "language"           => "en",
            "per_page"           => 1,
        ]);
        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $result["by_category"][$category] = $data["total_results"] ?? 0;
    }

    // Recent news
    $query = http_build_query([
        "api_key"             => $apiKey,
        "entity.name"         => $regulator,
        "entity.type"         => "organization",
        "published_at.start"  => $start,
        "language"            => "en",
        "source.rank.opr.min" => 0.5,
        "sort.by"             => "published_at",
        "sort.order"          => "desc",
        "per_page"            => 5,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $result["recent"] = array_map(fn($a) => [
        "title"        => $a["title"],
        "source"       => $a["source"]["domain"],
        "published_at" => $a["published_at"],
        "sentiment"    => $a["sentiment"]["overall"]["polarity"] ?? "neutral",
    ], $data["results"] ?? []);

    return $result;
}

function generateReport(): void
{
    global $jurisdictions;

    echo "REGULATORY COMPLIANCE REPORT\n";
    echo str_repeat("=", 70) . "\n";
    echo "Generated: " . date("Y-m-d H:i:s T") . "\n";
    echo "Analysis Period: Last 14 days\n";
    echo str_repeat("=", 70) . "\n";

    foreach ($jurisdictions as $jurisdiction => $regulators) {
        echo "\n{$jurisdiction} REGULATORS\n";
        echo str_repeat("-", 50) . "\n";

        $jurisdictionTotal = 0;

        foreach ($regulators as $regulator) {
            $analysis = analyzeRegulator($regulator, 14);
            $jurisdictionTotal += $analysis["total"];

            echo "\n  {$regulator}: {$analysis['total']} articles\n";

            // Category breakdown
            $categoryParts = [];
            foreach ($analysis["by_category"] as $cat => $count) {
                if ($count > 0) {
                    $categoryParts[] = "{$cat}: {$count}";
                }
            }
            if (!empty($categoryParts)) {
                echo "    Categories: " . implode(" | ", $categoryParts) . "\n";
            }

            // Recent headlines
            if (!empty($analysis["recent"])) {
                echo "    Recent:\n";
                foreach (array_slice($analysis["recent"], 0, 2) as $news) {
                    $icon = $news["sentiment"] === "negative" ? "📉" :
                           ($news["sentiment"] === "positive" ? "📈" : "📰");
                    echo "    {$icon} " . substr($news["title"], 0, 50) . "...\n";
                }
            }
        }

        echo "\n  Jurisdiction Total: {$jurisdictionTotal} articles\n";
    }
}

generateReport();
```

### Policy Change Detector

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$policyKeywords = [
    "new regulation", "proposed rule", "final rule", "policy change",
    "amendment", "legislation", "directive", "executive order"
];

$industries = [
    "Financial Services" => ["banking", "finance", "investment", "insurance"],
    "Healthcare"         => ["healthcare", "pharmaceutical", "medical", "FDA"],
    "Technology"         => ["technology", "data privacy", "AI", "cybersecurity"],
    "Energy"             => ["energy", "climate", "emissions", "renewable"],
];

function detectPolicyChanges(string $industry, array $topics, int $days = 7): array
{
    global $apiKey, $baseUrl, $policyKeywords;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    $allKeywords = array_merge($policyKeywords, $topics);

    $query = http_build_query([
        "api_key"             => $apiKey,
        "title"               => implode(",", $allKeywords),
        "published_at.start"  => $start,
        "language"            => "en",
        "source.rank.opr.min" => 0.5,
        "sort.by"             => "published_at",
        "sort.order"          => "desc",
        "per_page"            => 20,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

    $changes = [];
    foreach ($data["results"] ?? [] as $article) {
        $changes[] = [
            "title"        => $article["title"],
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "sentiment"    => $article["sentiment"]["overall"]["polarity"] ?? "neutral",
            "entities"     => array_column(
                array_filter($article["entities"] ?? [], fn($e) => $e["type"] === "organization"),
                "name"
            ),
        ];
    }

    return [
        "industry" => $industry,
        "total"    => $data["total_results"] ?? 0,
        "changes"  => $changes,
    ];
}

echo "POLICY CHANGE DETECTOR\n";
echo str_repeat("=", 70) . "\n";
echo "Analysis Period: Last 7 days\n\n";

foreach ($industries as $industry => $topics) {
    $result = detectPolicyChanges($industry, $topics, 7);

    echo "{$industry}: {$result['total']} policy-related articles\n";
    echo str_repeat("-", 50) . "\n";

    foreach (array_slice($result["changes"], 0, 3) as $change) {
        $icon = $change["sentiment"] === "negative" ? "⚠️" : "📋";
        echo "\n{$icon} " . substr($change["title"], 0, 60) . "...\n";
        echo "   Source: {$change['source']} | " . substr($change["published_at"], 0, 10) . "\n";

        if (!empty($change["entities"])) {
            echo "   Entities: " . implode(", ", array_slice($change["entities"], 0, 3)) . "\n";
        }
    }

    echo "\n";
}
```
