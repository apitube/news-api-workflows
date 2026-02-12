# Supply Chain Intelligence — Code Examples

Detailed examples for building supply chain monitoring and risk assessment systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Multi-Tier Supplier Risk Monitor

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

# Example supplier tiers
SUPPLIER_TIERS = {
    "Tier 1 (Direct)": [
        {"name": "TSMC", "type": "organization", "category": "semiconductors"},
        {"name": "Samsung Electronics", "type": "organization", "category": "electronics"},
        {"name": "Foxconn", "type": "organization", "category": "manufacturing"},
    ],
    "Tier 2 (Indirect)": [
        {"name": "ASML", "type": "organization", "category": "equipment"},
        {"name": "SK Hynix", "type": "organization", "category": "memory"},
        {"name": "Tokyo Electron", "type": "organization", "category": "equipment"},
    ],
    "Tier 3 (Raw Materials)": [
        {"name": "lithium", "type": "product", "category": "materials"},
        {"name": "rare earth", "type": "product", "category": "materials"},
        {"name": "cobalt", "type": "product", "category": "materials"},
    ],
}

RISK_KEYWORDS = [
    "shortage", "disruption", "delay", "fire", "flood", "earthquake",
    "strike", "bankruptcy", "sanctions", "recall", "quality issue"
]

def assess_supplier_risk(supplier, days=7):
    """Assess risk level for a single supplier."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": supplier["name"],
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    total = len(resp.json().get("results", []))

    # Negative coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": supplier["name"],
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    negative = len(resp.json().get("results", []))

    # Risk keywords
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": supplier["name"],
        "title": ",".join(RISK_KEYWORDS),
        "published_at.start": start,
        "language.code": "en",
        "per_page": 5,
    })
    risk_data = resp.json()
    risk_hits = risk_len(data.get("results", []))
    risk_headlines = [a["title"] for a in risk_data.get("results", [])[:3]]

    # Calculate risk score
    if total > 0:
        neg_ratio = negative / total
        risk_ratio = risk_hits / total
        risk_score = min(100, (neg_ratio * 40 + risk_ratio * 60) * 100)
    else:
        risk_score = 0

    return {
        "supplier": supplier["name"],
        "category": supplier["category"],
        "total_coverage": total,
        "negative_coverage": negative,
        "risk_keyword_hits": risk_hits,
        "risk_score": risk_score,
        "risk_level": "CRITICAL" if risk_score >= 70 else \
                     "HIGH" if risk_score >= 50 else \
                     "MEDIUM" if risk_score >= 25 else "LOW",
        "risk_headlines": risk_headlines,
    }

def generate_supplier_risk_report():
    """Generate comprehensive supplier risk report."""

    print("=" * 80)
    print("MULTI-TIER SUPPLIER RISK ASSESSMENT")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    all_risks = []

    for tier_name, suppliers in SUPPLIER_TIERS.items():
        print(f"\n{tier_name}")
        print("-" * 60)

        tier_risks = []
        for supplier in suppliers:
            risk = assess_supplier_risk(supplier, days=7)
            tier_risks.append(risk)
            all_risks.append({**risk, "tier": tier_name})

        # Sort tier by risk
        tier_risks.sort(key=lambda x: x["risk_score"], reverse=True)

        for r in tier_risks:
            emoji = "🔴" if r["risk_level"] == "CRITICAL" else \
                    "🟠" if r["risk_level"] == "HIGH" else \
                    "🟡" if r["risk_level"] == "MEDIUM" else "🟢"

            print(f"  {emoji} {r['supplier']:<25} Score: {r['risk_score']:>5.0f} [{r['risk_level']}]")
            print(f"     Coverage: {r['total_coverage']} total, {r['negative_coverage']} negative, "
                  f"{r['risk_keyword_hits']} risk keywords")

            if r["risk_headlines"]:
                print(f"     Recent risks:")
                for headline in r["risk_headlines"]:
                    print(f"       • {headline[:60]}...")

    # Summary
    print("\n" + "=" * 80)
    print("RISK SUMMARY")
    print("-" * 40)

    critical = [r for r in all_risks if r["risk_level"] == "CRITICAL"]
    high = [r for r in all_risks if r["risk_level"] == "HIGH"]

    if critical:
        print(f"\n🔴 CRITICAL RISK ({len(critical)} suppliers):")
        for r in critical:
            print(f"   • {r['supplier']} ({r['tier']})")

    if high:
        print(f"\n🟠 HIGH RISK ({len(high)} suppliers):")
        for r in high:
            print(f"   • {r['supplier']} ({r['tier']})")

    return all_risks

generate_supplier_risk_report()
```

### Port and Shipping Lane Monitor

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

MAJOR_PORTS = [
    {"name": "Shanghai Port", "region": "Asia"},
    {"name": "Singapore Port", "region": "Asia"},
    {"name": "Rotterdam Port", "region": "Europe"},
    {"name": "Los Angeles Port", "region": "North America"},
    {"name": "Long Beach Port", "region": "North America"},
    {"name": "Hamburg Port", "region": "Europe"},
    {"name": "Busan Port", "region": "Asia"},
    {"name": "Shenzhen Port", "region": "Asia"},
]

SHIPPING_LANES = [
    {"name": "Suez Canal", "criticality": "HIGH"},
    {"name": "Panama Canal", "criticality": "HIGH"},
    {"name": "Strait of Malacca", "criticality": "HIGH"},
    {"name": "Strait of Hormuz", "criticality": "HIGH"},
    {"name": "Bosphorus Strait", "criticality": "MEDIUM"},
]

DISRUPTION_KEYWORDS = [
    "congestion", "delay", "backlog", "closure", "blockade",
    "accident", "collision", "grounding", "strike", "weather"
]

def monitor_port(port, days=3):
    """Monitor a major port for disruption signals."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Get port news
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "location.name": port["name"],
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    total = len(resp.json().get("results", []))

    # Get disruption news
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "location.name": port["name"],
        "title": ",".join(DISRUPTION_KEYWORDS),
        "published_at.start": start,
        "language.code": "en",
        "per_page": 5,
    })
    disruption_data = resp.json()
    disruptions = disruption_len(data.get("results", []))
    recent_issues = [a["title"] for a in disruption_data.get("results", [])[:2]]

    # Calculate congestion score
    if total > 0:
        congestion_score = min(100, (disruptions / total) * 200)
    else:
        congestion_score = 0

    return {
        "port": port["name"],
        "region": port["region"],
        "total_coverage": total,
        "disruption_mentions": disruptions,
        "congestion_score": congestion_score,
        "status": "CONGESTED" if congestion_score >= 50 else \
                 "DELAYED" if congestion_score >= 25 else "NORMAL",
        "recent_issues": recent_issues,
    }

def monitor_shipping_lane(lane, days=3):
    """Monitor a critical shipping lane."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Get lane news
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "location.name": lane["name"],
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    total = len(resp.json().get("results", []))

    # Get disruption news
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "location.name": lane["name"],
        "title": ",".join(DISRUPTION_KEYWORDS),
        "published_at.start": start,
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 3,
    })
    disruption_data = resp.json()
    disruptions = disruption_len(data.get("results", []))
    recent_alerts = [
        {"title": a["title"], "source": a["source"]["domain"]}
        for a in disruption_data.get("results", [])
    ]

    return {
        "lane": lane["name"],
        "criticality": lane["criticality"],
        "total_coverage": total,
        "disruption_alerts": disruptions,
        "status": "DISRUPTED" if disruptions > 3 else \
                 "ALERT" if disruptions > 0 else "CLEAR",
        "recent_alerts": recent_alerts,
    }

print("GLOBAL SHIPPING MONITOR")
print("=" * 70)
print(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Monitor ports
print("\nMAJOR PORTS STATUS:")
print("-" * 70)
print(f"{'Port':<25} {'Region':<15} {'Coverage':>10} {'Disruptions':>12} {'Status'}")
print("-" * 70)

port_reports = []
for port in MAJOR_PORTS:
    report = monitor_port(port, days=3)
    port_reports.append(report)

# Sort by congestion
port_reports.sort(key=lambda x: x["congestion_score"], reverse=True)

for r in port_reports:
    emoji = "🔴" if r["status"] == "CONGESTED" else \
            "🟡" if r["status"] == "DELAYED" else "🟢"
    print(f"{r['port']:<25} {r['region']:<15} {r['total_coverage']:>10} "
          f"{r['disruption_mentions']:>12} {emoji} {r['status']}")

# Monitor shipping lanes
print("\nCRITICAL SHIPPING LANES:")
print("-" * 70)

lane_reports = []
for lane in SHIPPING_LANES:
    report = monitor_shipping_lane(lane, days=3)
    lane_reports.append(report)

for r in lane_reports:
    emoji = "🔴" if r["status"] == "DISRUPTED" else \
            "🟡" if r["status"] == "ALERT" else "🟢"
    print(f"\n{emoji} {r['lane']} [{r['criticality']} criticality]: {r['status']}")
    print(f"   Coverage: {r['total_coverage']} articles, {r['disruption_alerts']} alerts")

    if r["recent_alerts"]:
        print("   Recent Alerts:")
        for alert in r["recent_alerts"]:
            print(f"     • [{alert['source']}] {alert['title'][:50]}...")
```

### Commodity Supply Chain Tracker

```python
import requests
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

COMMODITIES = {
    "Energy": {
        "items": ["oil", "natural gas", "coal", "LNG"],
        "keywords": ["price", "supply", "production", "OPEC", "shortage", "demand"],
    },
    "Metals": {
        "items": ["copper", "aluminum", "steel", "nickel", "zinc"],
        "keywords": ["price", "mining", "production", "shortage", "demand", "export"],
    },
    "Critical Materials": {
        "items": ["lithium", "cobalt", "rare earth", "graphite"],
        "keywords": ["supply", "mining", "battery", "shortage", "price", "production"],
    },
    "Agriculture": {
        "items": ["wheat", "corn", "soybeans", "rice", "fertilizer"],
        "keywords": ["price", "harvest", "export", "shortage", "weather", "production"],
    },
    "Semiconductors": {
        "items": ["chip", "semiconductor", "wafer", "memory"],
        "keywords": ["shortage", "production", "fab", "supply", "demand", "capacity"],
    },
}

def analyze_commodity(commodity, keywords, days=7):
    """Analyze supply signals for a commodity."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Get total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": f"{commodity},{','.join(keywords)}",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 30,
    })
    data = resp.json()
    articles = data.get("results", [])
    total = len(data.get("results", []))

    # Analyze sentiment
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        polarity = article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")
        sentiments[polarity] += 1

    # Calculate supply pressure index
    total_sentiment = sum(sentiments.values()) or 1
    supply_pressure = sentiments["negative"] / total_sentiment

    # Get recent headlines
    headlines = [
        {"title": a["title"], "sentiment": a.get("sentiment", {}).get("overall", {}).get("polarity", "neutral")}
        for a in articles[:5]
    ]

    return {
        "commodity": commodity,
        "total_articles": total,
        "sentiments": sentiments,
        "supply_pressure": supply_pressure,
        "outlook": "TIGHT" if supply_pressure > 0.5 else \
                  "MIXED" if supply_pressure > 0.3 else "STABLE",
        "headlines": headlines,
    }

print("COMMODITY SUPPLY CHAIN TRACKER")
print("=" * 70)
print(f"Analysis Period: Last 7 days")
print(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

all_commodities = []

for category, config in COMMODITIES.items():
    print(f"\n{category.upper()}")
    print("-" * 50)
    print(f"{'Commodity':<20} {'Articles':>10} {'Pressure':>10} {'Outlook'}")
    print("-" * 50)

    for item in config["items"]:
        result = analyze_commodity(item, config["keywords"], days=7)
        all_commodities.append({**result, "category": category})

        emoji = "🔴" if result["outlook"] == "TIGHT" else \
                "🟡" if result["outlook"] == "MIXED" else "🟢"
        print(f"{result['commodity']:<20} {result['total_articles']:>10} "
              f"{result['supply_pressure']:>9.0%} {emoji} {result['outlook']}")

# Top concerns
print("\n" + "=" * 70)
print("TOP SUPPLY CONCERNS:")
print("-" * 50)

tight_supplies = sorted(
    [c for c in all_commodities if c["outlook"] == "TIGHT"],
    key=lambda x: x["supply_pressure"],
    reverse=True
)

for c in tight_supplies[:5]:
    print(f"\n🔴 {c['commodity']} ({c['category']})")
    print(f"   Pressure: {c['supply_pressure']:.0%} | Articles: {c['total_articles']}")
    if c["headlines"]:
        print("   Recent:")
        for h in c["headlines"][:2]:
            sentiment_icon = "📉" if h["sentiment"] == "negative" else "📈" if h["sentiment"] == "positive" else "➡️"
            print(f"     {sentiment_icon} {h['title'][:55]}...")
```

---

## JavaScript

### Real-Time Disruption Alert System

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const DISRUPTION_KEYWORDS = [
  "shortage", "disruption", "delay", "bottleneck", "congestion",
  "strike", "closure", "shutdown", "fire", "flood", "earthquake"
];

const CRITICAL_SUPPLIERS = [
  { name: "TSMC", type: "organization" },
  { name: "Samsung Electronics", type: "organization" },
  { name: "Foxconn", type: "organization" },
  { name: "Intel", type: "organization" },
];

const CRITICAL_ROUTES = [
  { name: "Suez Canal", type: "location" },
  { name: "Panama Canal", type: "location" },
  { name: "Strait of Malacca", type: "location" },
];

class SupplyChainAlertSystem {
  constructor(pollInterval = 300000) {
    this.pollInterval = pollInterval;
    this.seenAlerts = new Set();
    this.handlers = [];
  }

  onAlert(handler) {
    this.handlers.push(handler);
  }

  async checkEntity(entity, type) {
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": entity,
      title: DISRUPTION_KEYWORDS.join(","),
      "published_at.start": oneHourAgo,
      "sentiment.overall.polarity": "negative",
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "10",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const newAlerts = [];

    for (const article of data.results || []) {
      const alertId = `${entity}-${article.id}`;
      if (!this.seenAlerts.has(alertId)) {
        this.seenAlerts.add(alertId);
        newAlerts.push({
          entity,
          entityType: type,
          title: article.title,
          source: article.source.domain,
          publishedAt: article.published_at,
          url: article.href,
        });
      }
    }

    return newAlerts;
  }

  async poll() {
    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] Scanning for supply chain disruptions...`);

    // Check suppliers
    for (const supplier of CRITICAL_SUPPLIERS) {
      const alerts = await this.checkEntity(supplier.name, supplier.type);
      for (const alert of alerts) {
        alert.category = "SUPPLIER";
        this.handlers.forEach(h => h(alert));
      }
    }

    // Check shipping routes
    for (const route of CRITICAL_ROUTES) {
      const alerts = await this.checkEntity(route.name, route.type);
      for (const alert of alerts) {
        alert.category = "LOGISTICS";
        this.handlers.forEach(h => h(alert));
      }
    }
  }

  async start() {
    console.log("SUPPLY CHAIN ALERT SYSTEM");
    console.log("=".repeat(50));
    console.log(`Monitoring ${CRITICAL_SUPPLIERS.length} suppliers`);
    console.log(`Monitoring ${CRITICAL_ROUTES.length} shipping routes`);
    console.log(`Poll interval: ${this.pollInterval / 1000}s`);

    await this.poll();
    setInterval(() => this.poll(), this.pollInterval);
  }
}

// Initialize
const alertSystem = new SupplyChainAlertSystem(300000);

alertSystem.onAlert((alert) => {
  console.log("\n" + "!".repeat(60));
  console.log(`🚨 SUPPLY CHAIN ALERT [${alert.category}]`);
  console.log(`Entity: ${alert.entity}`);
  console.log(`Headline: ${alert.title}`);
  console.log(`Source: ${alert.source}`);
  console.log(`Time: ${alert.publishedAt}`);
  console.log("!".repeat(60));
});

alertSystem.start();
```

### Inventory Risk Analyzer

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

// Components that affect inventory planning
const INVENTORY_ITEMS = [
  { sku: "CHIP-001", description: "Semiconductor chips", searchTerms: ["semiconductor", "chip shortage"] },
  { sku: "BAT-001", description: "Lithium batteries", searchTerms: ["lithium", "battery supply"] },
  { sku: "DISP-001", description: "Display panels", searchTerms: ["display panel", "LCD", "OLED supply"] },
  { sku: "MEM-001", description: "Memory modules", searchTerms: ["DRAM", "NAND", "memory shortage"] },
];

async function analyzeItemRisk(item, days = 14) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const params = new URLSearchParams({
    api_key: API_KEY,
    title: item.searchTerms.join(","),
    "published_at.start": start,
    "language.code": "en",
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

  const total = Object.values(sentiments).reduce((a, b) => a + b, 0) || 1;
  const supplyRisk = sentiments.negative / total;

  // Recommendation based on risk
  let recommendation;
  if (supplyRisk > 0.5) {
    recommendation = "INCREASE_BUFFER";
  } else if (supplyRisk > 0.3) {
    recommendation = "MONITOR_CLOSELY";
  } else {
    recommendation = "MAINTAIN_CURRENT";
  }

  return {
    sku: item.sku,
    description: item.description,
    totalArticles: data.results?.length || 0,
    sentiments,
    supplyRisk,
    recommendation,
  };
}

async function generateInventoryReport() {
  console.log("INVENTORY RISK ANALYSIS");
  console.log("=".repeat(60));
  console.log(`Analysis Period: 14 days\n`);

  const results = [];

  for (const item of INVENTORY_ITEMS) {
    const risk = await analyzeItemRisk(item, 14);
    results.push(risk);
  }

  results.sort((a, b) => b.supplyRisk - a.supplyRisk);

  console.log(`${"SKU".padEnd(12)} ${"Description".padEnd(25)} ${"Risk".padStart(8)} Recommendation`);
  console.log("-".repeat(70));

  for (const r of results) {
    const emoji = r.recommendation === "INCREASE_BUFFER" ? "🔴" :
                  r.recommendation === "MONITOR_CLOSELY" ? "🟡" : "🟢";

    console.log(
      `${r.sku.padEnd(12)} ${r.description.padEnd(25)} ` +
      `${(r.supplyRisk * 100).toFixed(0).padStart(6)}% ${emoji} ${r.recommendation}`
    );
  }

  // Summary
  const highRisk = results.filter(r => r.recommendation === "INCREASE_BUFFER");
  if (highRisk.length > 0) {
    console.log("\n⚠️  ACTION REQUIRED:");
    for (const r of highRisk) {
      console.log(`   • ${r.sku}: Consider increasing safety stock`);
    }
  }

  return results;
}

generateInventoryReport();
```

---

## PHP

### Supplier News Aggregator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$suppliers = [
    ["name" => "TSMC", "category" => "Semiconductors"],
    ["name" => "Samsung Electronics", "category" => "Electronics"],
    ["name" => "Foxconn", "category" => "Manufacturing"],
    ["name" => "Intel", "category" => "Semiconductors"],
    ["name" => "Micron", "category" => "Memory"],
];

$riskKeywords = ["shortage", "disruption", "delay", "fire", "strike", "closure"];

function getSupplierNews(array $supplier, int $days = 7): array
{
    global $apiKey, $baseUrl, $riskKeywords;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Get all news
    $query = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $supplier["name"],
        "published_at.start" => $start,
        "language.code"      => "en",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "per_page"           => 20,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $articles = $data["results"] ?? [];

    // Get risk news
    $query = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $supplier["name"],
        "title"              => implode(",", $riskKeywords),
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 5,
    ]);

    $riskData = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

    // Analyze sentiment
    $sentiments = ["positive" => 0, "negative" => 0, "neutral" => 0];
    foreach ($articles as $article) {
        $polarity = $article["sentiment"]["overall"]["polarity"] ?? "neutral";
        $sentiments[$polarity]++;
    }

    $total = array_sum($sentiments) ?: 1;
    $riskScore = ($sentiments["negative"] / $total) * 50 +
                 ((count($riskData["results"] ?? [])) / max($total, 1)) * 50;

    return [
        "supplier"    => $supplier["name"],
        "category"    => $supplier["category"],
        "total_news"  => count($data["results"] ?? []),
        "sentiments"  => $sentiments,
        "risk_alerts" => count($riskData["results"] ?? []),
        "risk_score"  => min(100, $riskScore),
        "risk_level"  => $riskScore >= 50 ? "HIGH" : ($riskScore >= 25 ? "MEDIUM" : "LOW"),
        "recent"      => array_map(fn($a) => [
            "title"     => $a["title"],
            "source"    => $a["source"]["domain"],
            "sentiment" => $a["sentiment"]["overall"]["polarity"] ?? "neutral",
        ], array_slice($articles, 0, 5)),
    ];
}

echo "SUPPLIER NEWS AGGREGATOR\n";
echo str_repeat("=", 70) . "\n\n";

$results = [];
foreach ($suppliers as $supplier) {
    $results[] = getSupplierNews($supplier, 7);
}

usort($results, fn($a, $b) => $b["risk_score"] <=> $a["risk_score"]);

printf("%-25s %-15s %8s %8s %s\n",
    "Supplier", "Category", "News", "Risk", "Level");
echo str_repeat("-", 65) . "\n";

foreach ($results as $r) {
    $emoji = match ($r["risk_level"]) {
        "HIGH"   => "🔴",
        "MEDIUM" => "🟡",
        default  => "🟢",
    };

    printf("%-25s %-15s %8d %7.0f%% %s %s\n",
        $r["supplier"],
        $r["category"],
        $r["total_news"],
        $r["risk_score"],
        $emoji,
        $r["risk_level"]
    );
}

// Detailed view for high-risk suppliers
$highRisk = array_filter($results, fn($r) => $r["risk_level"] === "HIGH");

if (!empty($highRisk)) {
    echo "\nHIGH-RISK SUPPLIER DETAILS:\n";
    echo str_repeat("-", 50) . "\n";

    foreach ($highRisk as $r) {
        echo "\n🔴 {$r['supplier']}\n";
        echo "   Risk Alerts: {$r['risk_alerts']}\n";
        echo "   Recent Headlines:\n";

        foreach ($r["recent"] as $news) {
            $icon = $news["sentiment"] === "negative" ? "📉" :
                   ($news["sentiment"] === "positive" ? "📈" : "➡️");
            echo "   {$icon} [{$news['source']}] " . substr($news["title"], 0, 45) . "...\n";
        }
    }
}
```
