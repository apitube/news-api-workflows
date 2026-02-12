# Brand Health Scorecard — Code Examples

Advanced examples for building comprehensive brand health measurement systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Multi-Dimensional Brand Health Analyzer

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class BrandHealthAnalyzer:
    """Comprehensive multi-dimensional brand health analysis."""

    DIMENSIONS = {
        "volume": {"weight": 0.15, "description": "Media coverage volume"},
        "sentiment": {"weight": 0.25, "description": "Sentiment distribution"},
        "reach": {"weight": 0.20, "description": "Media tier distribution"},
        "momentum": {"weight": 0.15, "description": "Week-over-week growth"},
        "consistency": {"weight": 0.10, "description": "Coverage stability"},
        "share_of_voice": {"weight": 0.15, "description": "Competitive position"},
    }

    TIER_1_SOURCES = "reuters.com,bloomberg.com,nytimes.com,wsj.com,ft.com,bbc.com"
    TIER_2_SOURCES = "cnbc.com,forbes.com,businessinsider.com,techcrunch.com"

    def __init__(self, brand, competitors=None):
        self.brand = brand
        self.competitors = competitors or []
        self.metrics = {}
        self.dimension_scores = {}

    def fetch_volume_metrics(self, days=30):
        """Fetch volume-related metrics."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })

        total = len(resp.json().get("results", []))

        self.metrics["volume"] = {
            "total": total,
            "daily_avg": total / days,
            "period_days": days,
        }

        # Score: normalized against benchmark (100 articles/day = 100%)
        benchmark = 100 * days
        self.dimension_scores["volume"] = min(100, (total / benchmark) * 100)

    def fetch_sentiment_metrics(self, days=30):
        """Fetch sentiment distribution."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        sentiments = {}
        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": self.brand,
                "sentiment.overall.polarity": polarity,
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            sentiments[polarity] = len(resp.json().get("results", []))

        total = sum(sentiments.values()) or 1
        net_sentiment = (sentiments["positive"] - sentiments["negative"]) / total

        self.metrics["sentiment"] = {
            "distribution": sentiments,
            "positive_ratio": sentiments["positive"] / total,
            "negative_ratio": sentiments["negative"] / total,
            "net_sentiment": net_sentiment,
        }

        # Score: 50 + net_sentiment * 50 (range 0-100)
        self.dimension_scores["sentiment"] = 50 + net_sentiment * 50

    def fetch_reach_metrics(self, days=30):
        """Fetch media tier distribution."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Tier 1
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "source.domain": self.TIER_1_SOURCES,
            "published_at.start": start,
            "per_page": 1,
        })
        tier1 = len(resp.json().get("results", []))

        # Tier 2
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "source.domain": self.TIER_2_SOURCES,
            "published_at.start": start,
            "per_page": 1,
        })
        tier2 = len(resp.json().get("results", []))

        # High authority (OPR >= 5)
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "source.rank.opr.min": 5,
            "published_at.start": start,
            "per_page": 1,
        })
        high_authority = len(resp.json().get("results", []))

        total = self.metrics.get("volume", {}).get("total", 1) or 1

        self.metrics["reach"] = {
            "tier1_count": tier1,
            "tier2_count": tier2,
            "high_authority_count": high_authority,
            "tier1_ratio": tier1 / total,
            "high_authority_ratio": high_authority / total,
        }

        # Score: weighted by tier importance
        reach_score = (
            (tier1 / total) * 50 +
            (tier2 / total) * 30 +
            (high_authority / total) * 20
        ) * 100

        self.dimension_scores["reach"] = min(100, reach_score)

    def fetch_momentum_metrics(self):
        """Calculate week-over-week momentum."""
        current_week_start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        previous_week_start = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
        previous_week_end = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Current week
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "published_at.start": current_week_start,
            "language.code": "en",
            "per_page": 1,
        })
        current_week = len(resp.json().get("results", []))

        # Previous week
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.brand,
            "published_at.start": previous_week_start,
            "published_at.end": previous_week_end,
            "language.code": "en",
            "per_page": 1,
        })
        previous_week = len(resp.json().get("results", []))

        if previous_week > 0:
            growth_rate = (current_week - previous_week) / previous_week
        else:
            growth_rate = 1 if current_week > 0 else 0

        self.metrics["momentum"] = {
            "current_week": current_week,
            "previous_week": previous_week,
            "growth_rate": growth_rate,
        }

        # Score: 50 + growth_rate * 50 (capped at -50% to +50% growth)
        self.dimension_scores["momentum"] = 50 + min(50, max(-50, growth_rate * 50))

    def fetch_share_of_voice(self, days=30):
        """Calculate share of voice against competitors."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        volumes = {}
        all_brands = [self.brand] + self.competitors

        for brand in all_brands:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": brand,
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            volumes[brand] = len(resp.json().get("results", []))

        total = sum(volumes.values()) or 1
        share = volumes[self.brand] / total
        rank = sorted(volumes.keys(), key=lambda b: volumes[b], reverse=True).index(self.brand) + 1

        self.metrics["share_of_voice"] = {
            "volumes": volumes,
            "share": share,
            "rank": rank,
            "total_competitors": len(self.competitors),
        }

        # Score: share * 100 (higher share = higher score)
        self.dimension_scores["share_of_voice"] = share * 100

    def calculate_consistency(self, days=30):
        """Calculate coverage consistency (daily variance)."""
        # Simplified: use ratio of recent to average
        if "volume" in self.metrics and "momentum" in self.metrics:
            daily_avg = self.metrics["volume"]["daily_avg"]
            recent_daily = self.metrics["momentum"]["current_week"] / 7

            if daily_avg > 0:
                consistency = 1 - abs(recent_daily - daily_avg) / daily_avg
            else:
                consistency = 0

            self.metrics["consistency"] = {
                "daily_avg": daily_avg,
                "recent_daily": recent_daily,
                "consistency_ratio": consistency,
            }

            self.dimension_scores["consistency"] = max(0, consistency * 100)

    def calculate_composite_score(self):
        """Calculate weighted composite health score."""
        composite = 0

        for dimension, config in self.DIMENSIONS.items():
            score = self.dimension_scores.get(dimension, 50)
            composite += score * config["weight"]

        return composite

    def generate_report(self):
        """Generate comprehensive brand health report."""

        # Fetch all metrics
        print(f"Analyzing {self.brand}...")
        self.fetch_volume_metrics()
        self.fetch_sentiment_metrics()
        self.fetch_reach_metrics()
        self.fetch_momentum_metrics()
        self.calculate_consistency()

        if self.competitors:
            self.fetch_share_of_voice()

        composite_score = self.calculate_composite_score()

        return {
            "brand": self.brand,
            "generated_at": datetime.utcnow().isoformat(),
            "composite_score": composite_score,
            "grade": "A" if composite_score >= 80 else \
                    "B" if composite_score >= 65 else \
                    "C" if composite_score >= 50 else \
                    "D" if composite_score >= 35 else "F",
            "dimension_scores": self.dimension_scores,
            "metrics": self.metrics,
        }


# Generate report
brand = "Nike"
competitors = ["Adidas", "Puma", "Under Armour", "New Balance"]

analyzer = BrandHealthAnalyzer(brand, competitors)
report = analyzer.generate_report()

print("\n" + "=" * 70)
print("BRAND HEALTH SCORECARD")
print("=" * 70)

print(f"\n{report['brand']}")
print(f"Composite Score: {report['composite_score']:.1f}/100 (Grade: {report['grade']})")
print(f"Generated: {report['generated_at'][:19]}")

print("\nDIMENSION SCORES:")
print("-" * 50)

for dimension, config in BrandHealthAnalyzer.DIMENSIONS.items():
    score = report["dimension_scores"].get(dimension, 0)
    bar = "█" * int(score / 5)
    weight_pct = config["weight"] * 100
    print(f"  {dimension:<18}: {score:>5.1f} {bar:<20} (weight: {weight_pct:.0f}%)")

print("\nKEY METRICS:")
print("-" * 50)

vol = report["metrics"]["volume"]
print(f"  Volume: {vol['total']} articles ({vol['daily_avg']:.1f}/day)")

sent = report["metrics"]["sentiment"]
print(f"  Sentiment: +{sent['distribution']['positive']} / -{sent['distribution']['negative']} "
      f"(net: {sent['net_sentiment']:+.3f})")

reach = report["metrics"]["reach"]
print(f"  Tier-1 Reach: {reach['tier1_count']} ({reach['tier1_ratio']:.1%})")

mom = report["metrics"]["momentum"]
print(f"  WoW Growth: {mom['growth_rate']:+.1%} ({mom['previous_week']} → {mom['current_week']})")

if "share_of_voice" in report["metrics"]:
    sov = report["metrics"]["share_of_voice"]
    print(f"  Share of Voice: {sov['share']:.1%} (Rank #{sov['rank']})")

print("\nJSON EXPORT:")
print(json.dumps(report, indent=2, default=str))
```

### Competitive Brand Comparison Matrix

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

BRANDS = ["Apple", "Samsung", "Google", "Microsoft", "Amazon"]

METRICS_TO_COMPARE = [
    "total_coverage",
    "positive_ratio",
    "negative_ratio",
    "tier1_ratio",
    "net_sentiment",
]

def fetch_brand_metrics(brand, days=30):
    """Fetch key metrics for a brand."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": brand,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    total = len(resp.json().get("results", []))

    # Sentiment
    sentiments = {}
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": brand,
            "sentiment.overall.polarity": polarity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        sentiments[polarity] = len(resp.json().get("results", []))

    # Tier-1
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": brand,
        "source.rank.opr.min": 5,
        "published_at.start": start,
        "per_page": 1,
    })
    tier1 = len(resp.json().get("results", []))

    total_sentiment = sum(sentiments.values()) or 1

    return {
        "brand": brand,
        "total_coverage": total,
        "positive_ratio": sentiments["positive"] / total_sentiment,
        "negative_ratio": sentiments["negative"] / total_sentiment,
        "tier1_ratio": tier1 / max(total, 1),
        "net_sentiment": (sentiments["positive"] - sentiments["negative"]) / total_sentiment,
    }

print("COMPETITIVE BRAND COMPARISON MATRIX")
print("=" * 90)
print(f"Brands: {', '.join(BRANDS)}")
print(f"Period: Last 30 days\n")

# Fetch all brands
all_metrics = []
for brand in BRANDS:
    metrics = fetch_brand_metrics(brand, days=30)
    all_metrics.append(metrics)

# Print comparison table
header = f"{'Metric':<20}" + "".join(f"{b:<14}" for b in BRANDS)
print(header)
print("-" * len(header))

metric_labels = {
    "total_coverage": "Coverage",
    "positive_ratio": "% Positive",
    "negative_ratio": "% Negative",
    "tier1_ratio": "% Tier-1",
    "net_sentiment": "Net Sentiment",
}

for metric_key in METRICS_TO_COMPARE:
    row = f"{metric_labels[metric_key]:<20}"

    values = [m[metric_key] for m in all_metrics]
    max_val = max(values)
    min_val = min(values)

    for m in all_metrics:
        val = m[metric_key]

        # Format based on metric type
        if metric_key == "total_coverage":
            formatted = f"{val:,}"
        elif "ratio" in metric_key:
            formatted = f"{val:.1%}"
        else:
            formatted = f"{val:+.3f}"

        # Highlight best/worst
        if val == max_val and metric_key != "negative_ratio":
            formatted = f"🟢{formatted}"
        elif val == min_val and metric_key != "negative_ratio":
            formatted = f"🔴{formatted}"
        elif val == min_val and metric_key == "negative_ratio":
            formatted = f"🟢{formatted}"
        elif val == max_val and metric_key == "negative_ratio":
            formatted = f"🔴{formatted}"

        row += f"{formatted:<14}"

    print(row)

# Rankings
print("\n" + "=" * 90)
print("RANKINGS:")
print("-" * 50)

# By coverage
by_coverage = sorted(all_metrics, key=lambda x: x["total_coverage"], reverse=True)
print("\nBy Coverage:")
for i, m in enumerate(by_coverage, 1):
    print(f"  {i}. {m['brand']}: {m['total_coverage']:,}")

# By sentiment
by_sentiment = sorted(all_metrics, key=lambda x: x["net_sentiment"], reverse=True)
print("\nBy Net Sentiment:")
for i, m in enumerate(by_sentiment, 1):
    print(f"  {i}. {m['brand']}: {m['net_sentiment']:+.3f}")
```

---

## JavaScript

### Brand Health Dashboard Generator

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class BrandHealthDashboard {
  constructor(brand, competitors = []) {
    this.brand = brand;
    this.competitors = competitors;
    this.metrics = {};
  }

  async fetchMetrics(brandName, days = 30) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

    // Volume
    const volumeParams = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": brandName,
      "published_at.start": start,
      "language.code": "en",
      per_page: "1",
    });

    let response = await fetch(`${BASE_URL}?${volumeParams}`);
    let data = await response.json();
    const volume = data.results?.length || 0;

    // Sentiment
    const sentiment = {};
    for (const polarity of ["positive", "negative", "neutral"]) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": brandName,
        "sentiment.overall.polarity": polarity,
        "published_at.start": start,
        "language.code": "en",
        per_page: "1",
      });

      response = await fetch(`${BASE_URL}?${params}`);
      data = await response.json();
      sentiment[polarity] = data.results?.length || 0;
    }

    // Tier-1 reach
    const tier1Params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": brandName,
      "source.rank.opr.min": "5",
      "published_at.start": start,
      per_page: "1",
    });

    response = await fetch(`${BASE_URL}?${tier1Params}`);
    data = await response.json();
    const tier1 = data.results?.length || 0;

    const totalSentiment = Object.values(sentiment).reduce((a, b) => a + b, 0) || 1;

    return {
      brand: brandName,
      volume,
      sentiment,
      tier1,
      positiveRatio: sentiment.positive / totalSentiment,
      negativeRatio: sentiment.negative / totalSentiment,
      netSentiment: (sentiment.positive - sentiment.negative) / totalSentiment,
      tier1Ratio: tier1 / Math.max(volume, 1),
    };
  }

  calculateHealthScore(metrics) {
    const sentimentScore = 50 + metrics.netSentiment * 30;
    const reachScore = metrics.tier1Ratio * 20;

    return Math.max(0, Math.min(100, sentimentScore + reachScore));
  }

  async generate() {
    console.log("BRAND HEALTH DASHBOARD");
    console.log("=".repeat(60));

    // Main brand
    const mainMetrics = await this.fetchMetrics(this.brand, 30);
    const mainScore = this.calculateHealthScore(mainMetrics);

    console.log(`\n${this.brand}`);
    console.log("-".repeat(40));
    console.log(`  Health Score: ${mainScore.toFixed(1)}/100`);
    console.log(`  Volume: ${mainMetrics.volume.toLocaleString()} articles`);
    console.log(`  Sentiment: ${mainMetrics.netSentiment >= 0 ? "+" : ""}${mainMetrics.netSentiment.toFixed(3)}`);
    console.log(`  Tier-1: ${(mainMetrics.tier1Ratio * 100).toFixed(1)}%`);

    // Competitors
    if (this.competitors.length > 0) {
      console.log("\nCOMPETITOR COMPARISON:");
      console.log("-".repeat(40));

      const allBrands = [mainMetrics];

      for (const competitor of this.competitors) {
        const compMetrics = await this.fetchMetrics(competitor, 30);
        const compScore = this.calculateHealthScore(compMetrics);
        allBrands.push({ ...compMetrics, score: compScore });

        console.log(`\n  ${competitor}: ${compScore.toFixed(1)}/100`);
        console.log(`    Volume: ${compMetrics.volume.toLocaleString()}`);
        console.log(`    Sentiment: ${compMetrics.netSentiment >= 0 ? "+" : ""}${compMetrics.netSentiment.toFixed(3)}`);
      }

      // Share of voice
      const totalVolume = allBrands.reduce((sum, b) => sum + b.volume, 0);
      console.log("\nSHARE OF VOICE:");

      allBrands.sort((a, b) => b.volume - a.volume);
      for (const brand of allBrands) {
        const share = brand.volume / totalVolume;
        const bar = "#".repeat(Math.round(share * 30));
        const marker = brand.brand === this.brand ? " <- YOUR BRAND" : "";
        console.log(`  ${brand.brand.padEnd(15)}: ${(share * 100).toFixed(1).padStart(5)}% ${bar}${marker}`);
      }
    }

    this.metrics = mainMetrics;
    return { mainMetrics, score: mainScore };
  }
}

// Run dashboard
const dashboard = new BrandHealthDashboard("Nike", ["Adidas", "Puma", "Under Armour"]);
dashboard.generate();
```

---

## PHP

### Brand Performance Report

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function fetchBrandMetrics(string $brand, int $days = 30): array
{
    global $apiKey, $baseUrl;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Volume
    $query = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $brand,
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $volume = count($data["results"] ?? []);

    // Sentiment
    $sentiment = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "organization.name"          => $brand,
            "sentiment.overall.polarity" => $polarity,
            "published_at.start"         => $start,
            "language.code"              => "en",
            "per_page"                   => 1,
        ]);
        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $sentiment[$polarity] = count($data["results"] ?? []);
    }

    // Tier-1
    $query = http_build_query([
        "api_key"             => $apiKey,
        "organization.name"   => $brand,
        "source.rank.opr.min" => 5,
        "published_at.start"  => $start,
        "per_page"            => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $tier1 = count($data["results"] ?? []);

    $totalSentiment = array_sum($sentiment) ?: 1;

    return [
        "brand"          => $brand,
        "volume"         => $volume,
        "sentiment"      => $sentiment,
        "tier1"          => $tier1,
        "net_sentiment"  => ($sentiment["positive"] - $sentiment["negative"]) / $totalSentiment,
        "tier1_ratio"    => $tier1 / max($volume, 1),
    ];
}

function calculateScore(array $metrics): float
{
    $sentimentScore = 50 + $metrics["net_sentiment"] * 30;
    $reachScore = $metrics["tier1_ratio"] * 20;

    return max(0, min(100, $sentimentScore + $reachScore));
}

$brand = "Nike";
$competitors = ["Adidas", "Puma", "Under Armour"];

echo "BRAND PERFORMANCE REPORT\n";
echo str_repeat("=", 60) . "\n\n";

// Main brand
$mainMetrics = fetchBrandMetrics($brand, 30);
$mainScore = calculateScore($mainMetrics);

echo "{$brand}\n";
echo str_repeat("-", 40) . "\n";
printf("  Health Score: %.1f/100\n", $mainScore);
echo "  Volume: " . number_format($mainMetrics["volume"]) . " articles\n";
printf("  Net Sentiment: %+.3f\n", $mainMetrics["net_sentiment"]);
printf("  Tier-1 Ratio: %.1f%%\n", $mainMetrics["tier1_ratio"] * 100);

// Competitors
echo "\nCOMPETITOR COMPARISON:\n";
echo str_repeat("-", 40) . "\n";

$allBrands = [$mainMetrics];

foreach ($competitors as $competitor) {
    $metrics = fetchBrandMetrics($competitor, 30);
    $score = calculateScore($metrics);
    $allBrands[] = array_merge($metrics, ["score" => $score]);

    echo "\n  {$competitor}: " . sprintf("%.1f", $score) . "/100\n";
    echo "    Volume: " . number_format($metrics["volume"]) . "\n";
    printf("    Sentiment: %+.3f\n", $metrics["net_sentiment"]);
}

// Share of voice
$totalVolume = array_sum(array_column($allBrands, "volume")) ?: 1;

echo "\nSHARE OF VOICE:\n";

usort($allBrands, fn($a, $b) => $b["volume"] <=> $a["volume"]);

foreach ($allBrands as $b) {
    $share = $b["volume"] / $totalVolume * 100;
    $bar = str_repeat("#", (int) round($share / 3));
    $marker = $b["brand"] === $brand ? " <- YOUR BRAND" : "";
    printf("  %-15s: %5.1f%% %s%s\n", $b["brand"], $share, $bar, $marker);
}
```
