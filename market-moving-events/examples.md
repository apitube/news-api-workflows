# Market-Moving Events Detection — Code Examples

Advanced examples for building real-time market event detection systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Real-Time Event Velocity Analyzer

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class EventVelocityAnalyzer:
    """Detect market-moving events by analyzing mention velocity changes."""

    def __init__(self, watchlist):
        self.watchlist = watchlist
        self.baselines = {}
        self.history = defaultdict(list)

    def calculate_baseline(self, entity, days=7):
        """Calculate historical baseline for an entity."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })

        total = len(resp.json().get("results", []))
        daily_avg = total / days
        hourly_avg = daily_avg / 24

        self.baselines[entity] = {
            "daily_avg": daily_avg,
            "hourly_avg": hourly_avg,
            "total_7d": total,
        }

        return self.baselines[entity]

    def get_current_velocity(self, entity, minutes=60):
        """Get current mention velocity for an entity."""
        start = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })

        current_count = len(resp.json().get("results", []))
        hourly_rate = current_count * (60 / minutes)

        return {
            "count": current_count,
            "hourly_rate": hourly_rate,
            "window_minutes": minutes,
        }

    def detect_velocity_spikes(self, threshold_multiplier=3.0):
        """Detect entities with abnormal mention velocity."""
        spikes = []

        for entity in self.watchlist:
            if entity not in self.baselines:
                self.calculate_baseline(entity)

            baseline = self.baselines[entity]
            current = self.get_current_velocity(entity, minutes=60)

            if baseline["hourly_avg"] > 0:
                velocity_ratio = current["hourly_rate"] / baseline["hourly_avg"]
            else:
                velocity_ratio = current["hourly_rate"] * 10 if current["hourly_rate"] > 0 else 0

            if velocity_ratio >= threshold_multiplier:
                # Get recent articles for context
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "organization.name": entity,
                    "published_at.start": (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z",
                    "language.code": "en",
                    "source.rank.opr.min": 4,
                    "sort.by": "published_at",
                    "sort.order": "desc",
                    "per_page": 5,
                })

                articles = resp.json().get("results", [])

                spikes.append({
                    "entity": entity,
                    "baseline_hourly": baseline["hourly_avg"],
                    "current_hourly": current["hourly_rate"],
                    "velocity_ratio": velocity_ratio,
                    "recent_articles": [
                        {
                            "title": a["title"],
                            "source": a["source"]["domain"],
                            "sentiment": a.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
                        }
                        for a in articles
                    ],
                })

        return sorted(spikes, key=lambda x: x["velocity_ratio"], reverse=True)

    def get_sentiment_shift(self, entity, hours=24):
        """Detect sudden sentiment shifts."""
        # Recent sentiment (last 2 hours)
        recent_start = (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z"
        recent_sentiments = {}

        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity,
                "sentiment.overall.polarity": polarity,
                "published_at.start": recent_start,
                "language.code": "en",
                "per_page": 1,
            })
            recent_sentiments[polarity] = len(resp.json().get("results", []))

        # Historical sentiment (last 24 hours)
        hist_start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
        hist_sentiments = {}

        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity,
                "sentiment.overall.polarity": polarity,
                "published_at.start": hist_start,
                "language.code": "en",
                "per_page": 1,
            })
            hist_sentiments[polarity] = len(resp.json().get("results", []))

        # Calculate sentiment scores
        recent_total = sum(recent_sentiments.values()) or 1
        hist_total = sum(hist_sentiments.values()) or 1

        recent_score = (recent_sentiments["positive"] - recent_sentiments["negative"]) / recent_total
        hist_score = (hist_sentiments["positive"] - hist_sentiments["negative"]) / hist_total

        sentiment_shift = recent_score - hist_score

        return {
            "entity": entity,
            "recent_score": recent_score,
            "historical_score": hist_score,
            "shift": sentiment_shift,
            "shift_direction": "BULLISH" if sentiment_shift > 0.1 else "BEARISH" if sentiment_shift < -0.1 else "NEUTRAL",
        }


# Initialize analyzer
watchlist = ["Apple", "Tesla", "NVIDIA", "Microsoft", "Amazon", "Meta", "Google"]

analyzer = EventVelocityAnalyzer(watchlist)

print("MARKET EVENT VELOCITY ANALYZER")
print("=" * 70)
print(f"Monitoring: {', '.join(watchlist)}")
print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

# Calculate baselines
print("Calculating baselines...")
for entity in watchlist:
    baseline = analyzer.calculate_baseline(entity)
    print(f"  {entity}: {baseline['hourly_avg']:.1f} articles/hour (7-day avg)")

# Detect spikes
print("\nScanning for velocity spikes (3x+ baseline)...")
spikes = analyzer.detect_velocity_spikes(threshold_multiplier=3.0)

if spikes:
    print(f"\n🚨 DETECTED {len(spikes)} VELOCITY SPIKES:\n")

    for spike in spikes:
        print(f"{'='*60}")
        print(f"🔴 {spike['entity']}: {spike['velocity_ratio']:.1f}x normal velocity")
        print(f"   Baseline: {spike['baseline_hourly']:.1f}/hr → Current: {spike['current_hourly']:.1f}/hr")

        if spike["recent_articles"]:
            print("   Recent Headlines:")
            for article in spike["recent_articles"][:3]:
                sentiment_icon = "📈" if article["sentiment"] == "positive" else \
                                "📉" if article["sentiment"] == "negative" else "➡️"
                print(f"   {sentiment_icon} [{article['source']}] {article['title'][:50]}...")
else:
    print("\n✅ No significant velocity spikes detected.")

# Check sentiment shifts
print("\n" + "=" * 70)
print("SENTIMENT SHIFT ANALYSIS (2hr vs 24hr):\n")

for entity in watchlist:
    shift = analyzer.get_sentiment_shift(entity)
    if abs(shift["shift"]) > 0.1:
        emoji = "📈" if shift["shift_direction"] == "BULLISH" else \
               "📉" if shift["shift_direction"] == "BEARISH" else "➡️"
        print(f"{emoji} {entity}: {shift['shift_direction']} shift ({shift['shift']:+.3f})")
        print(f"   Historical: {shift['historical_score']:+.3f} → Recent: {shift['recent_score']:+.3f}")
```

### Cross-Asset Correlation Detector

```python
import requests
from datetime import datetime, timedelta
from itertools import combinations

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ASSET_GROUPS = {
    "mega_tech": ["Apple", "Microsoft", "Google", "Amazon", "Meta"],
    "semiconductors": ["NVIDIA", "AMD", "Intel", "TSMC", "Qualcomm"],
    "ev_sector": ["Tesla", "Rivian", "Lucid", "BYD", "NIO"],
    "financials": ["JPMorgan", "Goldman Sachs", "Morgan Stanley", "Bank of America"],
    "commodities": ["oil", "gold", "copper", "lithium"],
}

def get_co_mentions(entity1, entity2, hours=24):
    """Get articles mentioning both entities."""
    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": f"{entity1},{entity2}",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })

    return len(resp.json().get("results", []))

def calculate_correlation_strength(entity1, entity2, hours=24):
    """Calculate news correlation strength between two entities."""
    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    # Get individual counts
    resp1 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity1,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    count1 = len(resp1.json().get("results", []))

    resp2 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity2,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    count2 = len(resp2.json().get("results", []))

    # Get co-mentions
    co_mentions = get_co_mentions(entity1, entity2, hours)

    # Calculate Jaccard-like correlation
    union = count1 + count2 - co_mentions
    if union > 0:
        correlation = co_mentions / union
    else:
        correlation = 0

    return {
        "entity1": entity1,
        "entity2": entity2,
        "count1": count1,
        "count2": count2,
        "co_mentions": co_mentions,
        "correlation": correlation,
    }

def detect_correlation_anomalies(group_name, entities, baseline_hours=168, recent_hours=24):
    """Detect unusual correlation changes within an asset group."""
    anomalies = []

    pairs = list(combinations(entities, 2))

    for entity1, entity2 in pairs:
        # Get baseline correlation (7 days)
        baseline = calculate_correlation_strength(entity1, entity2, baseline_hours)

        # Get recent correlation (24 hours)
        recent = calculate_correlation_strength(entity1, entity2, recent_hours)

        # Detect significant changes
        if baseline["correlation"] > 0:
            change_ratio = recent["correlation"] / baseline["correlation"]
        else:
            change_ratio = recent["correlation"] * 10 if recent["correlation"] > 0 else 1

        if change_ratio > 2.0 or change_ratio < 0.5:
            anomalies.append({
                "pair": f"{entity1} - {entity2}",
                "baseline_correlation": baseline["correlation"],
                "recent_correlation": recent["correlation"],
                "change_ratio": change_ratio,
                "co_mentions_recent": recent["co_mentions"],
                "direction": "INCREASING" if change_ratio > 1 else "DECREASING",
            })

    return sorted(anomalies, key=lambda x: abs(x["change_ratio"] - 1), reverse=True)

print("CROSS-ASSET CORRELATION ANALYZER")
print("=" * 70)
print(f"Analysis Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

for group_name, entities in ASSET_GROUPS.items():
    print(f"\n{group_name.upper().replace('_', ' ')}")
    print("-" * 50)

    anomalies = detect_correlation_anomalies(group_name, entities)

    if anomalies:
        for anomaly in anomalies[:3]:
            direction_emoji = "📈" if anomaly["direction"] == "INCREASING" else "📉"
            print(f"\n  {direction_emoji} {anomaly['pair']}")
            print(f"     Correlation: {anomaly['baseline_correlation']:.3f} → {anomaly['recent_correlation']:.3f}")
            print(f"     Change: {anomaly['change_ratio']:.1f}x ({anomaly['direction']})")
            print(f"     Co-mentions (24h): {anomaly['co_mentions_recent']}")
    else:
        print("  No significant correlation changes detected.")
```

### Multi-Signal Event Scoring System

```python
import requests
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

TIER_1_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com"
TIER_2_SOURCES = "cnbc.com,marketwatch.com,seekingalpha.com,yahoo.com"

class MultiSignalEventScorer:
    """Score market events using multiple signal sources."""

    SIGNAL_WEIGHTS = {
        "breaking_news": 0.20,
        "source_authority": 0.15,
        "sentiment_extremity": 0.15,
        "mention_velocity": 0.15,
        "tier1_coverage": 0.15,
        "keyword_severity": 0.10,
        "cross_entity_spread": 0.10,
    }

    SEVERITY_KEYWORDS = {
        "critical": ["bankruptcy", "fraud", "crash", "collapse", "emergency", "halt"],
        "high": ["investigation", "lawsuit", "recall", "breach", "downgrade", "miss"],
        "medium": ["decline", "cut", "warning", "concern", "delay", "issue"],
    }

    def __init__(self, entity):
        self.entity = entity
        self.signals = {}

    def calculate_all_signals(self, minutes=60):
        """Calculate all signal components."""
        start = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"

        # 1. Breaking news signal
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "is_breaking": 1,
            "published_at.start": start,
            "per_page": 1,
        })
        breaking_count = len(resp.json().get("results", []))
        self.signals["breaking_news"] = min(1.0, breaking_count * 0.2)

        # 2. Source authority signal (% from tier-1 sources)
        resp_tier1 = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "source.domain": TIER_1_SOURCES,
            "published_at.start": start,
            "per_page": 1,
        })
        tier1_count = len(resp_tier1.json().get("results", []))

        resp_total = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "published_at.start": start,
            "per_page": 1,
        })
        total_count = len(resp_total.json().get("results", [])) or 1
        self.signals["source_authority"] = tier1_count / total_count
        self.signals["tier1_coverage"] = min(1.0, tier1_count * 0.1)

        # 3. Sentiment extremity signal
        resp_neg = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "sentiment.overall.polarity": "negative",
            "sentiment.overall.score.max": 0.3,  # Very negative
            "published_at.start": start,
            "per_page": 1,
        })
        extreme_neg = len(resp_neg.json().get("results", []))
        self.signals["sentiment_extremity"] = min(1.0, extreme_neg * 0.15)

        # 4. Mention velocity (normalized against baseline)
        baseline_resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "published_at.start": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "language.code": "en",
            "per_page": 1,
        })
        baseline_total = len(baseline_resp.json().get("results", []))
        baseline_hourly = baseline_total / (7 * 24)

        current_hourly = total_count * (60 / minutes)
        velocity_ratio = current_hourly / max(baseline_hourly, 0.1)
        self.signals["mention_velocity"] = min(1.0, (velocity_ratio - 1) * 0.2) if velocity_ratio > 1 else 0

        # 5. Keyword severity signal
        severity_score = 0
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": self.entity,
                "title": ",".join(keywords),
                "published_at.start": start,
                "per_page": 1,
            })
            count = len(resp.json().get("results", []))
            multiplier = {"critical": 1.0, "high": 0.6, "medium": 0.3}[severity]
            severity_score += count * multiplier * 0.1

        self.signals["keyword_severity"] = min(1.0, severity_score)

        # 6. Cross-entity spread (how many related entities are mentioned)
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": self.entity,
            "published_at.start": start,
            "per_page": 10,
        })
        articles = resp.json().get("results", [])
        related_entities = set()
        for article in articles:
            for entity in article.get("entities", []):
                if entity.get("type") == "organization" and entity["name"] != self.entity:
                    related_entities.add(entity["name"])

        self.signals["cross_entity_spread"] = min(1.0, len(related_entities) * 0.05)

        return self.signals

    def calculate_composite_score(self):
        """Calculate weighted composite score."""
        if not self.signals:
            self.calculate_all_signals()

        composite = sum(
            self.signals.get(signal, 0) * weight
            for signal, weight in self.SIGNAL_WEIGHTS.items()
        )

        return composite * 100  # Scale to 0-100

    def get_detailed_report(self):
        """Generate detailed scoring report."""
        if not self.signals:
            self.calculate_all_signals()

        composite = self.calculate_composite_score()

        return {
            "entity": self.entity,
            "composite_score": composite,
            "signals": self.signals,
            "signal_contributions": {
                signal: self.signals.get(signal, 0) * weight * 100
                for signal, weight in self.SIGNAL_WEIGHTS.items()
            },
            "alert_level": "CRITICAL" if composite >= 70 else \
                          "HIGH" if composite >= 50 else \
                          "MEDIUM" if composite >= 30 else "LOW",
        }


# Analyze multiple entities
entities = ["Apple", "Tesla", "NVIDIA", "Boeing", "Meta"]

print("MULTI-SIGNAL EVENT SCORING SYSTEM")
print("=" * 70)
print(f"Analysis Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Window: Last 60 minutes\n")

results = []

for entity in entities:
    scorer = MultiSignalEventScorer(entity)
    report = scorer.get_detailed_report()
    results.append(report)

# Sort by composite score
results.sort(key=lambda x: x["composite_score"], reverse=True)

print(f"{'Entity':<15} {'Score':>8} {'Level':<10} Top Contributing Signals")
print("-" * 70)

for r in results:
    # Get top 3 contributing signals
    top_signals = sorted(r["signal_contributions"].items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{s[0]}:{s[1]:.1f}" for s in top_signals)

    level_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[r["alert_level"]]

    print(f"{r['entity']:<15} {r['composite_score']:>7.1f} {level_emoji} {r['alert_level']:<8} {top_str}")

# Detailed view for high-scoring entities
high_score = [r for r in results if r["composite_score"] >= 30]

if high_score:
    print("\n" + "=" * 70)
    print("DETAILED SIGNAL BREAKDOWN (Score >= 30):")

    for r in high_score:
        print(f"\n{r['entity']} (Score: {r['composite_score']:.1f})")
        print("-" * 40)

        for signal, contribution in sorted(r["signal_contributions"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(contribution / 2)
            print(f"  {signal:<25}: {contribution:>5.1f} {bar}")
```

---

## JavaScript

### Real-Time Market Event Stream

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const TIER_1_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

const EVENT_CATEGORIES = {
  earnings: ["earnings", "beat", "miss", "EPS", "revenue", "guidance"],
  ma: ["merger", "acquisition", "buyout", "takeover", "deal"],
  regulatory: ["FDA", "SEC", "FTC", "investigation", "approval", "fine"],
  macro: ["Fed", "interest rate", "inflation", "recession", "GDP"],
};

class MarketEventStream {
  constructor(pollInterval = 60000) {
    this.pollInterval = pollInterval;
    this.seenEvents = new Set();
    this.handlers = [];
    this.baselines = {};
  }

  onEvent(handler) {
    this.handlers.push(handler);
  }

  async initializeBaselines(entities) {
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

    for (const entity of entities) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": entity,
        "published_at.start": sevenDaysAgo,
        "language.code": "en",
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      const total = data.results?.length || 0;

      this.baselines[entity] = {
        dailyAvg: total / 7,
        hourlyAvg: total / 168,
      };
    }

    console.log("Baselines initialized:", this.baselines);
  }

  classifyEvent(title) {
    const titleLower = title.toLowerCase();

    for (const [category, keywords] of Object.entries(EVENT_CATEGORIES)) {
      if (keywords.some(kw => titleLower.includes(kw.toLowerCase()))) {
        return category;
      }
    }

    return "general";
  }

  calculateImpactScore(article) {
    const sentiment = article.sentiment?.overall || {};
    const sourceRank = article.source?.rank?.opr || 0.5;
    const isBreaking = article.is_breaking || false;

    // Multi-factor scoring
    let score = 0;

    // Source authority (0-30)
    score += sourceRank * 30;

    // Sentiment extremity (0-25)
    const sentimentScore = sentiment.score || 0.5;
    const sentimentExtremity = Math.abs(sentimentScore - 0.5) * 2;
    score += sentimentExtremity * 25;

    // Breaking news bonus (0-25)
    if (isBreaking) score += 25;

    // Category severity (0-20)
    const category = this.classifyEvent(article.title);
    const categorySeverity = {
      regulatory: 20,
      ma: 18,
      earnings: 15,
      macro: 12,
      general: 5,
    };
    score += categorySeverity[category] || 5;

    return Math.min(100, score);
  }

  async poll() {
    const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
      api_key: API_KEY,
      is_breaking: "1",
      "source.domain": TIER_1_SOURCES,
      "published_at.start": tenMinutesAgo,
      "language.code": "en",
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "30",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();

    for (const article of data.results || []) {
      if (this.seenEvents.has(article.id)) continue;
      this.seenEvents.add(article.id);

      const impactScore = this.calculateImpactScore(article);
      const category = this.classifyEvent(article.title);

      const event = {
        id: article.id,
        title: article.title,
        source: article.source.domain,
        publishedAt: article.published_at,
        category,
        impactScore,
        sentiment: article.sentiment?.overall?.polarity || "neutral",
        tickers: (article.entities || [])
          .filter(e => e.type === "organization")
          .map(e => e.name)
          .slice(0, 5),
        url: article.href,
        alertLevel: impactScore >= 70 ? "CRITICAL" :
                    impactScore >= 50 ? "HIGH" :
                    impactScore >= 30 ? "MEDIUM" : "LOW",
      };

      // Only emit events above threshold
      if (impactScore >= 25) {
        this.handlers.forEach(h => h(event));
      }
    }
  }

  async start(entities = []) {
    console.log("MARKET EVENT STREAM");
    console.log("=".repeat(50));
    console.log(`Poll interval: ${this.pollInterval / 1000}s`);

    if (entities.length > 0) {
      console.log("Initializing baselines...");
      await this.initializeBaselines(entities);
    }

    console.log("Starting stream...\n");

    await this.poll();
    setInterval(() => this.poll(), this.pollInterval);
  }
}

// Initialize and run
const stream = new MarketEventStream(60000);

stream.onEvent((event) => {
  const levelEmoji = {
    CRITICAL: "🔴",
    HIGH: "🟠",
    MEDIUM: "🟡",
    LOW: "🟢",
  }[event.alertLevel];

  console.log(`\n${levelEmoji} [${event.category.toUpperCase()}] Impact: ${event.impactScore.toFixed(0)}/100`);
  console.log(`   ${event.title}`);
  console.log(`   Source: ${event.source} | Sentiment: ${event.sentiment}`);
  if (event.tickers.length > 0) {
    console.log(`   Tickers: ${event.tickers.join(", ")}`);
  }
});

stream.start(["Apple", "Tesla", "NVIDIA", "Microsoft"]);
```

---

## PHP

### Market Event Detection Engine

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$tier1Sources = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

$eventCategories = [
    "earnings"   => ["earnings", "beat", "miss", "EPS", "revenue"],
    "ma"         => ["merger", "acquisition", "buyout", "takeover"],
    "regulatory" => ["FDA", "SEC", "investigation", "approval", "fine"],
    "macro"      => ["Fed", "interest rate", "inflation", "recession"],
];

function classifyEvent(string $title): string
{
    global $eventCategories;

    $titleLower = strtolower($title);

    foreach ($eventCategories as $category => $keywords) {
        foreach ($keywords as $kw) {
            if (stripos($titleLower, $kw) !== false) {
                return $category;
            }
        }
    }

    return "general";
}

function calculateImpactScore(array $article): float
{
    $sentiment = $article["sentiment"]["overall"] ?? [];
    $sourceRank = $article["source"]["rank"]["opr"] ?? 0.5;
    $isBreaking = !empty($article["is_breaking"]);

    $score = 0;

    // Source authority (0-30)
    $score += $sourceRank * 30;

    // Sentiment extremity (0-25)
    $sentimentScore = $sentiment["score"] ?? 0.5;
    $extremity = abs($sentimentScore - 0.5) * 2;
    $score += $extremity * 25;

    // Breaking bonus (0-25)
    if ($isBreaking) $score += 25;

    // Category severity (0-20)
    $category = classifyEvent($article["title"]);
    $severities = [
        "regulatory" => 20,
        "ma"         => 18,
        "earnings"   => 15,
        "macro"      => 12,
        "general"    => 5,
    ];
    $score += $severities[$category] ?? 5;

    return min(100, $score);
}

function detectMarketEvents(int $minutes = 30): array
{
    global $apiKey, $baseUrl, $tier1Sources;

    $start = (new DateTime("-{$minutes} minutes"))->format("c");

    $query = http_build_query([
        "api_key"            => $apiKey,
        "is_breaking"        => 1,
        "source.domain"      => $tier1Sources,
        "published_at.start" => $start,
        "language.code"      => "en",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "per_page"           => 30,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $events = [];

    foreach ($data["results"] ?? [] as $article) {
        $impactScore = calculateImpactScore($article);
        $category = classifyEvent($article["title"]);

        $events[] = [
            "title"        => $article["title"],
            "source"       => $article["source"]["domain"],
            "published_at" => $article["published_at"],
            "category"     => $category,
            "impact_score" => $impactScore,
            "sentiment"    => $article["sentiment"]["overall"]["polarity"] ?? "neutral",
            "tickers"      => array_slice(
                array_column(
                    array_filter($article["entities"] ?? [], fn($e) => $e["type"] === "organization"),
                    "name"
                ),
                0, 5
            ),
            "alert_level"  => match (true) {
                $impactScore >= 70 => "CRITICAL",
                $impactScore >= 50 => "HIGH",
                $impactScore >= 30 => "MEDIUM",
                default            => "LOW",
            },
        ];
    }

    usort($events, fn($a, $b) => $b["impact_score"] <=> $a["impact_score"]);

    return $events;
}

echo "MARKET EVENT DETECTION ENGINE\n";
echo str_repeat("=", 70) . "\n";
echo "Scan Time: " . date("Y-m-d H:i:s T") . "\n";
echo "Window: 30 minutes\n\n";

$events = detectMarketEvents(30);
$significant = array_filter($events, fn($e) => $e["impact_score"] >= 25);

echo "Detected " . count($significant) . " significant events:\n\n";

foreach (array_slice($significant, 0, 10) as $i => $event) {
    $emoji = match ($event["alert_level"]) {
        "CRITICAL" => "🔴",
        "HIGH"     => "🟠",
        "MEDIUM"   => "🟡",
        default    => "🟢",
    };

    $num = $i + 1;
    echo "{$num}. {$emoji} [{$event['category']}] Impact: " . round($event["impact_score"]) . "/100\n";
    echo "   " . substr($event["title"], 0, 65) . "...\n";
    echo "   Source: {$event['source']} | Sentiment: {$event['sentiment']}\n";

    if (!empty($event["tickers"])) {
        echo "   Tickers: " . implode(", ", $event["tickers"]) . "\n";
    }

    echo "\n";
}
```
