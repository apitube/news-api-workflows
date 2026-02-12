# Crisis Management — Code Examples

Detailed examples for building enterprise crisis management systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Multi-Brand Crisis Dashboard with Escalation Tracking

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

BRANDS = {
    "Tesla": "organization",
    "Elon Musk": "person",
    "SpaceX": "organization",
    "Twitter": "organization",
}

CRISIS_KEYWORDS = [
    "lawsuit", "scandal", "fraud", "recall", "investigation",
    "bankruptcy", "layoff", "resignation", "fired", "arrested",
    "SEC", "lawsuit", "settlement", "fine", "penalty"
]

TIER_1_SOURCES = "reuters.com,bloomberg.com,nytimes.com,wsj.com,ft.com,bbc.com,cnn.com"
TIER_2_SOURCES = "techcrunch.com,theverge.com,wired.com,arstechnica.com,engadget.com"

def get_crisis_metrics(entity_name, entity_type, hours=24):
    """Collect comprehensive crisis metrics for an entity."""
    start_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    metrics = {
        "total_mentions": 0,
        "negative_mentions": 0,
        "positive_mentions": 0,
        "breaking_negative": 0,
        "tier1_negative": 0,
        "tier2_negative": 0,
        "crisis_keyword_hits": 0,
        "high_severity_negative": 0,  # sentiment score < 0.3
    }

    # Total mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["total_mentions"] = resp.json().get("total_results", 0)

    # Negative mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "sentiment.overall.polarity": "negative",
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["negative_mentions"] = resp.json().get("total_results", 0)

    # Positive mentions
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "sentiment.overall.polarity": "positive",
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["positive_mentions"] = resp.json().get("total_results", 0)

    # Breaking negative
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "is_breaking": "true",
        "sentiment.overall.polarity": "negative",
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["breaking_negative"] = resp.json().get("total_results", 0)

    # Tier 1 negative
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "sentiment.overall.polarity": "negative",
        "source.domain": TIER_1_SOURCES,
        "published_at.start": start_time,
        "per_page": 1,
    })
    metrics["tier1_negative"] = resp.json().get("total_results", 0)

    # Tier 2 negative
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "sentiment.overall.polarity": "negative",
        "source.domain": TIER_2_SOURCES,
        "published_at.start": start_time,
        "per_page": 1,
    })
    metrics["tier2_negative"] = resp.json().get("total_results", 0)

    # Crisis keywords
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "title": ",".join(CRISIS_KEYWORDS),
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["crisis_keyword_hits"] = resp.json().get("total_results", 0)

    # High severity negative (score < 0.3)
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "entity.name": entity_name,
        "entity.type": entity_type,
        "sentiment.overall.polarity": "negative",
        "sentiment.overall.score.max": 0.3,
        "published_at.start": start_time,
        "language": "en",
        "per_page": 1,
    })
    metrics["high_severity_negative"] = resp.json().get("total_results", 0)

    return metrics

def calculate_crisis_score(metrics):
    """Calculate weighted crisis score (0-100)."""
    total = metrics["total_mentions"] or 1

    # Base negative ratio (0-30 points)
    neg_ratio = metrics["negative_mentions"] / total
    base_score = neg_ratio * 30

    # Breaking news multiplier (0-25 points)
    breaking_score = min(25, metrics["breaking_negative"] * 5)

    # Tier 1 media coverage (0-20 points)
    tier1_score = min(20, metrics["tier1_negative"] * 4)

    # Crisis keywords (0-15 points)
    keyword_score = min(15, metrics["crisis_keyword_hits"] * 3)

    # High severity negative (0-10 points)
    severity_score = min(10, metrics["high_severity_negative"] * 2)

    return min(100, base_score + breaking_score + tier1_score + keyword_score + severity_score)

def get_status_emoji(score):
    if score >= 70:
        return "🔴 CRITICAL"
    elif score >= 50:
        return "🟠 HIGH"
    elif score >= 30:
        return "🟡 MEDIUM"
    elif score >= 10:
        return "🟢 LOW"
    return "⚪ NORMAL"

print("=" * 80)
print("MULTI-BRAND CRISIS DASHBOARD")
print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)

for entity_name, entity_type in BRANDS.items():
    metrics = get_crisis_metrics(entity_name, entity_type, hours=24)
    score = calculate_crisis_score(metrics)
    status = get_status_emoji(score)

    print(f"\n{entity_name} ({entity_type})")
    print("-" * 40)
    print(f"  Crisis Score: {score:.0f}/100  {status}")
    print(f"  Total Mentions (24h): {metrics['total_mentions']}")
    print(f"  Negative: {metrics['negative_mentions']} | Positive: {metrics['positive_mentions']}")
    print(f"  Breaking Negative: {metrics['breaking_negative']}")
    print(f"  Tier-1 Negative: {metrics['tier1_negative']}")
    print(f"  Crisis Keywords: {metrics['crisis_keyword_hits']}")
    print(f"  High Severity: {metrics['high_severity_negative']}")
```

### Crisis Timeline Reconstruction

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def reconstruct_crisis_timeline(entity_name, crisis_start_date, days=7):
    """Reconstruct the timeline of a crisis event."""

    timeline = []
    current_date = datetime.fromisoformat(crisis_start_date.replace("Z", ""))

    for day in range(days):
        day_start = current_date + timedelta(days=day)
        day_end = day_start + timedelta(days=1)

        day_metrics = {
            "date": day_start.strftime("%Y-%m-%d"),
            "total": 0,
            "negative": 0,
            "positive": 0,
            "neutral": 0,
            "breaking": 0,
            "top_sources": [],
            "key_headlines": [],
        }

        # Get daily totals by sentiment
        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": "organization",
                "sentiment.overall.polarity": polarity,
                "published_at.start": day_start.isoformat() + "Z",
                "published_at.end": day_end.isoformat() + "Z",
                "language": "en",
                "per_page": 1,
            })
            count = resp.json().get("total_results", 0)
            day_metrics[polarity] = count
            day_metrics["total"] += count

        # Get breaking news count
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity_name,
            "is_breaking": "true",
            "published_at.start": day_start.isoformat() + "Z",
            "published_at.end": day_end.isoformat() + "Z",
            "per_page": 1,
        })
        day_metrics["breaking"] = resp.json().get("total_results", 0)

        # Get key headlines
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity_name,
            "entity.type": "organization",
            "sentiment.overall.polarity": "negative",
            "published_at.start": day_start.isoformat() + "Z",
            "published_at.end": day_end.isoformat() + "Z",
            "source.rank.opr.min": 0.6,
            "sort.by": "source.rank.opr",
            "sort.order": "desc",
            "per_page": 5,
        })
        for article in resp.json().get("results", []):
            day_metrics["key_headlines"].append({
                "title": article["title"],
                "source": article["source"]["domain"],
                "time": article["published_at"],
            })

        timeline.append(day_metrics)

    return timeline

# Reconstruct timeline
entity = "Boeing"
crisis_date = "2024-01-05"  # Example: Boeing door plug incident

print(f"Crisis Timeline: {entity}")
print(f"Starting from: {crisis_date}")
print("=" * 80)

timeline = reconstruct_crisis_timeline(entity, crisis_date, days=7)

for day in timeline:
    neg_pct = (day["negative"] / day["total"] * 100) if day["total"] > 0 else 0

    print(f"\n{day['date']} — Total: {day['total']} articles")
    print(f"  Sentiment: +{day['positive']} / -{day['negative']} / ={day['neutral']} ({neg_pct:.1f}% negative)")
    print(f"  Breaking: {day['breaking']}")

    if day["key_headlines"]:
        print("  Key Headlines:")
        for h in day["key_headlines"][:3]:
            print(f"    • [{h['source']}] {h['title'][:70]}...")
```

### Real-Time Crisis Alert System with Webhooks

```python
import requests
import time
import json
from datetime import datetime, timedelta
from typing import Callable

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class CrisisAlertSystem:
    def __init__(self, entities: dict, alert_callback: Callable):
        self.entities = entities
        self.alert_callback = alert_callback
        self.last_check = {}
        self.baseline_metrics = {}

    def initialize_baselines(self, hours=168):  # 7 days
        """Establish baseline metrics for anomaly detection."""
        start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        for entity_name, entity_type in self.entities.items():
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": entity_type,
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            total = resp.json().get("total_results", 0)

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": entity_type,
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            negative = resp.json().get("total_results", 0)

            self.baseline_metrics[entity_name] = {
                "avg_daily_mentions": total / (hours / 24),
                "avg_daily_negative": negative / (hours / 24),
                "neg_ratio": negative / total if total > 0 else 0,
            }

            self.last_check[entity_name] = datetime.utcnow()

        print("Baselines initialized:")
        for name, metrics in self.baseline_metrics.items():
            print(f"  {name}: {metrics['avg_daily_mentions']:.1f} mentions/day, "
                  f"{metrics['neg_ratio']*100:.1f}% negative")

    def check_for_alerts(self):
        """Check current metrics against baselines."""
        alerts = []

        for entity_name, entity_type in self.entities.items():
            since = self.last_check.get(entity_name, datetime.utcnow() - timedelta(hours=1))
            start = since.isoformat() + "Z"

            # Get recent mentions
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": entity_type,
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            recent_total = resp.json().get("total_results", 0)

            # Get recent negative
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": entity_type,
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            recent_negative = resp.json().get("total_results", 0)

            # Get breaking negative
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": entity_name,
                "entity.type": entity_type,
                "is_breaking": "true",
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "per_page": 5,
            })
            breaking_data = resp.json()
            breaking_negative = breaking_data.get("total_results", 0)
            breaking_articles = breaking_data.get("results", [])

            # Calculate anomaly scores
            baseline = self.baseline_metrics[entity_name]
            hours_elapsed = (datetime.utcnow() - since).total_seconds() / 3600
            expected_mentions = baseline["avg_daily_mentions"] * (hours_elapsed / 24)
            expected_negative = baseline["avg_daily_negative"] * (hours_elapsed / 24)

            mention_multiplier = recent_total / max(expected_mentions, 1)
            negative_multiplier = recent_negative / max(expected_negative, 1)

            alert = None

            # Alert conditions
            if breaking_negative > 0:
                alert = {
                    "level": "CRITICAL",
                    "entity": entity_name,
                    "reason": f"Breaking negative news detected ({breaking_negative} articles)",
                    "articles": breaking_articles[:3],
                }
            elif negative_multiplier > 3:
                alert = {
                    "level": "HIGH",
                    "entity": entity_name,
                    "reason": f"Negative coverage spike ({negative_multiplier:.1f}x baseline)",
                    "recent_negative": recent_negative,
                }
            elif mention_multiplier > 5:
                alert = {
                    "level": "MEDIUM",
                    "entity": entity_name,
                    "reason": f"Mention volume spike ({mention_multiplier:.1f}x baseline)",
                    "recent_total": recent_total,
                }

            if alert:
                alerts.append(alert)
                self.alert_callback(alert)

            self.last_check[entity_name] = datetime.utcnow()

        return alerts

    def run(self, interval_seconds=300):
        """Run continuous monitoring loop."""
        print(f"Starting crisis monitoring (checking every {interval_seconds}s)...")

        while True:
            try:
                alerts = self.check_for_alerts()
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

                if alerts:
                    print(f"[{timestamp}] {len(alerts)} alert(s) triggered")
                else:
                    print(f"[{timestamp}] All clear")

                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("Monitoring stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)

# Example webhook callback
def send_alert(alert):
    """Send alert to webhook/Slack/email."""
    print(f"\n{'='*60}")
    print(f"ALERT [{alert['level']}]: {alert['entity']}")
    print(f"Reason: {alert['reason']}")
    if "articles" in alert:
        print("Headlines:")
        for a in alert["articles"]:
            print(f"  • {a['title'][:60]}...")
    print(f"{'='*60}\n")

    # Example: Send to Slack webhook
    # requests.post(SLACK_WEBHOOK_URL, json={
    #     "text": f"*{alert['level']} ALERT*: {alert['entity']}\n{alert['reason']}"
    # })

# Initialize and run
entities = {
    "Apple": "organization",
    "Google": "organization",
    "Microsoft": "organization",
}

system = CrisisAlertSystem(entities, send_alert)
system.initialize_baselines(hours=168)
system.run(interval_seconds=300)
```

### Cross-Language Crisis Propagation Tracker

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
}

def track_crisis_propagation(entity_name, hours=72):
    """Track how crisis coverage spreads across languages."""

    results = {}
    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    for lang_code, lang_name in LANGUAGES.items():
        # Get negative coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": entity_name,
            "entity.type": "organization",
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language": lang_code,
            "sort.by": "published_at",
            "sort.order": "asc",
            "per_page": 5,
        })
        data = resp.json()

        first_article = data["results"][0] if data.get("results") else None

        results[lang_code] = {
            "language": lang_name,
            "negative_count": data.get("total_results", 0),
            "first_mention": first_article["published_at"] if first_article else None,
            "first_headline": first_article["title"] if first_article else None,
            "first_source": first_article["source"]["domain"] if first_article else None,
        }

    return results

# Track propagation
entity = "Volkswagen"
print(f"Crisis Propagation Analysis: {entity}")
print(f"Last {72} hours")
print("=" * 80)

propagation = track_crisis_propagation(entity, hours=72)

# Sort by first mention time
sorted_langs = sorted(
    propagation.items(),
    key=lambda x: x[1]["first_mention"] or "9999",
)

print(f"\n{'Language':<12} {'Count':>8} {'First Mention':<22} {'First Source':<20}")
print("-" * 70)

for lang_code, data in sorted_langs:
    if data["negative_count"] > 0:
        first_time = data["first_mention"][:19] if data["first_mention"] else "N/A"
        source = (data["first_source"] or "N/A")[:18]
        print(f"{data['language']:<12} {data['negative_count']:>8} {first_time:<22} {source:<20}")

print("\nPropagation Timeline:")
for i, (lang_code, data) in enumerate(sorted_langs):
    if data["first_mention"]:
        time_str = data["first_mention"][:19]
        headline = (data["first_headline"] or "")[:50]
        print(f"  {i+1}. [{data['language']}] {time_str}")
        print(f"     {headline}...")
```

---

## JavaScript

### Crisis Command Center Dashboard

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const CRISIS_KEYWORDS = [
  "lawsuit", "scandal", "fraud", "recall", "investigation",
  "bankruptcy", "layoff", "resignation", "SEC", "fine"
];

const TIER_1_SOURCES = "reuters.com,bloomberg.com,nytimes.com,wsj.com,ft.com,bbc.com";

class CrisisCommandCenter {
  constructor(entities) {
    this.entities = entities;
    this.metrics = {};
  }

  async fetchMetrics(entityName, entityType, hours = 24) {
    const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

    const metrics = {
      total: 0,
      negative: 0,
      positive: 0,
      breaking: 0,
      tier1Negative: 0,
      crisisKeywords: 0,
      highSeverity: 0,
    };

    // Parallel fetch all metrics
    const requests = [
      // Total mentions
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        "entity.type": entityType,
        "published_at.start": start,
        language: "en",
        per_page: "1",
      })}`),
      // Negative
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        "entity.type": entityType,
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        language: "en",
        per_page: "1",
      })}`),
      // Positive
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        "entity.type": entityType,
        "sentiment.overall.polarity": "positive",
        "published_at.start": start,
        language: "en",
        per_page: "1",
      })}`),
      // Breaking
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        "entity.type": entityType,
        is_breaking: "true",
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        per_page: "1",
      })}`),
      // Tier 1 negative
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        "sentiment.overall.polarity": "negative",
        "source.domain": TIER_1_SOURCES,
        "published_at.start": start,
        per_page: "1",
      })}`),
      // Crisis keywords
      fetch(`${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "entity.name": entityName,
        title: CRISIS_KEYWORDS.join(","),
        "published_at.start": start,
        per_page: "1",
      })}`),
    ];

    const responses = await Promise.all(requests);
    const data = await Promise.all(responses.map(r => r.json()));

    metrics.total = data[0].total_results || 0;
    metrics.negative = data[1].total_results || 0;
    metrics.positive = data[2].total_results || 0;
    metrics.breaking = data[3].total_results || 0;
    metrics.tier1Negative = data[4].total_results || 0;
    metrics.crisisKeywords = data[5].total_results || 0;

    return metrics;
  }

  calculateScore(metrics) {
    const total = metrics.total || 1;
    const negRatio = metrics.negative / total;

    return Math.min(100, Math.round(
      negRatio * 30 +
      Math.min(25, metrics.breaking * 5) +
      Math.min(20, metrics.tier1Negative * 4) +
      Math.min(15, metrics.crisisKeywords * 3)
    ));
  }

  getStatus(score) {
    if (score >= 70) return { emoji: "🔴", level: "CRITICAL" };
    if (score >= 50) return { emoji: "🟠", level: "HIGH" };
    if (score >= 30) return { emoji: "🟡", level: "MEDIUM" };
    if (score >= 10) return { emoji: "🟢", level: "LOW" };
    return { emoji: "⚪", level: "NORMAL" };
  }

  async generateReport() {
    console.log("=".repeat(70));
    console.log("CRISIS COMMAND CENTER");
    console.log(`Generated: ${new Date().toISOString()}`);
    console.log("=".repeat(70));

    for (const [entityName, entityType] of Object.entries(this.entities)) {
      const metrics = await this.fetchMetrics(entityName, entityType);
      const score = this.calculateScore(metrics);
      const status = this.getStatus(score);

      this.metrics[entityName] = { metrics, score, status };

      console.log(`\n${entityName} (${entityType})`);
      console.log("-".repeat(40));
      console.log(`  Score: ${score}/100  ${status.emoji} ${status.level}`);
      console.log(`  Total: ${metrics.total} | +${metrics.positive} / -${metrics.negative}`);
      console.log(`  Breaking: ${metrics.breaking} | Tier-1: ${metrics.tier1Negative}`);
      console.log(`  Crisis Keywords: ${metrics.crisisKeywords}`);
    }

    return this.metrics;
  }
}

// Run dashboard
const entities = {
  "Tesla": "organization",
  "Apple": "organization",
  "Meta": "organization",
  "Amazon": "organization",
};

const center = new CrisisCommandCenter(entities);
center.generateReport();
```

### Real-Time Crisis Stream Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class CrisisStreamMonitor {
  constructor(entities, pollInterval = 60000) {
    this.entities = entities;
    this.pollInterval = pollInterval;
    this.seenArticles = new Set();
    this.alertHandlers = [];
  }

  onAlert(handler) {
    this.alertHandlers.push(handler);
  }

  async checkNewArticles(entityName, entityType) {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();

    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": entityName,
      "entity.type": entityType,
      "sentiment.overall.polarity": "negative",
      "published_at.start": fiveMinutesAgo,
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "20",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const newArticles = [];

    for (const article of data.results || []) {
      if (!this.seenArticles.has(article.id)) {
        this.seenArticles.add(article.id);
        newArticles.push(article);
      }
    }

    return newArticles;
  }

  async checkBreakingNews(entityName, entityType) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": entityName,
      "entity.type": entityType,
      is_breaking: "true",
      "sentiment.overall.polarity": "negative",
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "5",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const newBreaking = [];

    for (const article of data.results || []) {
      if (!this.seenArticles.has(article.id)) {
        this.seenArticles.add(article.id);
        newBreaking.push(article);
      }
    }

    return newBreaking;
  }

  async poll() {
    const timestamp = new Date().toISOString();
    console.log(`\n[${timestamp}] Polling...`);

    for (const [entityName, entityType] of Object.entries(this.entities)) {
      // Check breaking news first
      const breaking = await this.checkBreakingNews(entityName, entityType);

      for (const article of breaking) {
        const alert = {
          level: "CRITICAL",
          type: "BREAKING_NEGATIVE",
          entity: entityName,
          article: {
            title: article.title,
            source: article.source.domain,
            url: article.href,
            sentiment: article.sentiment?.overall,
          },
        };

        this.alertHandlers.forEach(h => h(alert));
      }

      // Check other negative news
      const negative = await this.checkNewArticles(entityName, entityType);

      if (negative.length > 3) {
        const alert = {
          level: "HIGH",
          type: "NEGATIVE_SURGE",
          entity: entityName,
          count: negative.length,
          articles: negative.slice(0, 3).map(a => ({
            title: a.title,
            source: a.source.domain,
          })),
        };

        this.alertHandlers.forEach(h => h(alert));
      }
    }
  }

  async start() {
    console.log("Starting Crisis Stream Monitor...");
    console.log(`Watching: ${Object.keys(this.entities).join(", ")}`);
    console.log(`Poll interval: ${this.pollInterval / 1000}s`);

    // Initial poll
    await this.poll();

    // Continuous polling
    setInterval(() => this.poll(), this.pollInterval);
  }
}

// Usage
const monitor = new CrisisStreamMonitor({
  "Tesla": "organization",
  "Apple": "organization",
}, 60000);

monitor.onAlert((alert) => {
  console.log("\n" + "!".repeat(60));
  console.log(`ALERT [${alert.level}]: ${alert.entity}`);
  console.log(`Type: ${alert.type}`);

  if (alert.article) {
    console.log(`Headline: ${alert.article.title}`);
    console.log(`Source: ${alert.article.source}`);
  }

  if (alert.articles) {
    console.log(`${alert.count} negative articles detected:`);
    alert.articles.forEach(a => console.log(`  • [${a.source}] ${a.title}`));
  }

  console.log("!".repeat(60));
});

monitor.start();
```

---

## PHP

### Enterprise Crisis Dashboard

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$crisisKeywords = ["lawsuit", "scandal", "fraud", "recall", "investigation", "SEC", "fine"];
$tier1Sources   = "reuters.com,bloomberg.com,nytimes.com,wsj.com,ft.com";

function fetchMetrics(string $entity, string $type, int $hours = 24): array
{
    global $apiKey, $baseUrl, $crisisKeywords, $tier1Sources;

    $start = (new DateTime("-{$hours} hours"))->format("c");

    $metrics = [
        "total"           => 0,
        "negative"        => 0,
        "positive"        => 0,
        "breaking"        => 0,
        "tier1_negative"  => 0,
        "crisis_keywords" => 0,
    ];

    // Total
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $entity,
        "entity.type"        => $type,
        "published_at.start" => $start,
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["total"] = $data["total_results"] ?? 0;

    // Negative
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $entity,
        "entity.type"                => $type,
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "language"                   => "en",
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["negative"] = $data["total_results"] ?? 0;

    // Positive
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $entity,
        "entity.type"                => $type,
        "sentiment.overall.polarity" => "positive",
        "published_at.start"         => $start,
        "language"                   => "en",
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["positive"] = $data["total_results"] ?? 0;

    // Breaking negative
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $entity,
        "entity.type"                => $type,
        "is_breaking"                => "true",
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["breaking"] = $data["total_results"] ?? 0;

    // Tier 1 negative
    $query = http_build_query([
        "api_key"                    => $apiKey,
        "entity.name"                => $entity,
        "sentiment.overall.polarity" => "negative",
        "source.domain"              => $tier1Sources,
        "published_at.start"         => $start,
        "per_page"                   => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["tier1_negative"] = $data["total_results"] ?? 0;

    // Crisis keywords
    $query = http_build_query([
        "api_key"            => $apiKey,
        "entity.name"        => $entity,
        "title"              => implode(",", $crisisKeywords),
        "published_at.start" => $start,
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $metrics["crisis_keywords"] = $data["total_results"] ?? 0;

    return $metrics;
}

function calculateScore(array $metrics): int
{
    $total    = $metrics["total"] ?: 1;
    $negRatio = $metrics["negative"] / $total;

    $score = $negRatio * 30
           + min(25, $metrics["breaking"] * 5)
           + min(20, $metrics["tier1_negative"] * 4)
           + min(15, $metrics["crisis_keywords"] * 3);

    return (int) min(100, $score);
}

function getStatus(int $score): array
{
    if ($score >= 70) return ["emoji" => "🔴", "level" => "CRITICAL"];
    if ($score >= 50) return ["emoji" => "🟠", "level" => "HIGH"];
    if ($score >= 30) return ["emoji" => "🟡", "level" => "MEDIUM"];
    if ($score >= 10) return ["emoji" => "🟢", "level" => "LOW"];
    return ["emoji" => "⚪", "level" => "NORMAL"];
}

// Entities to monitor
$entities = [
    "Tesla"     => "organization",
    "Apple"     => "organization",
    "Meta"      => "organization",
    "Microsoft" => "organization",
];

echo str_repeat("=", 70) . "\n";
echo "ENTERPRISE CRISIS DASHBOARD\n";
echo "Generated: " . date("Y-m-d H:i:s T") . "\n";
echo str_repeat("=", 70) . "\n";

foreach ($entities as $entity => $type) {
    $metrics = fetchMetrics($entity, $type, 24);
    $score   = calculateScore($metrics);
    $status  = getStatus($score);

    echo "\n{$entity} ({$type})\n";
    echo str_repeat("-", 40) . "\n";
    echo "  Score: {$score}/100  {$status['emoji']} {$status['level']}\n";
    echo "  Total: {$metrics['total']} | +{$metrics['positive']} / -{$metrics['negative']}\n";
    echo "  Breaking: {$metrics['breaking']} | Tier-1: {$metrics['tier1_negative']}\n";
    echo "  Crisis Keywords: {$metrics['crisis_keywords']}\n";
}
```

### Crisis Recovery Tracker

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function trackRecovery(string $entity, string $crisisDate, int $days = 30): array
{
    global $apiKey, $baseUrl;

    $timeline = [];
    $start    = new DateTime($crisisDate);

    for ($day = 0; $day < $days; $day++) {
        $dayStart = (clone $start)->modify("+{$day} days");
        $dayEnd   = (clone $dayStart)->modify("+1 day");

        $dayMetrics = [
            "date"     => $dayStart->format("Y-m-d"),
            "day"      => $day + 1,
            "total"    => 0,
            "positive" => 0,
            "negative" => 0,
            "neutral"  => 0,
        ];

        foreach (["positive", "negative", "neutral"] as $polarity) {
            $query = http_build_query([
                "api_key"                    => $apiKey,
                "entity.name"                => $entity,
                "entity.type"                => "organization",
                "sentiment.overall.polarity" => $polarity,
                "published_at.start"         => $dayStart->format("c"),
                "published_at.end"           => $dayEnd->format("c"),
                "language"                   => "en",
                "per_page"                   => 1,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            $count = $data["total_results"] ?? 0;
            $dayMetrics[$polarity] = $count;
            $dayMetrics["total"] += $count;
        }

        $timeline[] = $dayMetrics;
    }

    return $timeline;
}

function analyzeRecovery(array $timeline): array
{
    $analysis = [
        "peak_negative_day"    => null,
        "peak_negative_count"  => 0,
        "recovery_day"         => null,
        "sentiment_normalized" => false,
        "trend"                => [],
    ];

    $baselineNegRatio = null;

    foreach ($timeline as $i => $day) {
        $total    = $day["total"] ?: 1;
        $negRatio = $day["negative"] / $total;

        if ($i < 3) {
            $baselineNegRatio = $baselineNegRatio ?? $negRatio;
        }

        // Track peak negative
        if ($day["negative"] > $analysis["peak_negative_count"]) {
            $analysis["peak_negative_day"]   = $day["date"];
            $analysis["peak_negative_count"] = $day["negative"];
        }

        // Detect recovery (negative ratio returns to below 30%)
        if ($negRatio < 0.3 && $analysis["recovery_day"] === null && $i > 7) {
            $analysis["recovery_day"]         = $day["date"];
            $analysis["recovery_day_number"]  = $day["day"];
            $analysis["sentiment_normalized"] = true;
        }

        $analysis["trend"][] = [
            "date"      => $day["date"],
            "neg_ratio" => round($negRatio * 100, 1),
        ];
    }

    return $analysis;
}

// Track recovery for a company
$entity     = "Boeing";
$crisisDate = "2024-01-05";  // Door plug incident

echo "Crisis Recovery Analysis: {$entity}\n";
echo "Crisis Start Date: {$crisisDate}\n";
echo str_repeat("=", 70) . "\n\n";

$timeline = trackRecovery($entity, $crisisDate, 30);
$analysis = analyzeRecovery($timeline);

echo "Peak Negative Coverage:\n";
echo "  Date: {$analysis['peak_negative_day']}\n";
echo "  Count: {$analysis['peak_negative_count']} articles\n\n";

if ($analysis["recovery_day"]) {
    echo "Recovery Detected:\n";
    echo "  Date: {$analysis['recovery_day']} (Day {$analysis['recovery_day_number']})\n";
} else {
    echo "Recovery: NOT YET DETECTED\n";
}

echo "\nSentiment Trend (% Negative):\n";
echo str_repeat("-", 50) . "\n";

foreach ($analysis["trend"] as $point) {
    $bar = str_repeat("█", (int) ($point["neg_ratio"] / 2));
    printf("  %s: %5.1f%% %s\n", $point["date"], $point["neg_ratio"], $bar);
}
```
