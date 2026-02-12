# M&A Deal Intelligence

Workflow for tracking merger and acquisition activity, detecting deal rumors, analyzing transaction sentiment, monitoring regulatory approval progress, and assessing deal success probability using the [APITube News API](https://apitube.io).

## Overview

The **M&A Deal Intelligence** workflow provides comprehensive M&A coverage analysis by detecting deal rumors before announcements, tracking announced transactions through completion, monitoring regulatory and antitrust developments, analyzing deal sentiment and market reaction, and identifying potential acquisition targets. Combines rumor detection, timeline tracking, sentiment analysis, and regulatory monitoring. Ideal for investment banks, private equity, corporate development, and financial analysts.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by company involved in M&A.                                   |
| `title`                       | string  | Filter by M&A-related keywords.                                      |
| `sentiment.overall.polarity`  | string  | Filter by sentiment: `positive`, `negative`, `neutral`.             |
| `source.rank.opr.min`         | number  | Minimum source authority (0.0–1.0).                                 |
| `published_at.start`          | string  | Start date (ISO 8601 or `YYYY-MM-DD`).                             |
| `published_at.end`            | string  | End date (ISO 8601 or `YYYY-MM-DD`).                               |
| `language`                    | string  | Filter by language code.                                             |
| `sort.by`                     | string  | Sort field: `published_at`.                                          |
| `sort.order`                  | string  | Sort direction: `asc` or `desc`.                                    |
| `per_page`                    | integer | Number of results per page.                                          |

## Quick Start

### cURL

```bash
# Track M&A rumors
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=acquisition,merger,takeover,buyout,deal&source.rank.opr.min=0.7&language=en&per_page=30"

# Monitor specific company M&A activity
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Microsoft&title=acquire,acquisition,merger,buy&language=en&per_page=30"

# Track regulatory approvals
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&title=antitrust,FTC,DOJ,regulatory approval,merger approval&language=en&per_page=30"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

# M&A keyword taxonomy
MA_TAXONOMY = {
    "rumor_signals": [
        "rumor", "reportedly", "considering", "exploring", "in talks",
        "potential acquisition", "possible merger", "may acquire", "could buy",
        "sources say", "people familiar"
    ],
    "announcement_signals": [
        "announced", "confirms", "agrees to acquire", "to buy", "to merge",
        "deal signed", "acquisition agreement", "definitive agreement"
    ],
    "regulatory_signals": [
        "antitrust", "FTC", "DOJ", "regulatory approval", "EU approval",
        "competition authority", "merger review", "second request"
    ],
    "completion_signals": [
        "completed", "closed", "finalized", "deal closes",
        "acquisition complete", "merger complete"
    ],
    "failure_signals": [
        "terminated", "abandoned", "walks away", "deal collapses",
        "blocks merger", "rejected", "called off"
    ]
}

class MADealTracker:
    """Track M&A deals from rumor to completion."""

    def __init__(self):
        self.deals = {}
        self.rumors = []

    def detect_rumors(self, days=14, min_mentions=3):
        """Detect potential M&A rumors."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(MA_TAXONOMY["rumor_signals"]),
            "source.rank.opr.min": 0.6,
            "published_at.start": start,
            "language": "en",
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 100,
        })

        articles = resp.json().get("results", [])

        # Extract potential targets
        target_mentions = defaultdict(lambda: {"count": 0, "articles": [], "sentiment_sum": 0})

        for article in articles:
            entities = article.get("entities", [])
            sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)

            for entity in entities:
                if entity.get("type") == "organization":
                    name = entity.get("name")
                    target_mentions[name]["count"] += 1
                    target_mentions[name]["articles"].append(article)
                    target_mentions[name]["sentiment_sum"] += sentiment

        # Filter by minimum mentions
        rumors = []
        for company, data in target_mentions.items():
            if data["count"] >= min_mentions:
                rumors.append({
                    "company": company,
                    "mention_count": data["count"],
                    "avg_sentiment": data["sentiment_sum"] / data["count"],
                    "first_mention": min(a.get("published_at", "")[:10] for a in data["articles"]),
                    "latest_mention": max(a.get("published_at", "")[:10] for a in data["articles"]),
                    "sample_headlines": [a.get("title") for a in data["articles"][:3]]
                })

        self.rumors = sorted(rumors, key=lambda x: x["mention_count"], reverse=True)
        return self.rumors

    def track_deal(self, acquirer, target, announcement_date=None):
        """Track a specific M&A deal."""
        deal_key = f"{acquirer}_{target}"

        deal = {
            "acquirer": acquirer,
            "target": target,
            "announcement_date": announcement_date,
            "status": "tracking",
            "timeline": [],
            "coverage": {},
            "sentiment_history": [],
            "regulatory_status": "pending"
        }

        # Get deal coverage
        search_terms = f"{acquirer},{target}"

        for phase, keywords in MA_TAXONOMY.items():
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": f"{search_terms},{','.join(keywords[:5])}",
                "language": "en",
                "per_page": 1,
            })
            deal["coverage"][phase] = resp.json().get("total_results", 0)

        # Determine status
        if deal["coverage"]["completion_signals"] > 0:
            deal["status"] = "completed"
        elif deal["coverage"]["failure_signals"] > 0:
            deal["status"] = "failed"
        elif deal["coverage"]["regulatory_signals"] > 5:
            deal["status"] = "regulatory_review"
        elif deal["coverage"]["announcement_signals"] > 0:
            deal["status"] = "announced"
        elif deal["coverage"]["rumor_signals"] > 0:
            deal["status"] = "rumored"

        # Get timeline
        deal["timeline"] = self._build_deal_timeline(acquirer, target)

        # Sentiment analysis
        deal["sentiment_history"] = self._get_sentiment_history(acquirer, target, days=30)

        self.deals[deal_key] = deal
        return deal

    def _build_deal_timeline(self, acquirer, target, days=90):
        """Build timeline of deal events."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": f"{acquirer},{target}",
            "source.rank.opr.min": 0.7,
            "published_at.start": start,
            "language": "en",
            "sort.by": "published_at",
            "sort.order": "asc",
            "per_page": 50,
        })

        events = []
        for article in resp.json().get("results", []):
            title = article.get("title", "").lower()

            # Classify event type
            event_type = "coverage"
            for phase, keywords in MA_TAXONOMY.items():
                if any(kw.lower() in title for kw in keywords):
                    event_type = phase.replace("_signals", "")
                    break

            events.append({
                "date": article.get("published_at", "")[:10],
                "type": event_type,
                "title": article.get("title"),
                "source": article.get("source", {}).get("domain")
            })

        return events

    def _get_sentiment_history(self, acquirer, target, days=30):
        """Get sentiment trend over time."""
        history = []

        for i in range(days, 0, -7):  # Weekly data points
            start = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            end = (datetime.utcnow() - timedelta(days=max(0, i-7))).strftime("%Y-%m-%d")

            # Positive
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": f"{acquirer},{target}",
                "sentiment.overall.polarity": "positive",
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 1,
            })
            positive = resp.json().get("total_results", 0)

            # Negative
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": f"{acquirer},{target}",
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "published_at.end": end,
                "language": "en",
                "per_page": 1,
            })
            negative = resp.json().get("total_results", 0)

            total = positive + negative
            history.append({
                "week_start": start,
                "positive": positive,
                "negative": negative,
                "sentiment_ratio": positive / max(total, 1)
            })

        return history

    def analyze_regulatory_risk(self, acquirer, target):
        """Analyze regulatory/antitrust risk."""
        # Check for regulatory mentions
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": f"{acquirer},{target},{','.join(MA_TAXONOMY['regulatory_signals'])}",
            "language": "en",
            "per_page": 50,
        })
        articles = resp.json().get("results", [])

        risk_indicators = {
            "total_regulatory_mentions": len(articles),
            "ftc_mentions": 0,
            "doj_mentions": 0,
            "eu_mentions": 0,
            "antitrust_concerns": 0,
            "approval_mentions": 0
        }

        for article in articles:
            title = article.get("title", "").lower()

            if "ftc" in title:
                risk_indicators["ftc_mentions"] += 1
            if "doj" in title or "justice department" in title:
                risk_indicators["doj_mentions"] += 1
            if "eu" in title or "european" in title:
                risk_indicators["eu_mentions"] += 1
            if "antitrust" in title or "monopoly" in title or "competition" in title:
                risk_indicators["antitrust_concerns"] += 1
            if "approved" in title or "clears" in title:
                risk_indicators["approval_mentions"] += 1

        # Calculate risk score
        risk_score = (
            risk_indicators["antitrust_concerns"] * 10 +
            risk_indicators["ftc_mentions"] * 5 +
            risk_indicators["doj_mentions"] * 5 +
            risk_indicators["eu_mentions"] * 3 -
            risk_indicators["approval_mentions"] * 10
        )

        risk_indicators["risk_score"] = max(0, min(100, risk_score))
        risk_indicators["risk_level"] = (
            "high" if risk_indicators["risk_score"] >= 50 else
            "medium" if risk_indicators["risk_score"] >= 25 else
            "low"
        )

        return risk_indicators

    def calculate_deal_success_probability(self, deal_key):
        """Estimate deal success probability."""
        deal = self.deals.get(deal_key)
        if not deal:
            return None

        factors = {
            "sentiment_factor": 0,
            "regulatory_factor": 0,
            "momentum_factor": 0,
            "coverage_factor": 0
        }

        # Sentiment factor
        recent_sentiment = deal["sentiment_history"][-2:] if deal["sentiment_history"] else []
        if recent_sentiment:
            avg_ratio = sum(s["sentiment_ratio"] for s in recent_sentiment) / len(recent_sentiment)
            factors["sentiment_factor"] = avg_ratio * 100

        # Regulatory factor
        reg_risk = self.analyze_regulatory_risk(deal["acquirer"], deal["target"])
        factors["regulatory_factor"] = 100 - reg_risk["risk_score"]

        # Momentum factor (positive events)
        timeline = deal["timeline"]
        if timeline:
            positive_events = sum(1 for e in timeline if e["type"] in ["announcement", "regulatory"])
            negative_events = sum(1 for e in timeline if e["type"] in ["failure"])
            factors["momentum_factor"] = min(100, positive_events * 20 - negative_events * 30 + 50)

        # Coverage factor (more tier-1 coverage = more serious)
        total_coverage = sum(deal["coverage"].values())
        factors["coverage_factor"] = min(100, total_coverage * 2)

        # Weighted probability
        probability = (
            factors["sentiment_factor"] * 0.25 +
            factors["regulatory_factor"] * 0.35 +
            factors["momentum_factor"] * 0.25 +
            factors["coverage_factor"] * 0.15
        )

        return {
            "probability": probability,
            "factors": factors,
            "status": deal["status"]
        }

# Run M&A intelligence
print("M&A DEAL INTELLIGENCE")
print("=" * 70)

tracker = MADealTracker()

# Detect rumors
print("\nDETECTING M&A RUMORS...")
print("-" * 50)
rumors = tracker.detect_rumors(days=14, min_mentions=3)
print(f"Potential deals detected: {len(rumors)}")
for r in rumors[:5]:
    print(f"\n  {r['company']}")
    print(f"    Mentions: {r['mention_count']}")
    print(f"    Period: {r['first_mention']} to {r['latest_mention']}")
    print(f"    Sentiment: {'positive' if r['avg_sentiment'] > 0.1 else 'negative' if r['avg_sentiment'] < -0.1 else 'neutral'}")

# Track specific deal
print("\n" + "=" * 70)
print("TRACKING DEAL: Microsoft + Activision")
print("-" * 50)
deal = tracker.track_deal("Microsoft", "Activision")

print(f"Status: {deal['status'].upper()}")
print(f"\nCoverage by phase:")
for phase, count in deal['coverage'].items():
    print(f"  {phase}: {count}")

print(f"\nTimeline ({len(deal['timeline'])} events):")
for event in deal['timeline'][:5]:
    print(f"  {event['date']} [{event['type']}]: {event['title'][:50]}...")

# Regulatory risk
print("\n" + "=" * 70)
print("REGULATORY RISK ANALYSIS")
print("-" * 50)
risk = tracker.analyze_regulatory_risk("Microsoft", "Activision")
print(f"Risk Level: {risk['risk_level'].upper()}")
print(f"Risk Score: {risk['risk_score']}")
print(f"Antitrust mentions: {risk['antitrust_concerns']}")
print(f"FTC mentions: {risk['ftc_mentions']}")
print(f"Approval mentions: {risk['approval_mentions']}")

# Success probability
print("\n" + "=" * 70)
print("SUCCESS PROBABILITY")
print("-" * 50)
prob = tracker.calculate_deal_success_probability("Microsoft_Activision")
print(f"Estimated probability: {prob['probability']:.1f}%")
print(f"Factors:")
for factor, value in prob['factors'].items():
    print(f"  {factor}: {value:.1f}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const MA_KEYWORDS = {
  rumors: ["rumor", "reportedly", "considering", "exploring", "in talks", "potential"],
  announcements: ["announced", "confirms", "agrees to acquire", "deal signed"],
  regulatory: ["antitrust", "FTC", "DOJ", "regulatory approval", "merger review"],
  completion: ["completed", "closed", "finalized", "deal closes"],
  failure: ["terminated", "abandoned", "walks away", "blocks", "rejected"],
};

class MADealTracker {
  constructor() {
    this.deals = new Map();
    this.rumors = [];
  }

  async detectRumors(days = 14) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    const params = new URLSearchParams({
      api_key: API_KEY,
      title: MA_KEYWORDS.rumors.join(","),
      "source.rank.opr.min": "0.6",
      "published_at.start": start,
      language: "en",
      per_page: "50",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results || [];

    // Count mentions by company
    const mentions = new Map();
    for (const article of articles) {
      for (const entity of article.entities || []) {
        if (entity.type === "organization") {
          const current = mentions.get(entity.name) || { count: 0, articles: [] };
          current.count++;
          current.articles.push(article);
          mentions.set(entity.name, current);
        }
      }
    }

    // Filter and sort
    this.rumors = [...mentions.entries()]
      .filter(([, data]) => data.count >= 2)
      .map(([company, data]) => ({
        company,
        mentions: data.count,
        headlines: data.articles.slice(0, 3).map((a) => a.title),
      }))
      .sort((a, b) => b.mentions - a.mentions);

    return this.rumors;
  }

  async trackDeal(acquirer, target) {
    const deal = {
      acquirer,
      target,
      status: "tracking",
      coverage: {},
      timeline: [],
    };

    // Check each phase
    for (const [phase, keywords] of Object.entries(MA_KEYWORDS)) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        title: `${acquirer},${target},${keywords.slice(0, 3).join(",")}`,
        language: "en",
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      deal.coverage[phase] = data.total_results || 0;
    }

    // Determine status
    if (deal.coverage.completion > 0) deal.status = "completed";
    else if (deal.coverage.failure > 0) deal.status = "failed";
    else if (deal.coverage.regulatory > 3) deal.status = "regulatory_review";
    else if (deal.coverage.announcements > 0) deal.status = "announced";
    else if (deal.coverage.rumors > 0) deal.status = "rumored";

    // Get timeline
    const params = new URLSearchParams({
      api_key: API_KEY,
      title: `${acquirer},${target}`,
      "source.rank.opr.min": "0.7",
      language: "en",
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "20",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();

    deal.timeline = (data.results || []).map((a) => ({
      date: (a.published_at || "").slice(0, 10),
      title: a.title,
      source: a.source?.domain,
    }));

    this.deals.set(`${acquirer}_${target}`, deal);
    return deal;
  }

  async analyzeRegulatoryRisk(acquirer, target) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      title: `${acquirer},${target},${MA_KEYWORDS.regulatory.join(",")}`,
      language: "en",
      per_page: "30",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results || [];

    const risk = {
      totalMentions: articles.length,
      ftcMentions: 0,
      dojMentions: 0,
      antitrustConcerns: 0,
    };

    for (const article of articles) {
      const title = (article.title || "").toLowerCase();
      if (title.includes("ftc")) risk.ftcMentions++;
      if (title.includes("doj") || title.includes("justice department")) risk.dojMentions++;
      if (title.includes("antitrust") || title.includes("monopoly")) risk.antitrustConcerns++;
    }

    risk.riskScore = Math.min(
      100,
      risk.antitrustConcerns * 10 + risk.ftcMentions * 5 + risk.dojMentions * 5
    );
    risk.riskLevel =
      risk.riskScore >= 50 ? "high" : risk.riskScore >= 25 ? "medium" : "low";

    return risk;
  }
}

async function runTracker() {
  const tracker = new MADealTracker();

  console.log("M&A DEAL INTELLIGENCE");
  console.log("=".repeat(50));

  // Detect rumors
  console.log("\nDetecting rumors...");
  const rumors = await tracker.detectRumors(14);
  console.log(`Found ${rumors.length} potential deals`);
  rumors.slice(0, 5).forEach((r) => {
    console.log(`  ${r.company}: ${r.mentions} mentions`);
  });

  // Track specific deal
  console.log("\n" + "=".repeat(50));
  console.log("DEAL: Microsoft + Activision");
  const deal = await tracker.trackDeal("Microsoft", "Activision");
  console.log(`Status: ${deal.status}`);
  console.log("Coverage:");
  Object.entries(deal.coverage).forEach(([phase, count]) => {
    console.log(`  ${phase}: ${count}`);
  });

  // Regulatory risk
  console.log("\nRegulatory Risk:");
  const risk = await tracker.analyzeRegulatoryRisk("Microsoft", "Activision");
  console.log(`  Level: ${risk.riskLevel}`);
  console.log(`  Score: ${risk.riskScore}`);
  console.log(`  Antitrust concerns: ${risk.antitrustConcerns}`);
}

runTracker();
```

### PHP

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$maKeywords = [
    "rumors" => ["rumor", "reportedly", "considering", "in talks"],
    "announcements" => ["announced", "confirms", "agrees to acquire"],
    "regulatory" => ["antitrust", "FTC", "DOJ", "regulatory approval"],
    "completion" => ["completed", "closed", "finalized"],
    "failure" => ["terminated", "abandoned", "walks away", "blocks"],
];

class MADealTracker
{
    private string $apiKey;
    private string $baseUrl;
    private array $deals = [];

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function detectRumors(int $days = 14): array
    {
        global $maKeywords;

        $start = (new DateTime("-{$days} days"))->format("Y-m-d");

        $query = http_build_query([
            "api_key" => $this->apiKey,
            "title" => implode(",", $maKeywords["rumors"]),
            "source.rank.opr.min" => 0.6,
            "published_at.start" => $start,
            "language" => "en",
            "per_page" => 50,
        ]);

        $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
        $articles = $data["results"] ?? [];

        $mentions = [];
        foreach ($articles as $article) {
            foreach ($article["entities"] ?? [] as $entity) {
                if (($entity["type"] ?? "") === "organization") {
                    $name = $entity["name"];
                    if (!isset($mentions[$name])) {
                        $mentions[$name] = ["count" => 0, "articles" => []];
                    }
                    $mentions[$name]["count"]++;
                    $mentions[$name]["articles"][] = $article;
                }
            }
        }

        $rumors = [];
        foreach ($mentions as $company => $data) {
            if ($data["count"] >= 2) {
                $rumors[] = [
                    "company" => $company,
                    "mentions" => $data["count"],
                    "headlines" => array_slice(array_column($data["articles"], "title"), 0, 3),
                ];
            }
        }

        usort($rumors, fn($a, $b) => $b["mentions"] <=> $a["mentions"]);
        return $rumors;
    }

    public function trackDeal(string $acquirer, string $target): array
    {
        global $maKeywords;

        $deal = [
            "acquirer" => $acquirer,
            "target" => $target,
            "status" => "tracking",
            "coverage" => [],
        ];

        foreach ($maKeywords as $phase => $keywords) {
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "title" => "{$acquirer},{$target}," . implode(",", array_slice($keywords, 0, 3)),
                "language" => "en",
                "per_page" => 1,
            ]);

            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $deal["coverage"][$phase] = $data["total_results"] ?? 0;
        }

        // Determine status
        if ($deal["coverage"]["completion"] > 0) $deal["status"] = "completed";
        elseif ($deal["coverage"]["failure"] > 0) $deal["status"] = "failed";
        elseif ($deal["coverage"]["regulatory"] > 3) $deal["status"] = "regulatory_review";
        elseif ($deal["coverage"]["announcements"] > 0) $deal["status"] = "announced";
        elseif ($deal["coverage"]["rumors"] > 0) $deal["status"] = "rumored";

        $this->deals["{$acquirer}_{$target}"] = $deal;
        return $deal;
    }

    public function analyzeRegulatoryRisk(string $acquirer, string $target): array
    {
        global $maKeywords;

        $query = http_build_query([
            "api_key" => $this->apiKey,
            "title" => "{$acquirer},{$target}," . implode(",", $maKeywords["regulatory"]),
            "language" => "en",
            "per_page" => 30,
        ]);

        $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
        $articles = $data["results"] ?? [];

        $risk = [
            "total_mentions" => count($articles),
            "ftc_mentions" => 0,
            "doj_mentions" => 0,
            "antitrust_concerns" => 0,
        ];

        foreach ($articles as $article) {
            $title = strtolower($article["title"] ?? "");
            if (strpos($title, "ftc") !== false) $risk["ftc_mentions"]++;
            if (strpos($title, "doj") !== false) $risk["doj_mentions"]++;
            if (strpos($title, "antitrust") !== false) $risk["antitrust_concerns"]++;
        }

        $risk["risk_score"] = min(100,
            $risk["antitrust_concerns"] * 10 +
            $risk["ftc_mentions"] * 5 +
            $risk["doj_mentions"] * 5
        );

        $risk["risk_level"] = $risk["risk_score"] >= 50 ? "high" :
            ($risk["risk_score"] >= 25 ? "medium" : "low");

        return $risk;
    }
}

$tracker = new MADealTracker();

echo "M&A DEAL INTELLIGENCE\n";
echo str_repeat("=", 50) . "\n";

// Rumors
$rumors = $tracker->detectRumors(14);
echo "\nRumors detected: " . count($rumors) . "\n";
foreach (array_slice($rumors, 0, 5) as $r) {
    echo "  {$r['company']}: {$r['mentions']} mentions\n";
}

// Track deal
echo "\n" . str_repeat("=", 50) . "\n";
echo "DEAL: Microsoft + Activision\n";
$deal = $tracker->trackDeal("Microsoft", "Activision");
echo "Status: {$deal['status']}\n";
echo "Coverage:\n";
foreach ($deal["coverage"] as $phase => $count) {
    echo "  {$phase}: {$count}\n";
}

// Regulatory
$risk = $tracker->analyzeRegulatoryRisk("Microsoft", "Activision");
echo "\nRegulatory Risk: {$risk['risk_level']}\n";
echo "Score: {$risk['risk_score']}\n";
```

## Common Use Cases

- **Deal sourcing** — detect M&A rumors before official announcements.
- **Transaction tracking** — monitor deals from announcement to completion.
- **Regulatory monitoring** — track antitrust and approval developments.
- **Due diligence support** — gather media intelligence on targets.
- **Competitive intelligence** — monitor competitor M&A activity.
- **Market mapping** — identify consolidation trends in sectors.
- **Deal sentiment analysis** — assess market reaction to transactions.
- **Success prediction** — estimate deal completion probability.

## See Also

- [examples.md](./examples.md) — detailed code examples for M&A deal intelligence.
