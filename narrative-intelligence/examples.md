# Narrative Intelligence — Examples

Advanced code examples for narrative tracking, coordinated messaging detection, frame analysis, and counter-narrative dynamics.

---

## Python — Comprehensive Narrative Analysis System

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import re
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"


class AdvancedNarrativeAnalyzer:
    """
    Comprehensive narrative analysis with frame detection,
    coordination analysis, and narrative lifecycle tracking.
    """

    def __init__(self):
        self.narratives = {}
        self.article_cache = []

    def fetch_narrative_articles(self, keywords, days=30, max_articles=500):
        """Fetch articles for narrative analysis."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        articles = []
        page = 1

        while len(articles) < max_articles:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(keywords),
                "published_at.start": start,
                "language.code": "en",
                "sort.by": "published_at",
                "sort.order": "asc",
                "per_page": 50,
                "page": page,
            })
            batch = resp.json().get("results", [])

            if not batch:
                break

            articles.extend(batch)
            page += 1

            if len(batch) < 50:
                break

        self.article_cache = articles[:max_articles]
        return self.article_cache

    def build_narrative_timeline(self, keywords, name="narrative"):
        """Build detailed timeline of narrative evolution."""
        articles = self.fetch_narrative_articles(keywords, days=60)

        # Group by day
        daily_data = defaultdict(lambda: {
            "count": 0,
            "sources": set(),
            "sentiment_sum": 0,
            "tier1_count": 0,
            "titles": []
        })

        for article in articles:
            date = article.get("published_at", "")[:10]
            daily_data[date]["count"] += 1
            daily_data[date]["sources"].add(article.get("source", {}).get("domain", ""))
            daily_data[date]["sentiment_sum"] += article.get("sentiment", {}).get("overall", {}).get("score", 0)
            daily_data[date]["titles"].append(article.get("title", ""))

            opr = article.get("source", {}).get("rank", {}).get("opr", 0)
            if opr >= 6:
                daily_data[date]["tier1_count"] += 1

        # Convert to timeline
        timeline = []
        for date in sorted(daily_data.keys()):
            data = daily_data[date]
            timeline.append({
                "date": date,
                "article_count": data["count"],
                "unique_sources": len(data["sources"]),
                "avg_sentiment": data["sentiment_sum"] / max(data["count"], 1),
                "tier1_share": data["tier1_count"] / max(data["count"], 1),
                "sample_titles": data["titles"][:3]
            })

        self.narratives[name] = {
            "keywords": keywords,
            "timeline": timeline,
            "total_articles": len(articles),
            "total_sources": len(set(a.get("source", {}).get("domain") for a in articles))
        }

        return self.narratives[name]

    def detect_narrative_phases(self, narrative_name):
        """Detect lifecycle phases: emergence, growth, peak, decline."""
        narrative = self.narratives.get(narrative_name)
        if not narrative:
            return None

        timeline = narrative["timeline"]
        if len(timeline) < 14:
            return {"phases": [], "current_phase": "unknown"}

        # Calculate moving averages
        window = 7
        counts = [t["article_count"] for t in timeline]
        ma = []
        for i in range(len(counts)):
            if i >= window - 1:
                ma.append(sum(counts[i-window+1:i+1]) / window)
            else:
                ma.append(counts[i])

        # Detect phases
        phases = []
        phase_start = 0
        current_phase = "emergence"

        for i in range(1, len(ma)):
            growth_rate = (ma[i] - ma[i-1]) / max(ma[i-1], 1)

            if current_phase == "emergence" and growth_rate > 0.2:
                if i > phase_start:
                    phases.append({
                        "phase": "emergence",
                        "start": timeline[phase_start]["date"],
                        "end": timeline[i-1]["date"],
                        "avg_volume": sum(counts[phase_start:i]) / (i - phase_start)
                    })
                current_phase = "growth"
                phase_start = i

            elif current_phase == "growth" and growth_rate < 0.05:
                if i > phase_start:
                    phases.append({
                        "phase": "growth",
                        "start": timeline[phase_start]["date"],
                        "end": timeline[i-1]["date"],
                        "avg_volume": sum(counts[phase_start:i]) / (i - phase_start)
                    })
                current_phase = "peak"
                phase_start = i

            elif current_phase == "peak" and growth_rate < -0.1:
                if i > phase_start:
                    phases.append({
                        "phase": "peak",
                        "start": timeline[phase_start]["date"],
                        "end": timeline[i-1]["date"],
                        "avg_volume": sum(counts[phase_start:i]) / (i - phase_start)
                    })
                current_phase = "decline"
                phase_start = i

        # Add final phase
        if phase_start < len(timeline):
            phases.append({
                "phase": current_phase,
                "start": timeline[phase_start]["date"],
                "end": timeline[-1]["date"],
                "avg_volume": sum(counts[phase_start:]) / (len(counts) - phase_start)
            })

        return {
            "phases": phases,
            "current_phase": current_phase
        }

    def analyze_frame_distribution(self, narrative_name, frames):
        """Analyze how different frames are used in the narrative."""
        if not self.article_cache:
            return {}

        frame_counts = {frame: 0 for frame in frames}
        frame_sentiment = {frame: [] for frame in frames}

        for article in self.article_cache:
            title = article.get("title", "").lower()
            sentiment = article.get("sentiment", {}).get("overall", {}).get("score", 0)

            for frame, keywords in frames.items():
                if any(kw.lower() in title for kw in keywords):
                    frame_counts[frame] += 1
                    frame_sentiment[frame].append(sentiment)

        total = sum(frame_counts.values()) or 1

        return {
            frame: {
                "count": count,
                "share": count / total,
                "avg_sentiment": sum(frame_sentiment[frame]) / max(len(frame_sentiment[frame]), 1)
            }
            for frame, count in frame_counts.items()
        }

    def detect_coordination_signals(self, narrative_name, time_window_hours=6):
        """Detect potential coordinated messaging."""
        if not self.article_cache:
            return {"coordination_score": 0, "clusters": []}

        # Group by time windows
        time_clusters = defaultdict(list)

        for article in self.article_cache:
            pub_time = article.get("published_at", "")
            if pub_time:
                # Truncate to time window
                dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                window_key = dt.strftime("%Y-%m-%d") + f"_{dt.hour // time_window_hours}"
                time_clusters[window_key].append(article)

        # Analyze clusters for coordination signals
        suspicious_clusters = []

        for window, articles in time_clusters.items():
            if len(articles) < 5:
                continue

            # Check for unusual similarity
            titles = [a.get("title", "").lower() for a in articles]
            sources = [a.get("source", {}).get("domain", "") for a in articles]

            # Unique sources ratio (low = potentially coordinated)
            source_diversity = len(set(sources)) / len(sources)

            # Title similarity (simple word overlap)
            common_words = set()
            for title in titles:
                words = set(title.split())
                if not common_words:
                    common_words = words
                else:
                    common_words &= words

            title_similarity = len(common_words) / 10  # Normalized

            coordination_score = (1 - source_diversity) * 0.5 + title_similarity * 0.5

            if coordination_score > 0.3:
                suspicious_clusters.append({
                    "window": window,
                    "article_count": len(articles),
                    "unique_sources": len(set(sources)),
                    "coordination_score": coordination_score,
                    "common_words": list(common_words)[:10]
                })

        overall_score = sum(c["coordination_score"] for c in suspicious_clusters) / max(len(suspicious_clusters), 1)

        return {
            "coordination_score": overall_score,
            "suspicious_clusters": sorted(suspicious_clusters, key=lambda x: x["coordination_score"], reverse=True)[:5]
        }

    def analyze_counter_narrative(self, main_keywords, counter_keywords, days=30):
        """Analyze dynamics between main narrative and counter-narrative."""
        # Build both timelines
        main_start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        main_daily = defaultdict(int)
        counter_daily = defaultdict(int)

        # Main narrative by day
        for i in range(days, 0, -1):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            next_date = (datetime.utcnow() - timedelta(days=i-1)).strftime("%Y-%m-%d")

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(main_keywords),
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 1,
            })
            main_daily[date] = len(resp.json().get("results", []))

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": ",".join(counter_keywords),
                "published_at.start": date,
                "published_at.end": next_date,
                "language.code": "en",
                "per_page": 1,
            })
            counter_daily[date] = len(resp.json().get("results", []))

        # Calculate dynamics
        dates = sorted(main_daily.keys())
        dynamics = []

        for date in dates:
            main = main_daily[date]
            counter = counter_daily[date]
            total = main + counter

            dynamics.append({
                "date": date,
                "main": main,
                "counter": counter,
                "main_share": main / max(total, 1),
                "dominance": "main" if main > counter else "counter" if counter > main else "balanced"
            })

        # Identify momentum shifts
        shifts = []
        for i in range(7, len(dynamics)):
            recent_main_share = sum(d["main_share"] for d in dynamics[i-7:i]) / 7
            prior_main_share = sum(d["main_share"] for d in dynamics[max(0, i-14):i-7]) / 7

            if abs(recent_main_share - prior_main_share) > 0.1:
                shifts.append({
                    "date": dynamics[i]["date"],
                    "shift_to": "main" if recent_main_share > prior_main_share else "counter",
                    "magnitude": abs(recent_main_share - prior_main_share)
                })

        return {
            "dynamics": dynamics,
            "total_main": sum(main_daily.values()),
            "total_counter": sum(counter_daily.values()),
            "momentum_shifts": shifts,
            "current_dominant": dynamics[-1]["dominance"] if dynamics else "unknown"
        }


class NarrativeReportGenerator:
    """Generate comprehensive narrative reports."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def generate_report(self, narrative_name, frames=None):
        """Generate full narrative analysis report."""
        narrative = self.analyzer.narratives.get(narrative_name)
        if not narrative:
            return None

        report = {
            "name": narrative_name,
            "generated_at": datetime.utcnow().isoformat(),
            "overview": {
                "total_articles": narrative["total_articles"],
                "total_sources": narrative["total_sources"],
                "keywords": narrative["keywords"],
                "date_range": f"{narrative['timeline'][0]['date']} to {narrative['timeline'][-1]['date']}"
            }
        }

        # Phases
        phases = self.analyzer.detect_narrative_phases(narrative_name)
        report["lifecycle"] = phases

        # Frames
        if frames:
            report["frames"] = self.analyzer.analyze_frame_distribution(narrative_name, frames)

        # Coordination
        coordination = self.analyzer.detect_coordination_signals(narrative_name)
        report["coordination_analysis"] = coordination

        # Metrics
        timeline = narrative["timeline"]
        report["metrics"] = {
            "peak_day": max(timeline, key=lambda x: x["article_count"]),
            "avg_daily_volume": sum(t["article_count"] for t in timeline) / len(timeline),
            "avg_sources_per_day": sum(t["unique_sources"] for t in timeline) / len(timeline),
            "overall_sentiment": sum(t["avg_sentiment"] for t in timeline) / len(timeline),
            "elite_media_penetration": sum(t["tier1_share"] for t in timeline) / len(timeline)
        }

        return report


# Run comprehensive analysis
print("COMPREHENSIVE NARRATIVE ANALYSIS")
print("=" * 70)

analyzer = AdvancedNarrativeAnalyzer()

# Build narrative
print("\nBuilding narrative timeline...")
narrative = analyzer.build_narrative_timeline(
    keywords=["AI", "regulation", "safety", "risk"],
    name="AI_Regulation"
)

print(f"Total articles: {narrative['total_articles']}")
print(f"Total sources: {narrative['total_sources']}")

# Phases
print("\n" + "=" * 50)
print("NARRATIVE LIFECYCLE PHASES")
print("-" * 40)
phases = analyzer.detect_narrative_phases("AI_Regulation")
print(f"Current phase: {phases['current_phase']}")
for phase in phases["phases"]:
    print(f"  {phase['phase'].upper()}: {phase['start']} to {phase['end']} (avg: {phase['avg_volume']:.1f}/day)")

# Frame analysis
print("\n" + "=" * 50)
print("FRAME ANALYSIS")
print("-" * 40)
frames = {
    "risk_frame": ["danger", "threat", "risk", "concern", "warning"],
    "progress_frame": ["innovation", "breakthrough", "advance", "opportunity"],
    "regulation_frame": ["regulate", "law", "policy", "government", "control"],
    "ethics_frame": ["ethics", "moral", "responsible", "fair", "bias"]
}
frame_analysis = analyzer.analyze_frame_distribution("AI_Regulation", frames)
for frame, data in sorted(frame_analysis.items(), key=lambda x: x[1]["count"], reverse=True):
    sentiment_label = "+" if data["avg_sentiment"] > 0.1 else "-" if data["avg_sentiment"] < -0.1 else "○"
    print(f"  {frame}: {data['count']} ({data['share']:.1%}) [{sentiment_label}]")

# Coordination detection
print("\n" + "=" * 50)
print("COORDINATION ANALYSIS")
print("-" * 40)
coordination = analyzer.detect_coordination_signals("AI_Regulation")
print(f"Overall coordination score: {coordination['coordination_score']:.2f}")
for cluster in coordination["suspicious_clusters"][:3]:
    print(f"  {cluster['window']}: {cluster['article_count']} articles, "
          f"score={cluster['coordination_score']:.2f}")

# Counter-narrative analysis
print("\n" + "=" * 50)
print("COUNTER-NARRATIVE DYNAMICS")
print("-" * 40)
counter = analyzer.analyze_counter_narrative(
    main_keywords=["AI", "danger", "risk", "threat"],
    counter_keywords=["AI", "opportunity", "benefit", "progress"],
    days=21
)
print(f"Main narrative: {counter['total_main']:,} articles")
print(f"Counter narrative: {counter['total_counter']:,} articles")
print(f"Current dominant: {counter['current_dominant']}")
print(f"Momentum shifts: {len(counter['momentum_shifts'])}")

# Generate report
print("\n" + "=" * 50)
print("GENERATING REPORT")
print("-" * 40)
reporter = NarrativeReportGenerator(analyzer)
report = reporter.generate_report("AI_Regulation", frames)
print(f"Report generated: {report['overview']['date_range']}")
print(f"Peak day: {report['metrics']['peak_day']['date']} ({report['metrics']['peak_day']['article_count']} articles)")
print(f"Elite penetration: {report['metrics']['elite_media_penetration']:.1%}")
```

---

## JavaScript — Real-Time Narrative Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

class NarrativeMonitor {
  constructor() {
    this.narratives = new Map();
    this.alerts = [];
  }

  async trackNarrative(config) {
    const { name, keywords, counterKeywords = [], frames = {} } = config;

    const narrative = {
      name,
      keywords,
      counterKeywords,
      frames,
      timeline: [],
      alerts: [],
      lastUpdated: null,
    };

    this.narratives.set(name, narrative);
    return narrative;
  }

  async updateNarrative(name, days = 30) {
    const narrative = this.narratives.get(name);
    if (!narrative) return null;

    const timeline = [];

    for (let i = days; i > 0; i--) {
      const date = new Date(Date.now() - i * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];
      const nextDate = new Date(Date.now() - (i - 1) * 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];

      // Main count
      let params = new URLSearchParams({
        api_key: API_KEY,
        title: narrative.keywords.join(","),
        "published_at.start": date,
        "published_at.end": nextDate,
        "language.code": "en",
        per_page: "1",
      });

      let response = await fetch(`${BASE_URL}?${params}`);
      let data = await response.json();
      const mainCount = data.results?.length || 0;

      // Sentiment
      params = new URLSearchParams({
        api_key: API_KEY,
        title: narrative.keywords.join(","),
        "sentiment.overall.polarity": "positive",
        "published_at.start": date,
        "published_at.end": nextDate,
        "language.code": "en",
        per_page: "1",
      });

      response = await fetch(`${BASE_URL}?${params}`);
      data = await response.json();
      const positiveCount = data.results?.length || 0;

      // Counter narrative
      let counterCount = 0;
      if (narrative.counterKeywords.length > 0) {
        params = new URLSearchParams({
          api_key: API_KEY,
          title: narrative.counterKeywords.join(","),
          "published_at.start": date,
          "published_at.end": nextDate,
          "language.code": "en",
          per_page: "1",
        });

        response = await fetch(`${BASE_URL}?${params}`);
        data = await response.json();
        counterCount = data.results?.length || 0;
      }

      timeline.push({
        date,
        mainCount,
        counterCount,
        positiveRatio: positiveCount / Math.max(mainCount, 1),
        dominance: mainCount > counterCount ? "main" : counterCount > mainCount ? "counter" : "balanced",
      });
    }

    narrative.timeline = timeline;
    narrative.lastUpdated = new Date().toISOString();

    // Check for alerts
    this.checkAlerts(narrative);

    return narrative;
  }

  checkAlerts(narrative) {
    const timeline = narrative.timeline;
    if (timeline.length < 7) return;

    const recent = timeline.slice(-3);
    const prior = timeline.slice(-10, -3);

    // Volume spike alert
    const recentAvg = recent.reduce((s, t) => s + t.mainCount, 0) / recent.length;
    const priorAvg = prior.reduce((s, t) => s + t.mainCount, 0) / prior.length;

    if (recentAvg > priorAvg * 2) {
      this.alerts.push({
        narrative: narrative.name,
        type: "VOLUME_SPIKE",
        timestamp: new Date().toISOString(),
        message: `Volume surged ${((recentAvg / priorAvg - 1) * 100).toFixed(0)}%`,
        severity: recentAvg > priorAvg * 3 ? "HIGH" : "MEDIUM",
      });
    }

    // Sentiment shift alert
    const recentSentiment = recent.reduce((s, t) => s + t.positiveRatio, 0) / recent.length;
    const priorSentiment = prior.reduce((s, t) => s + t.positiveRatio, 0) / prior.length;

    if (Math.abs(recentSentiment - priorSentiment) > 0.2) {
      this.alerts.push({
        narrative: narrative.name,
        type: "SENTIMENT_SHIFT",
        timestamp: new Date().toISOString(),
        message: `Sentiment shifted ${recentSentiment > priorSentiment ? "positive" : "negative"}`,
        severity: "MEDIUM",
      });
    }

    // Counter-narrative surge
    const recentCounter = recent.reduce((s, t) => s + t.counterCount, 0);
    const priorCounter = prior.reduce((s, t) => s + t.counterCount, 0);

    if (priorCounter > 0 && recentCounter > priorCounter * 1.5) {
      this.alerts.push({
        narrative: narrative.name,
        type: "COUNTER_NARRATIVE_SURGE",
        timestamp: new Date().toISOString(),
        message: `Counter-narrative volume increased ${((recentCounter / priorCounter - 1) * 100).toFixed(0)}%`,
        severity: "HIGH",
      });
    }
  }

  calculateMetrics(name) {
    const narrative = this.narratives.get(name);
    if (!narrative || narrative.timeline.length === 0) return null;

    const timeline = narrative.timeline;
    const mainTotal = timeline.reduce((s, t) => s + t.mainCount, 0);
    const counterTotal = timeline.reduce((s, t) => s + t.counterCount, 0);

    // Velocity
    const recent7 = timeline.slice(-7).reduce((s, t) => s + t.mainCount, 0);
    const prior7 = timeline.slice(-14, -7).reduce((s, t) => s + t.mainCount, 0);
    const velocity = prior7 > 0 ? ((recent7 - prior7) / prior7) * 100 : 0;

    // Dominant frame
    let mainDays = 0;
    let counterDays = 0;
    timeline.forEach((t) => {
      if (t.dominance === "main") mainDays++;
      else if (t.dominance === "counter") counterDays++;
    });

    return {
      totalMain: mainTotal,
      totalCounter: counterTotal,
      dominanceRatio: mainTotal / Math.max(counterTotal, 1),
      velocity,
      velocityStatus: velocity > 20 ? "surging" : velocity < -20 ? "declining" : "stable",
      mainDominantDays: mainDays,
      counterDominantDays: counterDays,
      avgSentiment: timeline.reduce((s, t) => s + t.positiveRatio, 0) / timeline.length,
    };
  }

  getRecentAlerts(count = 10) {
    return this.alerts.slice(-count).reverse();
  }
}

async function runMonitor() {
  const monitor = new NarrativeMonitor();

  console.log("NARRATIVE MONITOR");
  console.log("=".repeat(50));

  // Set up narratives
  await monitor.trackNarrative({
    name: "AI_Safety",
    keywords: ["AI", "safety", "risk", "regulation", "control"],
    counterKeywords: ["AI", "innovation", "progress", "opportunity", "growth"],
  });

  await monitor.trackNarrative({
    name: "Climate_Action",
    keywords: ["climate", "crisis", "action", "emergency", "carbon"],
    counterKeywords: ["climate", "skeptic", "natural", "cycle", "exaggeration"],
  });

  // Update narratives
  console.log("\nUpdating narratives...");

  for (const [name] of monitor.narratives) {
    console.log(`\nProcessing ${name}...`);
    await monitor.updateNarrative(name, 21);

    const metrics = monitor.calculateMetrics(name);
    if (metrics) {
      console.log(`\n${name.toUpperCase()}`);
      console.log("-".repeat(40));
      console.log(`Total main: ${metrics.totalMain.toLocaleString()}`);
      console.log(`Total counter: ${metrics.totalCounter.toLocaleString()}`);
      console.log(`Dominance ratio: ${metrics.dominanceRatio.toFixed(2)}x`);
      console.log(`Velocity: ${metrics.velocity.toFixed(1)}% (${metrics.velocityStatus})`);
      console.log(`Main dominant: ${metrics.mainDominantDays} days`);
      console.log(`Avg sentiment: ${(metrics.avgSentiment * 100).toFixed(1)}% positive`);
    }
  }

  // Show alerts
  console.log("\n" + "=".repeat(50));
  console.log("ALERTS");
  console.log("-".repeat(40));

  const alerts = monitor.getRecentAlerts();
  if (alerts.length === 0) {
    console.log("No alerts triggered");
  } else {
    alerts.forEach((alert) => {
      console.log(`[${alert.severity}] ${alert.narrative}: ${alert.type}`);
      console.log(`  ${alert.message}`);
    });
  }
}

runMonitor();
```

---

## PHP — Narrative Analysis Service

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

class NarrativeService
{
    private string $apiKey;
    private string $baseUrl;
    private array $narratives = [];
    private array $articleCache = [];

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function fetchArticles(array $keywords, int $days = 30, int $max = 200): array
    {
        $start = (new DateTime("-{$days} days"))->format("Y-m-d");
        $articles = [];
        $page = 1;

        while (count($articles) < $max) {
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "title" => implode(",", $keywords),
                "published_at.start" => $start,
                "language.code" => "en",
                "sort.by" => "published_at",
                "sort.order" => "asc",
                "per_page" => 50,
                "page" => $page,
            ]);

            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $batch = $data["results"] ?? [];

            if (empty($batch)) break;
            $articles = array_merge($articles, $batch);
            $page++;

            if (count($batch) < 50) break;
        }

        $this->articleCache = array_slice($articles, 0, $max);
        return $this->articleCache;
    }

    public function buildTimeline(string $name, array $keywords): array
    {
        $articles = $this->fetchArticles($keywords, 30);

        // Group by day
        $daily = [];
        foreach ($articles as $article) {
            $date = substr($article["published_at"] ?? "", 0, 10);
            if (!isset($daily[$date])) {
                $daily[$date] = ["count" => 0, "sources" => [], "sentiment_sum" => 0];
            }
            $daily[$date]["count"]++;
            $daily[$date]["sources"][] = $article["source"]["domain"] ?? "";
            $daily[$date]["sentiment_sum"] += $article["sentiment"]["overall"]["score"] ?? 0;
        }

        $timeline = [];
        ksort($daily);
        foreach ($daily as $date => $data) {
            $timeline[] = [
                "date" => $date,
                "count" => $data["count"],
                "unique_sources" => count(array_unique($data["sources"])),
                "avg_sentiment" => $data["count"] > 0 ? $data["sentiment_sum"] / $data["count"] : 0,
            ];
        }

        $this->narratives[$name] = [
            "keywords" => $keywords,
            "timeline" => $timeline,
            "total" => count($articles),
        ];

        return $this->narratives[$name];
    }

    public function detectPhases(string $name): array
    {
        $narrative = $this->narratives[$name] ?? null;
        if (!$narrative) return [];

        $timeline = $narrative["timeline"];
        if (count($timeline) < 14) return ["current_phase" => "unknown", "phases" => []];

        $counts = array_column($timeline, "count");
        $phases = [];
        $currentPhase = "emergence";
        $phaseStart = 0;

        for ($i = 7; $i < count($counts); $i++) {
            $recentAvg = array_sum(array_slice($counts, $i - 7, 7)) / 7;
            $priorAvg = array_sum(array_slice($counts, max(0, $i - 14), 7)) / 7;

            $growthRate = $priorAvg > 0 ? ($recentAvg - $priorAvg) / $priorAvg : 0;

            if ($currentPhase === "emergence" && $growthRate > 0.3) {
                $phases[] = ["phase" => "emergence", "start" => $timeline[$phaseStart]["date"], "end" => $timeline[$i - 1]["date"]];
                $currentPhase = "growth";
                $phaseStart = $i;
            } elseif ($currentPhase === "growth" && $growthRate < 0.05) {
                $phases[] = ["phase" => "growth", "start" => $timeline[$phaseStart]["date"], "end" => $timeline[$i - 1]["date"]];
                $currentPhase = "peak";
                $phaseStart = $i;
            } elseif ($currentPhase === "peak" && $growthRate < -0.15) {
                $phases[] = ["phase" => "peak", "start" => $timeline[$phaseStart]["date"], "end" => $timeline[$i - 1]["date"]];
                $currentPhase = "decline";
                $phaseStart = $i;
            }
        }

        $phases[] = ["phase" => $currentPhase, "start" => $timeline[$phaseStart]["date"], "end" => $timeline[count($timeline) - 1]["date"]];

        return ["current_phase" => $currentPhase, "phases" => $phases];
    }

    public function analyzeFrames(string $name, array $frames): array
    {
        if (empty($this->articleCache)) return [];

        $frameCounts = array_fill_keys(array_keys($frames), 0);

        foreach ($this->articleCache as $article) {
            $title = strtolower($article["title"] ?? "");

            foreach ($frames as $frame => $keywords) {
                foreach ($keywords as $keyword) {
                    if (strpos($title, strtolower($keyword)) !== false) {
                        $frameCounts[$frame]++;
                        break;
                    }
                }
            }
        }

        $total = array_sum($frameCounts) ?: 1;
        $result = [];
        foreach ($frameCounts as $frame => $count) {
            $result[$frame] = ["count" => $count, "share" => $count / $total];
        }

        return $result;
    }

    public function calculateVelocity(string $name): array
    {
        $narrative = $this->narratives[$name] ?? null;
        if (!$narrative || count($narrative["timeline"]) < 14) {
            return ["velocity" => 0, "status" => "unknown"];
        }

        $timeline = $narrative["timeline"];
        $recent = array_sum(array_column(array_slice($timeline, -7), "count"));
        $prior = array_sum(array_column(array_slice($timeline, -14, 7), "count"));

        $velocity = $prior > 0 ? (($recent - $prior) / $prior) * 100 : 0;

        return [
            "velocity" => $velocity,
            "status" => $velocity > 20 ? "surging" : ($velocity < -20 ? "declining" : "stable"),
        ];
    }

    public function generateReport(string $name, array $frames = []): array
    {
        $narrative = $this->narratives[$name] ?? null;
        if (!$narrative) return [];

        $phases = $this->detectPhases($name);
        $velocity = $this->calculateVelocity($name);
        $frameAnalysis = !empty($frames) ? $this->analyzeFrames($name, $frames) : [];

        $timeline = $narrative["timeline"];
        $peakDay = array_reduce($timeline, function ($max, $day) {
            return $day["count"] > ($max["count"] ?? 0) ? $day : $max;
        }, ["count" => 0]);

        return [
            "name" => $name,
            "generated_at" => (new DateTime())->format("c"),
            "total_articles" => $narrative["total"],
            "date_range" => $timeline[0]["date"] . " to " . $timeline[count($timeline) - 1]["date"],
            "current_phase" => $phases["current_phase"],
            "velocity" => $velocity,
            "peak_day" => $peakDay,
            "frames" => $frameAnalysis,
            "avg_daily_volume" => $narrative["total"] / count($timeline),
        ];
    }
}

// Run analysis
$service = new NarrativeService();

echo "NARRATIVE ANALYSIS SERVICE\n";
echo str_repeat("=", 60) . "\n";

// Build narrative
$narrative = $service->buildTimeline(
    "AI_Regulation",
    ["AI", "regulation", "safety", "control", "policy"]
);

echo "\nNARRATIVE: AI_Regulation\n";
echo "Total articles: {$narrative['total']}\n";

// Phases
$phases = $service->detectPhases("AI_Regulation");
echo "\nCurrent phase: {$phases['current_phase']}\n";
echo "Phases:\n";
foreach ($phases["phases"] as $phase) {
    echo "  {$phase['phase']}: {$phase['start']} to {$phase['end']}\n";
}

// Velocity
$velocity = $service->calculateVelocity("AI_Regulation");
printf("\nVelocity: %.1f%% (%s)\n", $velocity["velocity"], $velocity["status"]);

// Frames
$frames = [
    "risk" => ["danger", "threat", "risk", "concern"],
    "innovation" => ["innovation", "progress", "advance", "breakthrough"],
    "policy" => ["regulation", "law", "policy", "government"],
];

$frameAnalysis = $service->analyzeFrames("AI_Regulation", $frames);
echo "\nFrame Analysis:\n";
foreach ($frameAnalysis as $frame => $data) {
    printf("  %s: %d (%.1f%%)\n", $frame, $data["count"], $data["share"] * 100);
}

// Full report
echo "\n" . str_repeat("=", 60) . "\n";
echo "FULL REPORT\n";
$report = $service->generateReport("AI_Regulation", $frames);
printf("Date range: %s\n", $report["date_range"]);
printf("Peak day: %s (%d articles)\n", $report["peak_day"]["date"], $report["peak_day"]["count"]);
printf("Avg daily volume: %.1f\n", $report["avg_daily_volume"]);
```

---

## See Also

- [README.md](./README.md) — Narrative Intelligence workflow overview and quick start.
