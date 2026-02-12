# Industry Disruption Radar — Code Examples

Advanced examples for building disruption detection and innovation tracking systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Comprehensive Disruption Intelligence System

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

class DisruptionRadar:
    """Comprehensive industry disruption detection system."""

    DISRUPTION_THEMES = {
        "Generative AI": {
            "keywords": ["ChatGPT", "generative AI", "LLM", "GPT-4", "Claude", "Midjourney", "DALL-E"],
            "incumbents": ["Google", "Microsoft", "Adobe", "Salesforce"],
            "disruptors": ["OpenAI", "Anthropic", "Stability AI", "Midjourney"],
        },
        "Electric Vehicles": {
            "keywords": ["EV", "electric vehicle", "Tesla", "battery", "charging"],
            "incumbents": ["Ford", "GM", "Toyota", "Volkswagen"],
            "disruptors": ["Tesla", "Rivian", "Lucid", "BYD", "NIO"],
        },
        "Fintech Banking": {
            "keywords": ["neobank", "digital bank", "fintech", "mobile banking"],
            "incumbents": ["JPMorgan", "Bank of America", "Wells Fargo", "Citi"],
            "disruptors": ["Chime", "Revolut", "Nubank", "SoFi", "Stripe"],
        },
        "Space Technology": {
            "keywords": ["SpaceX", "rocket", "satellite", "space tourism", "launch"],
            "incumbents": ["Boeing", "Lockheed Martin", "Northrop Grumman"],
            "disruptors": ["SpaceX", "Blue Origin", "Rocket Lab", "Planet Labs"],
        },
        "Digital Health": {
            "keywords": ["telehealth", "digital health", "healthtech", "remote monitoring"],
            "incumbents": ["UnitedHealth", "CVS Health", "Cigna"],
            "disruptors": ["Teladoc", "Livongo", "Ro", "Hims", "23andMe"],
        },
    }

    DISRUPTION_SIGNALS = [
        "disrupts", "disrupting", "disruptive", "transforms", "transforming",
        "replaces", "replacing", "obsolete", "threatens", "challenging",
        "revolutionizes", "revolutionizing", "game-changer", "breakthrough"
    ]

    MOMENTUM_SIGNALS = [
        "funding", "raised", "valuation", "Series A", "Series B", "IPO",
        "partnership", "acquisition", "launch", "expansion", "growth"
    ]

    def __init__(self):
        self.results = {}

    def analyze_theme(self, theme_name, config, days=30):
        """Analyze a disruption theme comprehensively."""

        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        analysis = {
            "theme": theme_name,
            "keywords": config["keywords"],
            "metrics": {},
            "incumbent_analysis": {},
            "disruptor_analysis": {},
            "top_stories": [],
        }

        # Overall theme coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(config["keywords"]),
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        analysis["metrics"]["total_coverage"] = len(resp.json().get("results", []))

        # Disruption signal intensity
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(config["keywords"] + self.DISRUPTION_SIGNALS),
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        analysis["metrics"]["disruption_signals"] = len(resp.json().get("results", []))

        # Momentum signals
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(config["keywords"] + self.MOMENTUM_SIGNALS),
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        analysis["metrics"]["momentum_signals"] = len(resp.json().get("results", []))

        # Tier-1 coverage
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(config["keywords"]),
            "source.rank.opr.min": 5,
            "published_at.start": start,
            "per_page": 1,
        })
        analysis["metrics"]["tier1_coverage"] = len(resp.json().get("results", []))

        # Analyze incumbents
        for incumbent in config["incumbents"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": incumbent,
                "title": ",".join(config["keywords"]),
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            total = len(resp.json().get("results", []))

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": incumbent,
                "title": ",".join(config["keywords"] + self.DISRUPTION_SIGNALS),
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            disruption = len(resp.json().get("results", []))

            analysis["incumbent_analysis"][incumbent] = {
                "coverage": total,
                "disruption_mentions": disruption,
                "disruption_ratio": disruption / max(total, 1),
            }

        # Analyze disruptors
        for disruptor in config["disruptors"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": disruptor,
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            total = len(resp.json().get("results", []))

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": disruptor,
                "title": ",".join(self.MOMENTUM_SIGNALS),
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            momentum = len(resp.json().get("results", []))

            analysis["disruptor_analysis"][disruptor] = {
                "coverage": total,
                "momentum_mentions": momentum,
                "momentum_ratio": momentum / max(total, 1),
            }

        # Get top stories
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(config["keywords"]),
            "source.rank.opr.min": 4,
            "published_at.start": start,
            "language.code": "en",
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 10,
        })

        analysis["top_stories"] = [
            {
                "title": a["title"],
                "source": a["source"]["domain"],
                "date": a["published_at"][:10],
                "sentiment": a.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
            }
            for a in resp.json().get("results", [])
        ]

        # Calculate composite disruption score
        total = analysis["metrics"]["total_coverage"] or 1
        analysis["metrics"]["disruption_intensity"] = analysis["metrics"]["disruption_signals"] / total
        analysis["metrics"]["momentum_intensity"] = analysis["metrics"]["momentum_signals"] / total
        analysis["metrics"]["tier1_ratio"] = analysis["metrics"]["tier1_coverage"] / total

        # Weighted score
        analysis["disruption_score"] = (
            analysis["metrics"]["disruption_intensity"] * 35 +
            analysis["metrics"]["momentum_intensity"] * 30 +
            analysis["metrics"]["tier1_ratio"] * 20 +
            min(0.15, total / 10000)  # Volume bonus
        ) * 100

        return analysis

    def scan_all_themes(self, days=30):
        """Scan all disruption themes."""

        for theme_name, config in self.DISRUPTION_THEMES.items():
            self.results[theme_name] = self.analyze_theme(theme_name, config, days)

        return self.results

    def generate_report(self):
        """Generate comprehensive disruption report."""

        if not self.results:
            self.scan_all_themes()

        # Sort by disruption score
        sorted_themes = sorted(
            self.results.items(),
            key=lambda x: x[1]["disruption_score"],
            reverse=True
        )

        print("=" * 80)
        print("INDUSTRY DISRUPTION RADAR - COMPREHENSIVE REPORT")
        print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)

        # Summary table
        print("\nDISRUPTION INTENSITY RANKING:")
        print("-" * 70)
        print(f"{'Theme':<25} {'Coverage':>10} {'Disruption':>12} {'Momentum':>12} {'Score':>10}")
        print("-" * 70)

        for theme, data in sorted_themes:
            print(f"{theme:<25} {data['metrics']['total_coverage']:>10,} "
                  f"{data['metrics']['disruption_signals']:>12,} "
                  f"{data['metrics']['momentum_signals']:>12,} "
                  f"{data['disruption_score']:>9.0f}")

        # Detailed analysis for top themes
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS - TOP 3 DISRUPTION THEMES")
        print("=" * 80)

        for theme, data in sorted_themes[:3]:
            print(f"\n{'='*60}")
            print(f"{theme.upper()}")
            print(f"Disruption Score: {data['disruption_score']:.0f}/100")
            print(f"{'='*60}")

            print(f"\nMetrics:")
            print(f"  Total Coverage: {data['metrics']['total_coverage']:,}")
            print(f"  Disruption Intensity: {data['metrics']['disruption_intensity']:.1%}")
            print(f"  Momentum Intensity: {data['metrics']['momentum_intensity']:.1%}")
            print(f"  Tier-1 Coverage: {data['metrics']['tier1_ratio']:.1%}")

            # Incumbent analysis
            print(f"\nIncumbent Response:")
            for name, metrics in sorted(
                data["incumbent_analysis"].items(),
                key=lambda x: x[1]["coverage"],
                reverse=True
            ):
                if metrics["coverage"] > 0:
                    print(f"  {name}: {metrics['coverage']} articles, "
                          f"{metrics['disruption_ratio']:.1%} disruption mentions")

            # Disruptor analysis
            print(f"\nDisruptor Momentum:")
            for name, metrics in sorted(
                data["disruptor_analysis"].items(),
                key=lambda x: x[1]["coverage"],
                reverse=True
            ):
                if metrics["coverage"] > 0:
                    print(f"  {name}: {metrics['coverage']} articles, "
                          f"{metrics['momentum_ratio']:.1%} momentum mentions")

            # Top stories
            print(f"\nRecent Headlines:")
            for story in data["top_stories"][:3]:
                icon = "📈" if story["sentiment"] == "positive" else \
                      "📉" if story["sentiment"] == "negative" else "📰"
                print(f"  {icon} [{story['source']}] {story['title'][:50]}...")

        return self.results


# Run the radar
radar = DisruptionRadar()
radar.generate_report()
```

### Startup Funding Tracker

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

FUNDING_KEYWORDS = {
    "seed": ["seed round", "seed funding", "pre-seed"],
    "series_a": ["Series A", "Series A funding", "Series A round"],
    "series_b": ["Series B", "Series B funding", "Series B round"],
    "series_c_plus": ["Series C", "Series D", "Series E", "late-stage"],
    "ipo": ["IPO", "goes public", "public offering", "SPAC"],
}

SECTORS = ["AI", "fintech", "healthtech", "cleantech", "SaaS", "crypto"]

def track_funding_activity(days=14):
    """Track startup funding activity across sectors."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    results = {
        "by_stage": {},
        "by_sector": {},
        "top_deals": [],
    }

    # By funding stage
    for stage, keywords in FUNDING_KEYWORDS.items():
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": ",".join(keywords),
            "published_at.start": start,
            "language.code": "en",
            "source.rank.opr.min": 4,
            "per_page": 1,
        })
        results["by_stage"][stage] = len(resp.json().get("results", []))

    # By sector
    for sector in SECTORS:
        all_funding_keywords = [kw for keywords in FUNDING_KEYWORDS.values() for kw in keywords]

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "title": f"{sector}," + ",".join(all_funding_keywords[:5]),
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        results["by_sector"][sector] = len(resp.json().get("results", []))

    # Get top deals (recent high-authority funding news)
    all_funding = [kw for keywords in FUNDING_KEYWORDS.values() for kw in keywords]
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "title": ",".join(all_funding[:10]),
        "source.rank.opr.min": 4,
        "published_at.start": start,
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 15,
    })

    for article in resp.json().get("results", []):
        entities = article.get("entities", [])
        companies = [e["name"] for e in entities if e.get("type") == "organization"]

        results["top_deals"].append({
            "title": article["title"],
            "source": article["source"]["domain"],
            "date": article["published_at"][:10],
            "companies": companies[:3],
        })

    return results

print("STARTUP FUNDING TRACKER")
print("=" * 60)
print(f"Period: Last 14 days\n")

funding = track_funding_activity(days=14)

print("FUNDING BY STAGE:")
print("-" * 40)

for stage, count in sorted(funding["by_stage"].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (count // 5)
    print(f"  {stage:<15}: {count:>5} {bar}")

print("\nFUNDING BY SECTOR:")
print("-" * 40)

for sector, count in sorted(funding["by_sector"].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (count // 3)
    print(f"  {sector:<15}: {count:>5} {bar}")

print("\nTOP RECENT DEALS:")
print("-" * 60)

for deal in funding["top_deals"][:10]:
    companies = ", ".join(deal["companies"]) if deal["companies"] else "N/A"
    print(f"\n  {deal['title'][:55]}...")
    print(f"    Companies: {companies}")
    print(f"    Source: {deal['source']} | {deal['date']}")
```

---

## JavaScript

### Disruption Trend Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const DISRUPTION_THEMES = {
  "Generative AI": ["ChatGPT", "generative AI", "LLM", "GPT-4"],
  "Electric Vehicles": ["EV", "electric vehicle", "Tesla", "battery"],
  "Fintech": ["fintech", "neobank", "digital payments"],
  "Space Tech": ["SpaceX", "rocket", "satellite", "launch"],
};

const DISRUPTION_SIGNALS = ["disrupts", "transforms", "replaces", "revolutionizes"];

async function analyzeDisruptionTheme(themeName, keywords, days = 30) {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  // Total coverage
  const totalParams = new URLSearchParams({
    api_key: API_KEY,
    title: keywords.join(","),
    "published_at.start": start,
    "language.code": "en",
    per_page: "1",
  });

  let response = await fetch(`${BASE_URL}?${totalParams}`);
  let data = await response.json();
  const totalCoverage = data.results?.length || 0;

  // Disruption signals
  const signalParams = new URLSearchParams({
    api_key: API_KEY,
    title: [...keywords, ...DISRUPTION_SIGNALS].join(","),
    "published_at.start": start,
    "language.code": "en",
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${signalParams}`);
  data = await response.json();
  const disruptionSignals = data.results?.length || 0;

  // Tier-1 coverage
  const tier1Params = new URLSearchParams({
    api_key: API_KEY,
    title: keywords.join(","),
    "source.rank.opr.min": "5",
    "published_at.start": start,
    per_page: "1",
  });

  response = await fetch(`${BASE_URL}?${tier1Params}`);
  data = await response.json();
  const tier1Coverage = data.results?.length || 0;

  // Calculate scores
  const disruptionIntensity = disruptionSignals / Math.max(totalCoverage, 1);
  const tier1Ratio = tier1Coverage / Math.max(totalCoverage, 1);

  const disruptionScore = (
    disruptionIntensity * 50 +
    tier1Ratio * 30 +
    Math.min(0.2, totalCoverage / 5000)
  ) * 100;

  return {
    theme: themeName,
    totalCoverage,
    disruptionSignals,
    tier1Coverage,
    disruptionIntensity,
    disruptionScore,
  };
}

async function runDisruptionRadar() {
  console.log("DISRUPTION TREND MONITOR");
  console.log("=".repeat(60));
  console.log(`Analysis Period: Last 30 days\n`);

  const results = [];

  for (const [theme, keywords] of Object.entries(DISRUPTION_THEMES)) {
    const analysis = await analyzeDisruptionTheme(theme, keywords, 30);
    results.push(analysis);
  }

  results.sort((a, b) => b.disruptionScore - a.disruptionScore);

  console.log(`${"Theme".padEnd(20)} ${"Coverage".padStart(10)} ${"Signals".padStart(10)} ${"Tier-1".padStart(10)} ${"Score".padStart(10)}`);
  console.log("-".repeat(65));

  for (const r of results) {
    const emoji = r.disruptionScore >= 50 ? "🔥" : r.disruptionScore >= 30 ? "📈" : "📊";
    console.log(
      `${emoji} ${r.theme.padEnd(18)} ${String(r.totalCoverage).padStart(10)} ` +
      `${String(r.disruptionSignals).padStart(10)} ${String(r.tier1Coverage).padStart(10)} ` +
      `${r.disruptionScore.toFixed(0).padStart(10)}`
    );
  }

  return results;
}

runDisruptionRadar();
```

---

## PHP

### Industry Disruption Scanner

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$themes = [
    "Generative AI"      => ["ChatGPT", "generative AI", "LLM", "GPT-4"],
    "Electric Vehicles"  => ["EV", "electric vehicle", "Tesla", "battery"],
    "Fintech"            => ["fintech", "neobank", "digital payments"],
    "Space Tech"         => ["SpaceX", "rocket", "satellite"],
];

$disruptionSignals = ["disrupts", "transforms", "replaces", "revolutionizes"];

function analyzeTheme(string $theme, array $keywords, int $days = 30): array
{
    global $apiKey, $baseUrl, $disruptionSignals;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    // Total coverage
    $query = http_build_query([
        "api_key"            => $apiKey,
        "title"              => implode(",", $keywords),
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $totalCoverage = count($data["results"] ?? []);

    // Disruption signals
    $query = http_build_query([
        "api_key"            => $apiKey,
        "title"              => implode(",", array_merge($keywords, $disruptionSignals)),
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $signals = count($data["results"] ?? []);

    // Tier-1 coverage
    $query = http_build_query([
        "api_key"             => $apiKey,
        "title"               => implode(",", $keywords),
        "source.rank.opr.min" => 5,
        "published_at.start"  => $start,
        "per_page"            => 1,
    ]);
    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $tier1 = count($data["results"] ?? []);

    $total = max($totalCoverage, 1);
    $intensity = $signals / $total;
    $tier1Ratio = $tier1 / $total;

    $score = ($intensity * 50 + $tier1Ratio * 30 + min(0.2, $totalCoverage / 5000)) * 100;

    return [
        "theme"     => $theme,
        "coverage"  => $totalCoverage,
        "signals"   => $signals,
        "tier1"     => $tier1,
        "intensity" => $intensity,
        "score"     => $score,
    ];
}

echo "INDUSTRY DISRUPTION SCANNER\n";
echo str_repeat("=", 65) . "\n";
echo "Analysis Period: Last 30 days\n\n";

$results = [];
foreach ($themes as $theme => $keywords) {
    $results[] = analyzeTheme($theme, $keywords, 30);
}

usort($results, fn($a, $b) => $b["score"] <=> $a["score"]);

printf("%-20s %10s %10s %10s %10s\n",
    "Theme", "Coverage", "Signals", "Tier-1", "Score");
echo str_repeat("-", 65) . "\n";

foreach ($results as $r) {
    $emoji = $r["score"] >= 50 ? "🔥" : ($r["score"] >= 30 ? "📈" : "📊");
    printf("%s %-18s %10d %10d %10d %10.0f\n",
        $emoji, $r["theme"], $r["coverage"], $r["signals"], $r["tier1"], $r["score"]);
}
```
