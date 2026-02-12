# ESG Media Intelligence — Examples

Advanced code examples for ESG scoring, controversy detection, greenwashing analysis, and peer benchmarking.

---

## Python — Comprehensive ESG Analysis Platform

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

# Comprehensive ESG taxonomy
ESG_TAXONOMY = {
    "environmental": {
        "climate_change": {
            "keywords": ["climate change", "global warming", "carbon footprint", "greenhouse gas", "emissions reduction"],
            "weight": 0.25
        },
        "net_zero": {
            "keywords": ["net zero", "carbon neutral", "decarbonization", "climate commitment", "science-based targets"],
            "weight": 0.20
        },
        "pollution": {
            "keywords": ["pollution", "toxic", "contamination", "hazardous waste", "air quality", "water quality"],
            "weight": 0.20
        },
        "biodiversity": {
            "keywords": ["biodiversity", "deforestation", "habitat", "ecosystem", "species protection"],
            "weight": 0.15
        },
        "circular_economy": {
            "keywords": ["recycling", "circular economy", "waste reduction", "sustainable packaging", "reuse"],
            "weight": 0.10
        },
        "renewable_energy": {
            "keywords": ["renewable energy", "solar", "wind power", "clean energy", "energy transition"],
            "weight": 0.10
        }
    },
    "social": {
        "labor_practices": {
            "keywords": ["labor rights", "working conditions", "fair wages", "employee welfare", "workplace safety"],
            "weight": 0.25
        },
        "diversity_inclusion": {
            "keywords": ["diversity", "inclusion", "equity", "DEI", "gender equality", "racial diversity"],
            "weight": 0.20
        },
        "human_rights": {
            "keywords": ["human rights", "forced labor", "child labor", "modern slavery", "supply chain ethics"],
            "weight": 0.20
        },
        "community_impact": {
            "keywords": ["community investment", "social impact", "philanthropy", "local development"],
            "weight": 0.15
        },
        "health_safety": {
            "keywords": ["health and safety", "occupational safety", "worker protection", "safety standards"],
            "weight": 0.10
        },
        "customer_welfare": {
            "keywords": ["consumer protection", "product safety", "data privacy", "customer rights"],
            "weight": 0.10
        }
    },
    "governance": {
        "board_composition": {
            "keywords": ["board diversity", "independent directors", "board oversight", "corporate governance"],
            "weight": 0.20
        },
        "executive_compensation": {
            "keywords": ["executive pay", "CEO compensation", "pay ratio", "incentive alignment"],
            "weight": 0.15
        },
        "ethics_compliance": {
            "keywords": ["business ethics", "anti-corruption", "compliance", "code of conduct", "whistleblower"],
            "weight": 0.25
        },
        "transparency": {
            "keywords": ["transparency", "disclosure", "reporting", "accountability", "audit"],
            "weight": 0.20
        },
        "shareholder_rights": {
            "keywords": ["shareholder rights", "proxy voting", "investor engagement", "shareholder activism"],
            "weight": 0.10
        },
        "risk_management": {
            "keywords": ["risk management", "internal controls", "cybersecurity", "operational risk"],
            "weight": 0.10
        }
    }
}

CONTROVERSY_INDICATORS = {
    "high_severity": [
        "scandal", "criminal", "fraud", "bribery", "corruption", "violation",
        "lawsuit", "investigation", "penalty", "fine", "sanction"
    ],
    "medium_severity": [
        "controversy", "criticism", "concern", "complaint", "dispute",
        "allegation", "accused", "questioned"
    ],
    "greenwashing": [
        "greenwashing", "misleading", "false claims", "exaggerated",
        "deceptive", "unsubstantiated"
    ]
}


class ESGMediaPlatform:
    """Comprehensive ESG media analysis platform."""

    def __init__(self):
        self.company_profiles = {}

    def analyze_company(self, company, days=30):
        """Full ESG analysis for a company."""
        profile = {
            "company": company,
            "analysis_date": datetime.utcnow().isoformat(),
            "period_days": days,
            "dimensions": {},
            "controversies": [],
            "greenwashing_risk": {},
            "scores": {}
        }

        # Analyze each dimension
        for dim_name, categories in ESG_TAXONOMY.items():
            profile["dimensions"][dim_name] = self._analyze_dimension(
                company, categories, days
            )

        # Detect controversies
        profile["controversies"] = self._detect_controversies(company, days)

        # Greenwashing analysis
        profile["greenwashing_risk"] = self._analyze_greenwashing(company, days)

        # Calculate scores
        profile["scores"] = self._calculate_scores(profile)

        self.company_profiles[company] = profile
        return profile

    def _analyze_dimension(self, company, categories, days):
        """Analyze a single ESG dimension."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        results = {}

        for cat_name, cat_config in categories.items():
            keywords = cat_config["keywords"]

            # Total coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": company,
                "title": ",".join(keywords),
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            total = resp.json().get("total_results", 0)

            # Sentiment breakdown
            sentiment_data = {}
            for polarity in ["positive", "negative", "neutral"]:
                resp = requests.get(BASE_URL, params={
                    "api_key": API_KEY,
                    "entity.name": company,
                    "title": ",".join(keywords),
                    "sentiment.overall.polarity": polarity,
                    "published_at.start": start,
                    "language": "en",
                    "per_page": 1,
                })
                sentiment_data[polarity] = resp.json().get("total_results", 0)

            # Source tier analysis
            tier1_resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": company,
                "title": ",".join(keywords),
                "source.rank.opr.min": 0.7,
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            tier1_count = tier1_resp.json().get("total_results", 0)

            # Calculate metrics
            sentiment_score = 0
            if total > 0:
                sentiment_score = (sentiment_data["positive"] - sentiment_data["negative"]) / total

            results[cat_name] = {
                "total_coverage": total,
                "positive": sentiment_data["positive"],
                "negative": sentiment_data["negative"],
                "neutral": sentiment_data["neutral"],
                "tier1_coverage": tier1_count,
                "tier1_share": tier1_count / max(total, 1),
                "sentiment_score": sentiment_score,
                "weight": cat_config["weight"]
            }

        return results

    def _detect_controversies(self, company, days):
        """Detect and categorize controversies."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        controversies = []

        # High severity
        for severity, indicators in CONTROVERSY_INDICATORS.items():
            if severity == "greenwashing":
                continue

            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": company,
                "title": ",".join(indicators),
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "source.rank.opr.min": 0.6,
                "language": "en",
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 30,
            })

            for article in resp.json().get("results", []):
                title = article.get("title", "").lower()

                # Categorize by ESG dimension
                esg_category = self._categorize_controversy(title)

                controversies.append({
                    "date": article.get("published_at", "")[:10],
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("domain"),
                    "severity": severity,
                    "esg_category": esg_category,
                    "url": article.get("url")
                })

        # Sort by date
        controversies.sort(key=lambda x: x["date"], reverse=True)

        return controversies

    def _categorize_controversy(self, title):
        """Categorize controversy by ESG dimension."""
        for dim_name, categories in ESG_TAXONOMY.items():
            for cat_name, cat_config in categories.items():
                if any(kw.lower() in title for kw in cat_config["keywords"]):
                    return f"{dim_name[0].upper()}:{cat_name}"
        return "general"

    def _analyze_greenwashing(self, company, days):
        """Analyze greenwashing risk."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Positive environmental claims
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": company,
            "title": "sustainable,green,eco-friendly,carbon neutral,net zero,clean,renewable",
            "sentiment.overall.polarity": "positive",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        positive_claims = resp.json().get("total_results", 0)

        # Environmental criticism
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": company,
            "title": "pollution,emissions,environmental,damage,harmful,toxic",
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        env_criticism = resp.json().get("total_results", 0)

        # Direct greenwashing accusations
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": company,
            "title": ",".join(CONTROVERSY_INDICATORS["greenwashing"]),
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        greenwashing_mentions = resp.json().get("total_results", 0)

        # Calculate risk score
        claim_ratio = positive_claims / max(env_criticism, 1)
        risk_score = 0

        if greenwashing_mentions > 10:
            risk_score = 90
        elif greenwashing_mentions > 5:
            risk_score = 70
        elif claim_ratio > 10 and positive_claims > 20:
            risk_score = 60
        elif claim_ratio > 5:
            risk_score = 40
        else:
            risk_score = 20

        return {
            "positive_claims": positive_claims,
            "environmental_criticism": env_criticism,
            "greenwashing_mentions": greenwashing_mentions,
            "claim_criticism_ratio": claim_ratio,
            "risk_score": risk_score,
            "risk_level": "high" if risk_score >= 70 else "medium" if risk_score >= 40 else "low"
        }

    def _calculate_scores(self, profile):
        """Calculate comprehensive ESG scores."""
        scores = {}

        for dim_name in ["environmental", "social", "governance"]:
            dim_data = profile["dimensions"].get(dim_name, {})
            if not dim_data:
                scores[dim_name] = 50
                continue

            # Weighted average
            total_weight = sum(cat["weight"] for cat in dim_data.values())
            weighted_sentiment = sum(
                cat["sentiment_score"] * cat["weight"]
                for cat in dim_data.values()
            ) / total_weight if total_weight > 0 else 0

            # Convert [-1, 1] to [0, 100]
            base_score = (weighted_sentiment + 1) * 50

            # Controversy penalty
            dim_prefix = dim_name[0].upper()
            relevant_controversies = [
                c for c in profile["controversies"]
                if c["esg_category"].startswith(dim_prefix)
            ]

            high_severity = sum(1 for c in relevant_controversies if c["severity"] == "high_severity")
            medium_severity = sum(1 for c in relevant_controversies if c["severity"] == "medium_severity")

            penalty = high_severity * 5 + medium_severity * 2
            scores[dim_name] = max(0, min(100, base_score - penalty))

        # Composite score
        scores["composite"] = (
            scores.get("environmental", 50) * 0.35 +
            scores.get("social", 50) * 0.35 +
            scores.get("governance", 50) * 0.30
        )

        # Risk-adjusted score
        greenwashing_penalty = profile["greenwashing_risk"]["risk_score"] * 0.1
        scores["risk_adjusted"] = max(0, scores["composite"] - greenwashing_penalty)

        return scores

    def compare_companies(self, companies, days=30):
        """Compare ESG profiles across companies."""
        comparison = {"companies": {}, "rankings": {}}

        for company in companies:
            profile = self.analyze_company(company, days)
            comparison["companies"][company] = {
                "scores": profile["scores"],
                "controversy_count": len(profile["controversies"]),
                "greenwashing_risk": profile["greenwashing_risk"]["risk_level"]
            }

        # Generate rankings
        for metric in ["composite", "environmental", "social", "governance", "risk_adjusted"]:
            ranked = sorted(
                comparison["companies"].items(),
                key=lambda x: x[1]["scores"].get(metric, 0),
                reverse=True
            )
            comparison["rankings"][metric] = [
                {"company": c, "score": d["scores"].get(metric, 0)}
                for c, d in ranked
            ]

        return comparison

    def generate_report(self, company):
        """Generate comprehensive ESG report."""
        profile = self.company_profiles.get(company)
        if not profile:
            profile = self.analyze_company(company)

        report = {
            "executive_summary": {
                "company": company,
                "composite_score": profile["scores"]["composite"],
                "risk_adjusted_score": profile["scores"]["risk_adjusted"],
                "controversy_count": len(profile["controversies"]),
                "greenwashing_risk": profile["greenwashing_risk"]["risk_level"]
            },
            "dimension_scores": {
                "environmental": profile["scores"]["environmental"],
                "social": profile["scores"]["social"],
                "governance": profile["scores"]["governance"]
            },
            "top_controversies": profile["controversies"][:5],
            "recommendations": []
        }

        # Generate recommendations
        if profile["scores"]["environmental"] < 40:
            report["recommendations"].append(
                "Environmental score below threshold - review environmental practices"
            )
        if profile["greenwashing_risk"]["risk_level"] == "high":
            report["recommendations"].append(
                "High greenwashing risk detected - verify environmental claims"
            )
        if len(profile["controversies"]) > 10:
            report["recommendations"].append(
                "Elevated controversy count - conduct detailed risk assessment"
            )

        return report


# Run comprehensive analysis
print("ESG MEDIA INTELLIGENCE PLATFORM")
print("=" * 70)

platform = ESGMediaPlatform()

# Single company analysis
company = "Shell"
print(f"\nAnalyzing {company}...")
profile = platform.analyze_company(company, days=30)

print(f"\nESG SCORES FOR {company.upper()}")
print("-" * 50)
print(f"Composite Score: {profile['scores']['composite']:.1f}/100")
print(f"Risk-Adjusted Score: {profile['scores']['risk_adjusted']:.1f}/100")
print(f"\n  Environmental: {profile['scores']['environmental']:.1f}")
print(f"  Social: {profile['scores']['social']:.1f}")
print(f"  Governance: {profile['scores']['governance']:.1f}")

print(f"\nGREENWASHING ANALYSIS")
print("-" * 50)
gw = profile["greenwashing_risk"]
print(f"Risk Level: {gw['risk_level'].upper()}")
print(f"Positive claims: {gw['positive_claims']}")
print(f"Environmental criticism: {gw['environmental_criticism']}")
print(f"Greenwashing mentions: {gw['greenwashing_mentions']}")

print(f"\nCONTROVERSIES ({len(profile['controversies'])} total)")
print("-" * 50)
for c in profile["controversies"][:5]:
    print(f"  [{c['severity']}] {c['esg_category']}")
    print(f"    {c['date']}: {c['title'][:60]}...")

# Company comparison
print("\n" + "=" * 70)
print("PEER COMPARISON")
print("-" * 50)

companies = ["Shell", "BP", "ExxonMobil"]
comparison = platform.compare_companies(companies, days=30)

print("\nComposite Score Ranking:")
for i, item in enumerate(comparison["rankings"]["composite"], 1):
    print(f"  {i}. {item['company']}: {item['score']:.1f}")

print("\nRisk-Adjusted Ranking:")
for i, item in enumerate(comparison["rankings"]["risk_adjusted"], 1):
    print(f"  {i}. {item['company']}: {item['score']:.1f}")

# Generate report
print("\n" + "=" * 70)
print("EXECUTIVE REPORT")
print("-" * 50)
report = platform.generate_report(company)
print(f"Company: {report['executive_summary']['company']}")
print(f"Composite: {report['executive_summary']['composite_score']:.1f}")
print(f"Controversies: {report['executive_summary']['controversy_count']}")
print(f"Greenwashing Risk: {report['executive_summary']['greenwashing_risk']}")

if report["recommendations"]:
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
```

---

## JavaScript — ESG Monitoring Dashboard

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const ESG_CATEGORIES = {
  E: {
    climate: { keywords: ["climate", "carbon", "emissions", "net zero"], weight: 0.4 },
    pollution: { keywords: ["pollution", "toxic", "waste", "contamination"], weight: 0.3 },
    resources: { keywords: ["renewable", "sustainable", "recycling"], weight: 0.3 },
  },
  S: {
    labor: { keywords: ["workers", "labor", "wages", "safety"], weight: 0.35 },
    diversity: { keywords: ["diversity", "inclusion", "DEI", "equity"], weight: 0.35 },
    community: { keywords: ["community", "philanthropy", "social impact"], weight: 0.3 },
  },
  G: {
    ethics: { keywords: ["ethics", "corruption", "fraud", "compliance"], weight: 0.4 },
    transparency: { keywords: ["transparency", "disclosure", "reporting"], weight: 0.3 },
    leadership: { keywords: ["board", "executive", "governance"], weight: 0.3 },
  },
};

class ESGDashboard {
  constructor() {
    this.companyData = new Map();
    this.alerts = [];
  }

  async analyzeCompany(company, days = 30) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    const data = {
      company,
      timestamp: new Date().toISOString(),
      dimensions: { E: {}, S: {}, G: {} },
      scores: {},
      controversies: [],
    };

    // Analyze each dimension
    for (const [dim, categories] of Object.entries(ESG_CATEGORIES)) {
      for (const [cat, config] of Object.entries(categories)) {
        // Total
        let params = new URLSearchParams({
          api_key: API_KEY,
          "entity.name": company,
          title: config.keywords.join(","),
          "published_at.start": start,
          language: "en",
          per_page: "1",
        });

        let response = await fetch(`${BASE_URL}?${params}`);
        let result = await response.json();
        const total = result.total_results || 0;

        // Positive
        params.set("sentiment.overall.polarity", "positive");
        response = await fetch(`${BASE_URL}?${params}`);
        result = await response.json();
        const positive = result.total_results || 0;

        // Negative
        params.set("sentiment.overall.polarity", "negative");
        response = await fetch(`${BASE_URL}?${params}`);
        result = await response.json();
        const negative = result.total_results || 0;

        data.dimensions[dim][cat] = {
          total,
          positive,
          negative,
          sentiment: total > 0 ? (positive - negative) / total : 0,
          weight: config.weight,
        };
      }
    }

    // Calculate scores
    for (const dim of ["E", "S", "G"]) {
      const cats = Object.values(data.dimensions[dim]);
      const totalWeight = cats.reduce((s, c) => s + c.weight, 0);

      if (totalWeight > 0) {
        const weightedSentiment =
          cats.reduce((s, c) => s + c.sentiment * c.weight, 0) / totalWeight;
        data.scores[dim] = (weightedSentiment + 1) * 50;
      } else {
        data.scores[dim] = 50;
      }
    }

    data.scores.composite = (data.scores.E + data.scores.S + data.scores.G) / 3;

    // Detect controversies
    data.controversies = await this.detectControversies(company, start);

    // Apply controversy penalty
    const penalty = Math.min(data.controversies.length * 2, 20);
    data.scores.adjusted = Math.max(0, data.scores.composite - penalty);

    this.companyData.set(company, data);
    this.checkAlerts(data);

    return data;
  }

  async detectControversies(company, start) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": company,
      title: "scandal,controversy,lawsuit,fine,violation,investigation",
      "sentiment.overall.polarity": "negative",
      "published_at.start": start,
      language: "en",
      per_page: "20",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();

    return (data.results || []).map((a) => ({
      date: (a.published_at || "").slice(0, 10),
      title: a.title,
      source: a.source?.domain,
    }));
  }

  checkAlerts(data) {
    // Low score alert
    if (data.scores.composite < 40) {
      this.alerts.push({
        type: "LOW_SCORE",
        company: data.company,
        message: `ESG score below 40: ${data.scores.composite.toFixed(1)}`,
        severity: "HIGH",
        timestamp: new Date().toISOString(),
      });
    }

    // Controversy spike
    if (data.controversies.length > 5) {
      this.alerts.push({
        type: "CONTROVERSY_SPIKE",
        company: data.company,
        message: `${data.controversies.length} controversies detected`,
        severity: "MEDIUM",
        timestamp: new Date().toISOString(),
      });
    }

    // Dimension imbalance
    const scores = [data.scores.E, data.scores.S, data.scores.G];
    const range = Math.max(...scores) - Math.min(...scores);
    if (range > 30) {
      this.alerts.push({
        type: "DIMENSION_IMBALANCE",
        company: data.company,
        message: `Large ESG dimension gap: ${range.toFixed(1)} points`,
        severity: "LOW",
        timestamp: new Date().toISOString(),
      });
    }
  }

  async compareCompanies(companies, days = 30) {
    const results = [];

    for (const company of companies) {
      const data = await this.analyzeCompany(company, days);
      results.push({
        company,
        composite: data.scores.composite,
        adjusted: data.scores.adjusted,
        E: data.scores.E,
        S: data.scores.S,
        G: data.scores.G,
        controversies: data.controversies.length,
      });
    }

    return results.sort((a, b) => b.adjusted - a.adjusted);
  }

  getAlerts(count = 10) {
    return this.alerts.slice(-count).reverse();
  }
}

async function runDashboard() {
  const dashboard = new ESGDashboard();

  console.log("ESG MONITORING DASHBOARD");
  console.log("=".repeat(60));

  // Analyze single company
  const company = "Shell";
  console.log(`\nAnalyzing ${company}...`);
  const data = await dashboard.analyzeCompany(company, 30);

  console.log(`\nESG SCORES: ${company}`);
  console.log("-".repeat(40));
  console.log(`Composite: ${data.scores.composite.toFixed(1)}`);
  console.log(`Adjusted: ${data.scores.adjusted.toFixed(1)}`);
  console.log(`  E: ${data.scores.E.toFixed(1)}`);
  console.log(`  S: ${data.scores.S.toFixed(1)}`);
  console.log(`  G: ${data.scores.G.toFixed(1)}`);
  console.log(`Controversies: ${data.controversies.length}`);

  // Compare companies
  console.log("\n" + "=".repeat(60));
  console.log("PEER COMPARISON");
  console.log("-".repeat(40));

  const comparison = await dashboard.compareCompanies(
    ["Shell", "BP", "TotalEnergies"],
    30
  );

  comparison.forEach((c, i) => {
    console.log(
      `${i + 1}. ${c.company}: ${c.adjusted.toFixed(1)} ` +
        `(E:${c.E.toFixed(0)} S:${c.S.toFixed(0)} G:${c.G.toFixed(0)}) ` +
        `[${c.controversies} controversies]`
    );
  });

  // Alerts
  console.log("\n" + "=".repeat(60));
  console.log("ALERTS");
  console.log("-".repeat(40));

  const alerts = dashboard.getAlerts();
  if (alerts.length === 0) {
    console.log("No alerts");
  } else {
    alerts.forEach((a) => {
      console.log(`[${a.severity}] ${a.company}: ${a.type}`);
      console.log(`  ${a.message}`);
    });
  }
}

runDashboard();
```

---

## PHP — ESG Reporting Service

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$esgCategories = [
    "E" => [
        "climate" => ["keywords" => ["climate", "carbon", "emissions"], "weight" => 0.4],
        "pollution" => ["keywords" => ["pollution", "toxic", "waste"], "weight" => 0.3],
        "resources" => ["keywords" => ["renewable", "sustainable"], "weight" => 0.3],
    ],
    "S" => [
        "labor" => ["keywords" => ["workers", "labor", "wages"], "weight" => 0.35],
        "diversity" => ["keywords" => ["diversity", "inclusion", "DEI"], "weight" => 0.35],
        "community" => ["keywords" => ["community", "philanthropy"], "weight" => 0.3],
    ],
    "G" => [
        "ethics" => ["keywords" => ["ethics", "corruption", "compliance"], "weight" => 0.4],
        "transparency" => ["keywords" => ["transparency", "disclosure"], "weight" => 0.3],
        "leadership" => ["keywords" => ["board", "executive"], "weight" => 0.3],
    ],
];

class ESGReportingService
{
    private string $apiKey;
    private string $baseUrl;
    private array $companyData = [];

    public function __construct()
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
    }

    public function analyzeCompany(string $company, int $days = 30): array
    {
        global $esgCategories;

        $start = (new DateTime("-{$days} days"))->format("Y-m-d");
        $data = [
            "company" => $company,
            "dimensions" => ["E" => [], "S" => [], "G" => []],
            "scores" => [],
        ];

        foreach ($esgCategories as $dim => $categories) {
            foreach ($categories as $cat => $config) {
                // Total
                $query = http_build_query([
                    "api_key" => $this->apiKey,
                    "entity.name" => $company,
                    "title" => implode(",", $config["keywords"]),
                    "published_at.start" => $start,
                    "language" => "en",
                    "per_page" => 1,
                ]);
                $result = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
                $total = $result["total_results"] ?? 0;

                // Positive
                $query = http_build_query([
                    "api_key" => $this->apiKey,
                    "entity.name" => $company,
                    "title" => implode(",", $config["keywords"]),
                    "sentiment.overall.polarity" => "positive",
                    "published_at.start" => $start,
                    "language" => "en",
                    "per_page" => 1,
                ]);
                $result = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
                $positive = $result["total_results"] ?? 0;

                // Negative
                $query = http_build_query([
                    "api_key" => $this->apiKey,
                    "entity.name" => $company,
                    "title" => implode(",", $config["keywords"]),
                    "sentiment.overall.polarity" => "negative",
                    "published_at.start" => $start,
                    "language" => "en",
                    "per_page" => 1,
                ]);
                $result = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
                $negative = $result["total_results"] ?? 0;

                $data["dimensions"][$dim][$cat] = [
                    "total" => $total,
                    "positive" => $positive,
                    "negative" => $negative,
                    "sentiment" => $total > 0 ? ($positive - $negative) / $total : 0,
                    "weight" => $config["weight"],
                ];
            }
        }

        // Calculate scores
        foreach (["E", "S", "G"] as $dim) {
            $cats = $data["dimensions"][$dim];
            $totalWeight = array_sum(array_column($cats, "weight"));

            if ($totalWeight > 0) {
                $weighted = 0;
                foreach ($cats as $cat) {
                    $weighted += $cat["sentiment"] * $cat["weight"];
                }
                $data["scores"][$dim] = ($weighted / $totalWeight + 1) * 50;
            } else {
                $data["scores"][$dim] = 50;
            }
        }

        $data["scores"]["composite"] = ($data["scores"]["E"] + $data["scores"]["S"] + $data["scores"]["G"]) / 3;

        $this->companyData[$company] = $data;
        return $data;
    }

    public function compareCompanies(array $companies, int $days = 30): array
    {
        $results = [];

        foreach ($companies as $company) {
            $data = $this->analyzeCompany($company, $days);
            $results[] = [
                "company" => $company,
                "composite" => $data["scores"]["composite"],
                "E" => $data["scores"]["E"],
                "S" => $data["scores"]["S"],
                "G" => $data["scores"]["G"],
            ];
        }

        usort($results, fn($a, $b) => $b["composite"] <=> $a["composite"]);
        return $results;
    }

    public function generateReport(string $company): array
    {
        $data = $this->companyData[$company] ?? $this->analyzeCompany($company);

        return [
            "company" => $company,
            "generated_at" => (new DateTime())->format("c"),
            "scores" => $data["scores"],
            "dimension_details" => $data["dimensions"],
            "rating" => $this->getESGRating($data["scores"]["composite"]),
        ];
    }

    private function getESGRating(float $score): string
    {
        if ($score >= 80) return "AAA";
        if ($score >= 70) return "AA";
        if ($score >= 60) return "A";
        if ($score >= 50) return "BBB";
        if ($score >= 40) return "BB";
        if ($score >= 30) return "B";
        return "CCC";
    }
}

// Run analysis
$service = new ESGReportingService();

echo "ESG REPORTING SERVICE\n";
echo str_repeat("=", 60) . "\n";

$company = "Shell";
$data = $service->analyzeCompany($company, 30);

echo "\nESG SCORES: {$company}\n";
echo str_repeat("-", 40) . "\n";
printf("Composite: %.1f\n", $data["scores"]["composite"]);
printf("  E: %.1f | S: %.1f | G: %.1f\n",
    $data["scores"]["E"],
    $data["scores"]["S"],
    $data["scores"]["G"]
);

// Comparison
echo "\n" . str_repeat("=", 60) . "\n";
echo "PEER COMPARISON\n";
echo str_repeat("-", 40) . "\n";

$comparison = $service->compareCompanies(["Shell", "BP", "ExxonMobil"], 30);
foreach ($comparison as $i => $c) {
    printf("%d. %s: %.1f (E:%.0f S:%.0f G:%.0f)\n",
        $i + 1,
        $c["company"],
        $c["composite"],
        $c["E"],
        $c["S"],
        $c["G"]
    );
}

// Report
echo "\n" . str_repeat("=", 60) . "\n";
echo "REPORT\n";
$report = $service->generateReport($company);
printf("Company: %s\n", $report["company"]);
printf("Rating: %s\n", $report["rating"]);
printf("Composite: %.1f\n", $report["scores"]["composite"]);
```

---

## See Also

- [README.md](./README.md) — ESG Media Intelligence workflow overview and quick start.
