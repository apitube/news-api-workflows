# ESG Media Intelligence

Workflow for tracking Environmental, Social, and Governance media coverage, calculating ESG sentiment scores, detecting greenwashing signals, monitoring controversy events, and benchmarking ESG reputation across companies using the [APITube News API](https://apitube.io).

## Overview

The **ESG Media Intelligence** workflow provides comprehensive ESG-focused media analysis by tracking coverage across Environmental, Social, and Governance dimensions, detecting controversy events, identifying greenwashing patterns, calculating ESG reputation scores, and benchmarking against industry peers. Combines multi-dimensional scoring, controversy detection, and temporal trend analysis. Ideal for ESG analysts, sustainability teams, investors, and corporate communications.

## API Endpoints

```
GET https://api.apitube.io/v1/news/everything
```

## Key Parameters

| Parameter                      | Type    | Description                                                          |
|-------------------------------|---------|----------------------------------------------------------------------|
| `api_key`                     | string  | **Required.** Your API key.                                          |
| `entity.name`                 | string  | Filter by company name.                                              |
| `title`                       | string  | Filter by ESG-related keywords.                                      |
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
# Track environmental coverage
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=ExxonMobil&title=climate,carbon,emissions,environment&language=en&per_page=30"

# Monitor social issues
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Amazon&title=workers,labor,diversity,safety&language=en&per_page=30"

# Track governance controversies
curl -s "https://api.apitube.io/v1/news/everything?api_key=YOUR_API_KEY&entity.name=Meta&title=board,executive,ethics,compliance,scandal&sentiment.overall.polarity=negative&language=en&per_page=30"
```

### Python

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

# ESG keyword taxonomy
ESG_TAXONOMY = {
    "environmental": {
        "climate": ["climate", "carbon", "emissions", "greenhouse", "net zero", "carbon neutral"],
        "pollution": ["pollution", "toxic", "waste", "contamination", "spill"],
        "resources": ["renewable", "sustainable", "recycling", "biodiversity", "deforestation"],
        "energy": ["clean energy", "solar", "wind", "fossil fuel", "oil", "coal"],
    },
    "social": {
        "labor": ["workers", "labor", "wages", "union", "strike", "working conditions"],
        "diversity": ["diversity", "inclusion", "DEI", "discrimination", "gender", "racial"],
        "safety": ["safety", "injury", "accident", "health", "workplace"],
        "community": ["community", "philanthropy", "donation", "social impact"],
        "human_rights": ["human rights", "forced labor", "child labor", "supply chain ethics"],
    },
    "governance": {
        "leadership": ["CEO", "board", "executive", "leadership", "management"],
        "ethics": ["ethics", "corruption", "bribery", "fraud", "misconduct"],
        "compliance": ["compliance", "regulation", "fine", "penalty", "lawsuit"],
        "transparency": ["transparency", "disclosure", "reporting", "audit"],
        "shareholder": ["shareholder", "investor", "proxy", "vote", "activism"],
    }
}

CONTROVERSY_SIGNALS = [
    "scandal", "controversy", "investigation", "lawsuit", "fine", "penalty",
    "accused", "alleged", "violation", "misconduct", "breach", "whistleblower"
]

class ESGAnalyzer:
    """Comprehensive ESG media analysis."""

    def __init__(self, company):
        self.company = company
        self.scores = {"E": {}, "S": {}, "G": {}}
        self.controversies = []

    def analyze_dimension(self, dimension, categories, days=30):
        """Analyze a single ESG dimension."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        results = {}

        for category, keywords in categories.items():
            # Total coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.company,
                "title": ",".join(keywords),
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            total = resp.json().get("total_results", 0)

            # Positive coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.company,
                "title": ",".join(keywords),
                "sentiment.overall.polarity": "positive",
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            positive = resp.json().get("total_results", 0)

            # Negative coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.company,
                "title": ",".join(keywords),
                "sentiment.overall.polarity": "negative",
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            negative = resp.json().get("total_results", 0)

            # Tier-1 coverage
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "entity.name": self.company,
                "title": ",".join(keywords),
                "source.rank.opr.min": 0.7,
                "published_at.start": start,
                "language": "en",
                "per_page": 1,
            })
            tier1 = resp.json().get("total_results", 0)

            results[category] = {
                "total": total,
                "positive": positive,
                "negative": negative,
                "tier1": tier1,
                "sentiment_score": (positive - negative) / max(total, 1),
                "tier1_share": tier1 / max(total, 1)
            }

        return results

    def calculate_esg_scores(self, days=30):
        """Calculate comprehensive ESG scores."""
        # Environmental
        self.scores["E"] = self.analyze_dimension("E", ESG_TAXONOMY["environmental"], days)

        # Social
        self.scores["S"] = self.analyze_dimension("S", ESG_TAXONOMY["social"], days)

        # Governance
        self.scores["G"] = self.analyze_dimension("G", ESG_TAXONOMY["governance"], days)

        return self.scores

    def calculate_composite_score(self):
        """Calculate composite ESG score (0-100)."""
        dimension_scores = {}

        for dim, categories in self.scores.items():
            if not categories:
                dimension_scores[dim] = 50
                continue

            # Weighted by coverage volume
            total_coverage = sum(c["total"] for c in categories.values())
            if total_coverage == 0:
                dimension_scores[dim] = 50
                continue

            weighted_sentiment = sum(
                c["sentiment_score"] * c["total"]
                for c in categories.values()
            ) / total_coverage

            # Convert from [-1, 1] to [0, 100]
            dimension_scores[dim] = (weighted_sentiment + 1) * 50

        # Composite (equal weights)
        composite = sum(dimension_scores.values()) / 3

        return {
            "composite": composite,
            "E": dimension_scores.get("E", 50),
            "S": dimension_scores.get("S", 50),
            "G": dimension_scores.get("G", 50)
        }

    def detect_controversies(self, days=30):
        """Detect ESG-related controversies."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.company,
            "title": ",".join(CONTROVERSY_SIGNALS),
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "source.rank.opr.min": 0.6,
            "language": "en",
            "sort.by": "published_at",
            "sort.order": "desc",
            "per_page": 50,
        })

        articles = resp.json().get("results", [])

        controversies = []
        for article in articles:
            title = article.get("title", "").lower()

            # Categorize controversy
            category = "general"
            for dim, cats in ESG_TAXONOMY.items():
                for cat, keywords in cats.items():
                    if any(kw.lower() in title for kw in keywords):
                        category = f"{dim[0].upper()}:{cat}"
                        break

            controversies.append({
                "date": article.get("published_at", "")[:10],
                "title": article.get("title"),
                "source": article.get("source", {}).get("domain"),
                "category": category,
                "url": article.get("url")
            })

        self.controversies = controversies
        return controversies

    def detect_greenwashing_signals(self, days=30):
        """Detect potential greenwashing patterns."""
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Positive environmental claims
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.company,
            "title": "sustainable,green,eco-friendly,carbon neutral,net zero,renewable",
            "sentiment.overall.polarity": "positive",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        green_claims = resp.json().get("total_results", 0)

        # Environmental criticism
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.company,
            "title": "pollution,emissions,environmental damage,greenwashing,misleading",
            "sentiment.overall.polarity": "negative",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        green_criticism = resp.json().get("total_results", 0)

        # Greenwashing mentions
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "entity.name": self.company,
            "title": "greenwashing,misleading,false claims,exaggerated",
            "published_at.start": start,
            "language": "en",
            "per_page": 1,
        })
        greenwashing_mentions = resp.json().get("total_results", 0)

        ratio = green_claims / max(green_criticism, 1)

        return {
            "green_claims": green_claims,
            "green_criticism": green_criticism,
            "greenwashing_mentions": greenwashing_mentions,
            "claim_criticism_ratio": ratio,
            "risk_level": "high" if greenwashing_mentions > 5 or ratio > 10 else "medium" if ratio > 5 else "low"
        }

# Run ESG analysis
company = "Shell"

print("ESG MEDIA INTELLIGENCE")
print("=" * 70)
print(f"Company: {company}")
print(f"Analysis Period: Last 30 days\n")

analyzer = ESGAnalyzer(company)

# Calculate scores
print("Calculating ESG scores...")
scores = analyzer.calculate_esg_scores(days=30)

# Display by dimension
for dim, name in [("E", "Environmental"), ("S", "Social"), ("G", "Governance")]:
    print(f"\n{name.upper()}")
    print("-" * 40)
    for category, data in scores[dim].items():
        sentiment_indicator = "+" if data["sentiment_score"] > 0.1 else "-" if data["sentiment_score"] < -0.1 else "○"
        print(f"  {category}: {data['total']} articles [{sentiment_indicator}] "
              f"(Pos: {data['positive']}, Neg: {data['negative']})")

# Composite score
composite = analyzer.calculate_composite_score()
print(f"\nCOMPOSITE ESG SCORE: {composite['composite']:.1f}/100")
print(f"  Environmental: {composite['E']:.1f}")
print(f"  Social: {composite['S']:.1f}")
print(f"  Governance: {composite['G']:.1f}")

# Controversies
print("\n" + "=" * 50)
print("CONTROVERSY DETECTION")
print("-" * 40)
controversies = analyzer.detect_controversies()
print(f"Controversies found: {len(controversies)}")
for c in controversies[:5]:
    print(f"  [{c['category']}] {c['date']}: {c['title'][:60]}...")

# Greenwashing
print("\n" + "=" * 50)
print("GREENWASHING ANALYSIS")
print("-" * 40)
greenwashing = analyzer.detect_greenwashing_signals()
print(f"Green claims: {greenwashing['green_claims']}")
print(f"Green criticism: {greenwashing['green_criticism']}")
print(f"Greenwashing mentions: {greenwashing['greenwashing_mentions']}")
print(f"Risk level: {greenwashing['risk_level'].upper()}")
```

### JavaScript (Node.js)

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const ESG_KEYWORDS = {
  E: {
    climate: ["climate", "carbon", "emissions", "net zero"],
    pollution: ["pollution", "toxic", "waste", "spill"],
    resources: ["renewable", "sustainable", "recycling"],
  },
  S: {
    labor: ["workers", "labor", "wages", "union"],
    diversity: ["diversity", "inclusion", "DEI", "discrimination"],
    safety: ["safety", "injury", "accident", "workplace"],
  },
  G: {
    ethics: ["ethics", "corruption", "fraud", "misconduct"],
    compliance: ["compliance", "fine", "penalty", "lawsuit"],
    leadership: ["CEO", "board", "executive", "management"],
  },
};

class ESGAnalyzer {
  constructor(company) {
    this.company = company;
    this.scores = { E: {}, S: {}, G: {} };
  }

  async analyzeDimension(dim, categories, days = 30) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    for (const [category, keywords] of Object.entries(categories)) {
      // Total
      let params = new URLSearchParams({
        api_key: API_KEY,
        "entity.name": this.company,
        title: keywords.join(","),
        "published_at.start": start,
        language: "en",
        per_page: "1",
      });

      let response = await fetch(`${BASE_URL}?${params}`);
      let data = await response.json();
      const total = data.total_results || 0;

      // Positive
      params.set("sentiment.overall.polarity", "positive");
      response = await fetch(`${BASE_URL}?${params}`);
      data = await response.json();
      const positive = data.total_results || 0;

      // Negative
      params.set("sentiment.overall.polarity", "negative");
      response = await fetch(`${BASE_URL}?${params}`);
      data = await response.json();
      const negative = data.total_results || 0;

      this.scores[dim][category] = {
        total,
        positive,
        negative,
        sentimentScore: total > 0 ? (positive - negative) / total : 0,
      };
    }
  }

  async calculateScores(days = 30) {
    await this.analyzeDimension("E", ESG_KEYWORDS.E, days);
    await this.analyzeDimension("S", ESG_KEYWORDS.S, days);
    await this.analyzeDimension("G", ESG_KEYWORDS.G, days);
    return this.scores;
  }

  getCompositeScore() {
    const dimensionScores = {};

    for (const [dim, categories] of Object.entries(this.scores)) {
      const cats = Object.values(categories);
      if (cats.length === 0) {
        dimensionScores[dim] = 50;
        continue;
      }

      const totalCoverage = cats.reduce((s, c) => s + c.total, 0);
      if (totalCoverage === 0) {
        dimensionScores[dim] = 50;
        continue;
      }

      const weightedSentiment =
        cats.reduce((s, c) => s + c.sentimentScore * c.total, 0) / totalCoverage;
      dimensionScores[dim] = (weightedSentiment + 1) * 50;
    }

    return {
      composite: (dimensionScores.E + dimensionScores.S + dimensionScores.G) / 3,
      ...dimensionScores,
    };
  }

  async detectControversies(days = 30) {
    const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    const params = new URLSearchParams({
      api_key: API_KEY,
      "entity.name": this.company,
      title: "scandal,controversy,investigation,lawsuit,fine,violation",
      "sentiment.overall.polarity": "negative",
      "published_at.start": start,
      language: "en",
      per_page: "30",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();

    return (data.results || []).map((a) => ({
      date: (a.published_at || "").slice(0, 10),
      title: a.title,
      source: a.source?.domain,
    }));
  }
}

async function runAnalysis() {
  const company = "Shell";
  const analyzer = new ESGAnalyzer(company);

  console.log("ESG MEDIA INTELLIGENCE");
  console.log("=".repeat(50));
  console.log(`Company: ${company}\n`);

  console.log("Calculating ESG scores...");
  await analyzer.calculateScores(30);

  for (const [dim, name] of [
    ["E", "Environmental"],
    ["S", "Social"],
    ["G", "Governance"],
  ]) {
    console.log(`\n${name}`);
    console.log("-".repeat(30));
    for (const [cat, data] of Object.entries(analyzer.scores[dim])) {
      const indicator =
        data.sentimentScore > 0.1 ? "+" : data.sentimentScore < -0.1 ? "-" : "○";
      console.log(`  ${cat}: ${data.total} [${indicator}]`);
    }
  }

  const composite = analyzer.getCompositeScore();
  console.log(`\nCOMPOSITE: ${composite.composite.toFixed(1)}/100`);
  console.log(`  E: ${composite.E.toFixed(1)} | S: ${composite.S.toFixed(1)} | G: ${composite.G.toFixed(1)}`);

  console.log("\nControversies:");
  const controversies = await analyzer.detectControversies();
  controversies.slice(0, 5).forEach((c) => {
    console.log(`  ${c.date}: ${c.title?.slice(0, 50)}...`);
  });
}

runAnalysis();
```

### PHP

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$esgKeywords = [
    "E" => [
        "climate" => ["climate", "carbon", "emissions", "net zero"],
        "pollution" => ["pollution", "toxic", "waste"],
        "resources" => ["renewable", "sustainable", "recycling"],
    ],
    "S" => [
        "labor" => ["workers", "labor", "wages", "union"],
        "diversity" => ["diversity", "inclusion", "DEI"],
        "safety" => ["safety", "injury", "accident"],
    ],
    "G" => [
        "ethics" => ["ethics", "corruption", "fraud"],
        "compliance" => ["compliance", "fine", "penalty"],
        "leadership" => ["CEO", "board", "executive"],
    ],
];

class ESGAnalyzer
{
    private string $apiKey;
    private string $baseUrl;
    private string $company;
    private array $scores = ["E" => [], "S" => [], "G" => []];

    public function __construct(string $company)
    {
        global $apiKey, $baseUrl;
        $this->apiKey = $apiKey;
        $this->baseUrl = $baseUrl;
        $this->company = $company;
    }

    public function analyzeDimension(string $dim, array $categories, int $days = 30): void
    {
        $start = (new DateTime("-{$days} days"))->format("Y-m-d");

        foreach ($categories as $category => $keywords) {
            // Total
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "entity.name" => $this->company,
                "title" => implode(",", $keywords),
                "published_at.start" => $start,
                "language" => "en",
                "per_page" => 1,
            ]);
            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $total = $data["total_results"] ?? 0;

            // Positive
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "entity.name" => $this->company,
                "title" => implode(",", $keywords),
                "sentiment.overall.polarity" => "positive",
                "published_at.start" => $start,
                "language" => "en",
                "per_page" => 1,
            ]);
            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $positive = $data["total_results"] ?? 0;

            // Negative
            $query = http_build_query([
                "api_key" => $this->apiKey,
                "entity.name" => $this->company,
                "title" => implode(",", $keywords),
                "sentiment.overall.polarity" => "negative",
                "published_at.start" => $start,
                "language" => "en",
                "per_page" => 1,
            ]);
            $data = json_decode(file_get_contents("{$this->baseUrl}?{$query}"), true);
            $negative = $data["total_results"] ?? 0;

            $this->scores[$dim][$category] = [
                "total" => $total,
                "positive" => $positive,
                "negative" => $negative,
                "sentiment" => $total > 0 ? ($positive - $negative) / $total : 0,
            ];
        }
    }

    public function calculateScores(int $days = 30): array
    {
        global $esgKeywords;

        $this->analyzeDimension("E", $esgKeywords["E"], $days);
        $this->analyzeDimension("S", $esgKeywords["S"], $days);
        $this->analyzeDimension("G", $esgKeywords["G"], $days);

        return $this->scores;
    }

    public function getComposite(): array
    {
        $dimScores = [];

        foreach ($this->scores as $dim => $categories) {
            $totalCoverage = array_sum(array_column($categories, "total"));
            if ($totalCoverage === 0) {
                $dimScores[$dim] = 50;
                continue;
            }

            $weighted = 0;
            foreach ($categories as $cat) {
                $weighted += $cat["sentiment"] * $cat["total"];
            }
            $dimScores[$dim] = ($weighted / $totalCoverage + 1) * 50;
        }

        return [
            "composite" => array_sum($dimScores) / 3,
            "E" => $dimScores["E"],
            "S" => $dimScores["S"],
            "G" => $dimScores["G"],
        ];
    }
}

$company = "Shell";
$analyzer = new ESGAnalyzer($company);

echo "ESG MEDIA INTELLIGENCE\n";
echo str_repeat("=", 50) . "\n";
echo "Company: {$company}\n\n";

$scores = $analyzer->calculateScores(30);

foreach (["E" => "Environmental", "S" => "Social", "G" => "Governance"] as $dim => $name) {
    echo "{$name}\n";
    echo str_repeat("-", 30) . "\n";
    foreach ($scores[$dim] as $cat => $data) {
        $ind = $data["sentiment"] > 0.1 ? "+" : ($data["sentiment"] < -0.1 ? "-" : "○");
        echo "  {$cat}: {$data['total']} [{$ind}]\n";
    }
}

$composite = $analyzer->getComposite();
printf("\nCOMPOSITE: %.1f/100\n", $composite["composite"]);
printf("  E: %.1f | S: %.1f | G: %.1f\n", $composite["E"], $composite["S"], $composite["G"]);
```

## Common Use Cases

- **ESG scoring** — calculate media-based ESG scores for companies.
- **Controversy monitoring** — detect and track ESG-related controversies.
- **Greenwashing detection** — identify patterns suggesting greenwashing.
- **Peer benchmarking** — compare ESG media profiles across competitors.
- **Investor due diligence** — assess ESG risks before investment.
- **Sustainability reporting** — support ESG reports with media analysis.
- **Regulatory risk assessment** — track compliance and governance issues.
- **Stakeholder sentiment** — understand public perception of ESG efforts.

## See Also

- [examples.md](./examples.md) — detailed code examples for ESG media intelligence.
