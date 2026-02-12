# Investment Research — Code Examples

Detailed examples for building investment research and quantitative analysis systems using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Earnings Calendar Sentiment Tracker

```python
import requests
from datetime import datetime, timedelta
from typing import Dict, List

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

FINANCIAL_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com,marketwatch.com,seekingalpha.com"

EARNINGS_CALENDAR = [
    {"symbol": "NVDA", "name": "NVIDIA", "date": "2026-02-26"},
    {"symbol": "AAPL", "name": "Apple", "date": "2026-01-30"},
    {"symbol": "MSFT", "name": "Microsoft", "date": "2026-01-28"},
    {"symbol": "GOOGL", "name": "Alphabet", "date": "2026-02-04"},
    {"symbol": "AMZN", "name": "Amazon", "date": "2026-02-06"},
    {"symbol": "META", "name": "Meta", "date": "2026-02-05"},
]

def analyze_pre_earnings_sentiment(company: str, earnings_date: str, days_before: int = 14) -> Dict:
    """Analyze sentiment in the days leading up to earnings."""

    end = datetime.fromisoformat(earnings_date)
    start = end - timedelta(days=days_before)

    # Get all articles
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "source.domain": FINANCIAL_SOURCES,
        "published_at.start": start.strftime("%Y-%m-%d"),
        "published_at.end": end.strftime("%Y-%m-%d"),
        "language.code": "en",
        "per_page": 100,
    })
    data = response.json()
    total = len(data.get("results", []))

    # Get sentiment breakdown
    sentiments = {}
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": company,
            "sentiment.overall.polarity": polarity,
            "source.domain": FINANCIAL_SOURCES,
            "published_at.start": start.strftime("%Y-%m-%d"),
            "published_at.end": end.strftime("%Y-%m-%d"),
            "language.code": "en",
            "per_page": 1,
        })
        sentiments[polarity] = len(resp.json().get("results", []))

    # Calculate metrics
    total_sentiment = sum(sentiments.values()) or 1
    net_sentiment = (sentiments["positive"] - sentiments["negative"]) / total_sentiment

    # Get earnings-specific coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "title": "earnings,revenue,guidance,forecast,results",
        "source.domain": FINANCIAL_SOURCES,
        "published_at.start": start.strftime("%Y-%m-%d"),
        "published_at.end": end.strftime("%Y-%m-%d"),
        "language.code": "en",
        "per_page": 1,
    })
    earnings_mentions = len(resp.json().get("results", []))

    return {
        "company": company,
        "earnings_date": earnings_date,
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "total_articles": total,
        "sentiments": sentiments,
        "net_sentiment": net_sentiment,
        "earnings_mentions": earnings_mentions,
        "signal": "BULLISH" if net_sentiment > 0.1 else "BEARISH" if net_sentiment < -0.1 else "NEUTRAL",
    }

def generate_earnings_report():
    """Generate a comprehensive pre-earnings sentiment report."""

    print("=" * 80)
    print("PRE-EARNINGS SENTIMENT ANALYSIS")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    results = []

    for company in EARNINGS_CALENDAR:
        analysis = analyze_pre_earnings_sentiment(
            company["name"],
            company["date"],
            days_before=14
        )
        results.append({**company, **analysis})

    # Sort by net sentiment
    results.sort(key=lambda x: x["net_sentiment"], reverse=True)

    print(f"\n{'Symbol':<8} {'Company':<12} {'Date':<12} {'Articles':>10} "
          f"{'Pos':>6} {'Neg':>6} {'Net':>8} {'Signal':<10}")
    print("-" * 80)

    for r in results:
        symbol_indicator = "📈" if r["signal"] == "BULLISH" else "📉" if r["signal"] == "BEARISH" else "➡️"
        print(f"{r['symbol']:<8} {r['name']:<12} {r['date']:<12} {r['total_articles']:>10} "
              f"{r['sentiments']['positive']:>6} {r['sentiments']['negative']:>6} "
              f"{r['net_sentiment']:>+7.3f} {symbol_indicator} {r['signal']:<8}")

    return results

# Run the report
generate_earnings_report()
```

### Sector Rotation Analyzer

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

SECTORS = {
    "Technology": ["Apple", "Microsoft", "NVIDIA", "Google", "Meta"],
    "Healthcare": ["Johnson & Johnson", "UnitedHealth", "Pfizer", "Eli Lilly", "Merck"],
    "Financials": ["JPMorgan", "Bank of America", "Goldman Sachs", "Visa", "Mastercard"],
    "Energy": ["ExxonMobil", "Chevron", "Shell", "ConocoPhillips", "Schlumberger"],
    "Consumer": ["Amazon", "Tesla", "Walmart", "Home Depot", "Nike"],
}

def calculate_sector_sentiment(sector_name: str, companies: list, days: int = 7) -> dict:
    """Calculate aggregate sentiment for a sector."""

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    total_positive = 0
    total_negative = 0
    total_neutral = 0
    total_articles = 0

    for company in companies:
        for polarity in ["positive", "negative", "neutral"]:
            resp = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": company,
                "sentiment.overall.polarity": polarity,
                "published_at.start": start,
                "language.code": "en",
                "per_page": 1,
            })
            count = len(resp.json().get("results", []))

            if polarity == "positive":
                total_positive += count
            elif polarity == "negative":
                total_negative += count
            else:
                total_neutral += count

            total_articles += count

    total = total_positive + total_negative + total_neutral or 1
    net_sentiment = (total_positive - total_negative) / total

    return {
        "sector": sector_name,
        "companies": len(companies),
        "total_articles": total_articles,
        "positive": total_positive,
        "negative": total_negative,
        "neutral": total_neutral,
        "net_sentiment": net_sentiment,
        "positive_ratio": total_positive / total,
        "negative_ratio": total_negative / total,
    }

def detect_rotation(current_week: dict, previous_week: dict) -> dict:
    """Detect sector rotation by comparing week-over-week sentiment changes."""

    rotation = {}

    for sector in current_week:
        if sector in previous_week:
            sentiment_change = current_week[sector]["net_sentiment"] - previous_week[sector]["net_sentiment"]
            rotation[sector] = {
                "current": current_week[sector]["net_sentiment"],
                "previous": previous_week[sector]["net_sentiment"],
                "change": sentiment_change,
                "signal": "INFLOW" if sentiment_change > 0.05 else "OUTFLOW" if sentiment_change < -0.05 else "STABLE",
            }

    return rotation

print("SECTOR ROTATION ANALYSIS")
print("=" * 70)
print(f"Analysis Date: {datetime.utcnow().strftime('%Y-%m-%d')}")
print()

# Calculate current week sentiment
current_week = {}
for sector_name, companies in SECTORS.items():
    current_week[sector_name] = calculate_sector_sentiment(sector_name, companies, days=7)

# Sort by net sentiment
sorted_sectors = sorted(current_week.items(), key=lambda x: x[1]["net_sentiment"], reverse=True)

print(f"{'Sector':<15} {'Articles':>10} {'Positive':>10} {'Negative':>10} {'Net':>10} {'Signal'}")
print("-" * 70)

for sector_name, data in sorted_sectors:
    signal = "🟢" if data["net_sentiment"] > 0.1 else "🔴" if data["net_sentiment"] < -0.1 else "🟡"
    print(f"{sector_name:<15} {data['total_articles']:>10} {data['positive']:>10} "
          f"{data['negative']:>10} {data['net_sentiment']:>+9.3f} {signal}")

print("\nTop Performing Sectors:")
for i, (sector, data) in enumerate(sorted_sectors[:3], 1):
    print(f"  {i}. {sector}: {data['net_sentiment']:+.3f}")

print("\nUnderperforming Sectors:")
for i, (sector, data) in enumerate(sorted_sectors[-3:], 1):
    print(f"  {i}. {sector}: {data['net_sentiment']:+.3f}")
```

### M&A Rumor Detection System

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

MA_KEYWORDS = [
    "acquisition", "merger", "takeover", "buyout", "deal",
    "bid", "target", "acquires", "merges", "acquire",
    "private equity", "LBO", "consolidation"
]

PREMIUM_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com"

def scan_ma_rumors(hours: int = 24) -> list:
    """Scan for M&A rumors and potential deal activity."""

    start = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

    # Get M&A related news from premium sources
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": "mergers_acquisitions",
        "source.domain": PREMIUM_SOURCES,
        "published_at.start": start,
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 50,
    })

    articles = response.json().get("results", [])
    rumors = []

    for article in articles:
        # Extract mentioned entities
        entities = article.get("entities", [])
        organizations = [e for e in entities if e.get("type") == "organization"]

        if len(organizations) >= 2:
            # Potential deal involving multiple companies
            rumor = {
                "title": article["title"],
                "source": article["source"]["domain"],
                "published_at": article["published_at"],
                "url": article["href"],
                "sentiment": article.get("sentiment", {}).get("overall", {}).get("polarity", "neutral"),
                "companies": [org["name"] for org in organizations[:4]],
                "score": article["source"].get("rank", {}).get("opr", 0),
            }
            rumors.append(rumor)

    # Sort by source authority
    rumors.sort(key=lambda x: x["score"], reverse=True)

    return rumors

def group_by_company(rumors: list) -> dict:
    """Group rumors by mentioned companies."""

    by_company = defaultdict(list)

    for rumor in rumors:
        for company in rumor["companies"]:
            by_company[company].append(rumor)

    # Sort by mention frequency
    sorted_companies = sorted(by_company.items(), key=lambda x: len(x[1]), reverse=True)

    return dict(sorted_companies)

print("M&A RUMOR DETECTION SYSTEM")
print("=" * 70)
print(f"Scan Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Lookback: 24 hours")
print()

rumors = scan_ma_rumors(hours=24)

print(f"Found {len(rumors)} potential M&A-related articles\n")

if rumors:
    print("TOP M&A RUMORS (by source authority):")
    print("-" * 70)

    for i, rumor in enumerate(rumors[:10], 1):
        companies = ", ".join(rumor["companies"][:3])
        sentiment_icon = "📈" if rumor["sentiment"] == "positive" else "📉" if rumor["sentiment"] == "negative" else "➡️"

        print(f"\n{i}. {rumor['title'][:70]}...")
        print(f"   Companies: {companies}")
        print(f"   Source: {rumor['source']} (OPR: {rumor['score']:.2f}) {sentiment_icon}")
        print(f"   Time: {rumor['published_at'][:19]}")

    print("\n" + "=" * 70)
    print("COMPANIES WITH MOST M&A MENTIONS:")
    print("-" * 70)

    by_company = group_by_company(rumors)

    for company, company_rumors in list(by_company.items())[:10]:
        print(f"  {company}: {len(company_rumors)} mentions")
```

### Quantitative Sentiment Signal Generator

```python
import requests
from datetime import datetime, timedelta
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

UNIVERSE = [
    "Apple", "Microsoft", "NVIDIA", "Google", "Amazon",
    "Meta", "Tesla", "JPMorgan", "Visa", "Mastercard",
]

FINANCIAL_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com,marketwatch.com"

def calculate_sentiment_signal(stock: str, lookback_days: int = 7) -> dict:
    """Calculate quantitative sentiment signals for a stock."""

    start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    signals = {
        "stock": stock,
        "timestamp": datetime.utcnow().isoformat(),
        "lookback_days": lookback_days,
    }

    # Get total coverage
    resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": stock,
        "source.domain": FINANCIAL_SOURCES,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    signals["total_articles"] = len(resp.json().get("results", []))

    # Get sentiment counts
    for polarity in ["positive", "negative", "neutral"]:
        resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": stock,
            "sentiment.overall.polarity": polarity,
            "source.domain": FINANCIAL_SOURCES,
            "published_at.start": start,
            "language.code": "en",
            "per_page": 1,
        })
        signals[f"{polarity}_count"] = len(resp.json().get("results", []))

    # Calculate derived signals
    total = signals["positive_count"] + signals["negative_count"] + signals["neutral_count"] or 1

    # 1. Net Sentiment Score (-1 to +1)
    signals["net_sentiment"] = (signals["positive_count"] - signals["negative_count"]) / total

    # 2. Sentiment Momentum (positive ratio)
    signals["positive_ratio"] = signals["positive_count"] / total

    # 3. Controversy Score (negative ratio)
    signals["controversy_score"] = signals["negative_count"] / total

    # 4. Coverage Intensity (articles per day)
    signals["coverage_intensity"] = signals["total_articles"] / lookback_days

    # 5. Sentiment Strength (absolute sentiment)
    signals["sentiment_strength"] = abs(signals["net_sentiment"])

    # Generate trading signal
    if signals["net_sentiment"] > 0.15 and signals["coverage_intensity"] > 5:
        signals["trading_signal"] = "STRONG_BUY"
    elif signals["net_sentiment"] > 0.05:
        signals["trading_signal"] = "BUY"
    elif signals["net_sentiment"] < -0.15 and signals["coverage_intensity"] > 5:
        signals["trading_signal"] = "STRONG_SELL"
    elif signals["net_sentiment"] < -0.05:
        signals["trading_signal"] = "SELL"
    else:
        signals["trading_signal"] = "HOLD"

    return signals

print("QUANTITATIVE SENTIMENT SIGNALS")
print("=" * 80)
print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Universe: {len(UNIVERSE)} stocks")
print()

all_signals = []

for stock in UNIVERSE:
    signal = calculate_sentiment_signal(stock, lookback_days=7)
    all_signals.append(signal)

# Sort by net sentiment
all_signals.sort(key=lambda x: x["net_sentiment"], reverse=True)

print(f"{'Stock':<12} {'Articles':>10} {'Pos':>6} {'Neg':>6} {'Net':>8} "
      f"{'Intensity':>10} {'Signal':<12}")
print("-" * 80)

for s in all_signals:
    signal_emoji = {
        "STRONG_BUY": "🟢🟢",
        "BUY": "🟢",
        "HOLD": "🟡",
        "SELL": "🔴",
        "STRONG_SELL": "🔴🔴",
    }.get(s["trading_signal"], "")

    print(f"{s['stock']:<12} {s['total_articles']:>10} {s['positive_count']:>6} "
          f"{s['negative_count']:>6} {s['net_sentiment']:>+7.3f} "
          f"{s['coverage_intensity']:>9.1f} {signal_emoji} {s['trading_signal']:<10}")

# Output as JSON for integration
print("\n" + "=" * 80)
print("JSON OUTPUT FOR TRADING SYSTEM:")
print(json.dumps(all_signals, indent=2, default=str))
```

---

## JavaScript

### Real-Time Earnings Monitor

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const FINANCIAL_SOURCES = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

const WATCHLIST = [
  { symbol: "NVDA", name: "NVIDIA" },
  { symbol: "AAPL", name: "Apple" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "GOOGL", name: "Alphabet" },
  { symbol: "AMZN", name: "Amazon" },
];

async function getStockSentiment(company, hours = 24) {
  const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

  const sentiments = {};

  for (const polarity of ["positive", "negative", "neutral"]) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": company,
      "sentiment.overall.polarity": polarity,
      "source.domain": FINANCIAL_SOURCES,
      "published_at.start": start,
      "language.code": "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    sentiments[polarity] = data.results?.length || 0;
  }

  return sentiments;
}

async function getEarningsNews(company, hours = 24) {
  const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

  const params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": company,
    title: "earnings,revenue,guidance,forecast,beat,miss",
    "source.domain": FINANCIAL_SOURCES,
    "published_at.start": start,
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "10",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  return data.results || [];
}

async function generateDashboard() {
  console.log("=".repeat(70));
  console.log("REAL-TIME EARNINGS MONITOR");
  console.log(`Updated: ${new Date().toISOString()}`);
  console.log("=".repeat(70));

  const results = [];

  for (const stock of WATCHLIST) {
    const sentiments = await getStockSentiment(stock.name, 24);
    const earningsNews = await getEarningsNews(stock.name, 24);

    const total = sentiments.positive + sentiments.negative + sentiments.neutral || 1;
    const netSentiment = (sentiments.positive - sentiments.negative) / total;

    results.push({
      ...stock,
      sentiments,
      total,
      netSentiment,
      earningsNewsCount: earningsNews.length,
      latestHeadline: earningsNews[0]?.title || "No recent earnings news",
    });
  }

  // Sort by net sentiment
  results.sort((a, b) => b.netSentiment - a.netSentiment);

  console.log(
    `\n${"Symbol".padEnd(8)} ${"Company".padEnd(12)} ${"Pos".padStart(6)} ` +
    `${"Neg".padStart(6)} ${"Net".padStart(8)} ${"Earn News".padStart(10)}`
  );
  console.log("-".repeat(60));

  for (const r of results) {
    const signal = r.netSentiment > 0.1 ? "📈" : r.netSentiment < -0.1 ? "📉" : "➡️";
    console.log(
      `${r.symbol.padEnd(8)} ${r.name.padEnd(12)} ` +
      `${String(r.sentiments.positive).padStart(6)} ` +
      `${String(r.sentiments.negative).padStart(6)} ` +
      `${(r.netSentiment >= 0 ? "+" : "") + r.netSentiment.toFixed(3).padStart(7)} ` +
      `${String(r.earningsNewsCount).padStart(10)} ${signal}`
    );
  }

  console.log("\nLatest Headlines:");
  for (const r of results) {
    if (r.latestHeadline !== "No recent earnings news") {
      console.log(`  [${r.symbol}] ${r.latestHeadline.slice(0, 60)}...`);
    }
  }

  return results;
}

generateDashboard();
```

### Portfolio Sentiment Tracker

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

// Example portfolio with weights
const PORTFOLIO = [
  { symbol: "AAPL", name: "Apple", weight: 0.25 },
  { symbol: "MSFT", name: "Microsoft", weight: 0.20 },
  { symbol: "NVDA", name: "NVIDIA", weight: 0.15 },
  { symbol: "GOOGL", name: "Alphabet", weight: 0.15 },
  { symbol: "AMZN", name: "Amazon", weight: 0.10 },
  { symbol: "META", name: "Meta", weight: 0.10 },
  { symbol: "TSLA", name: "Tesla", weight: 0.05 },
];

async function calculatePortfolioSentiment() {
  console.log("PORTFOLIO SENTIMENT ANALYSIS");
  console.log("=".repeat(60));

  const start = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
  let portfolioSentiment = 0;
  const stockResults = [];

  for (const stock of PORTFOLIO) {
    const sentiments = { positive: 0, negative: 0, neutral: 0 };

    for (const polarity of ["positive", "negative", "neutral"]) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "organization.name": stock.name,
        "sentiment.overall.polarity": polarity,
        "published_at.start": start,
        "language.code": "en",
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      sentiments[polarity] = data.results?.length || 0;
    }

    const total = sentiments.positive + sentiments.negative + sentiments.neutral || 1;
    const netSentiment = (sentiments.positive - sentiments.negative) / total;
    const weightedSentiment = netSentiment * stock.weight;

    portfolioSentiment += weightedSentiment;

    stockResults.push({
      ...stock,
      sentiments,
      netSentiment,
      weightedSentiment,
    });
  }

  // Display results
  console.log(
    `\n${"Symbol".padEnd(8)} ${"Weight".padStart(8)} ${"Net Sent".padStart(10)} ` +
    `${"Weighted".padStart(10)} ${"Contribution"}`
  );
  console.log("-".repeat(60));

  stockResults.sort((a, b) => b.weightedSentiment - a.weightedSentiment);

  for (const r of stockResults) {
    const bar = r.weightedSentiment >= 0
      ? "█".repeat(Math.round(r.weightedSentiment * 100))
      : "░".repeat(Math.round(Math.abs(r.weightedSentiment) * 100));

    console.log(
      `${r.symbol.padEnd(8)} ${(r.weight * 100).toFixed(0).padStart(6)}% ` +
      `${(r.netSentiment >= 0 ? "+" : "") + r.netSentiment.toFixed(3).padStart(9)} ` +
      `${(r.weightedSentiment >= 0 ? "+" : "") + r.weightedSentiment.toFixed(4).padStart(9)} ` +
      `${bar}`
    );
  }

  console.log("-".repeat(60));
  const signal = portfolioSentiment > 0.05 ? "🟢 BULLISH" :
                 portfolioSentiment < -0.05 ? "🔴 BEARISH" : "🟡 NEUTRAL";
  console.log(
    `${"PORTFOLIO".padEnd(8)} ${"100%".padStart(8)} ` +
    `${"".padStart(10)} ` +
    `${(portfolioSentiment >= 0 ? "+" : "") + portfolioSentiment.toFixed(4).padStart(9)} ` +
    `${signal}`
  );

  return { portfolioSentiment, stockResults };
}

calculatePortfolioSentiment();
```

---

## PHP

### Stock Screener by Sentiment

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$universe = [
    "Apple", "Microsoft", "NVIDIA", "Google", "Amazon",
    "Meta", "Tesla", "Netflix", "AMD", "Intel",
    "Salesforce", "Adobe", "Oracle", "IBM", "Cisco",
];

$financialSources = "reuters.com,bloomberg.com,wsj.com,ft.com,cnbc.com";

function calculateStockMetrics(string $stock, int $days = 7): array
{
    global $apiKey, $baseUrl, $financialSources;

    $start = (new DateTime("-{$days} days"))->format("Y-m-d");

    $sentiments = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "organization.name"          => $stock,
            "sentiment.overall.polarity" => $polarity,
            "source.domain"              => $financialSources,
            "published_at.start"         => $start,
            "language.code"              => "en",
            "per_page"                   => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $sentiments[$polarity] = count($data["results"] ?? []);
    }

    $total = array_sum($sentiments) ?: 1;
    $netSentiment = ($sentiments["positive"] - $sentiments["negative"]) / $total;

    return [
        "stock"         => $stock,
        "total"         => $total,
        "positive"      => $sentiments["positive"],
        "negative"      => $sentiments["negative"],
        "neutral"       => $sentiments["neutral"],
        "net_sentiment" => $netSentiment,
        "signal"        => $netSentiment > 0.1 ? "BUY" : ($netSentiment < -0.1 ? "SELL" : "HOLD"),
    ];
}

echo "SENTIMENT-BASED STOCK SCREENER\n";
echo str_repeat("=", 70) . "\n";
echo "Universe: " . count($universe) . " stocks\n";
echo "Lookback: 7 days\n\n";

$results = [];
foreach ($universe as $stock) {
    $results[] = calculateStockMetrics($stock, 7);
}

// Sort by net sentiment
usort($results, fn($a, $b) => $b["net_sentiment"] <=> $a["net_sentiment"]);

printf("%-12s %8s %6s %6s %8s %8s\n",
    "Stock", "Total", "Pos", "Neg", "Net", "Signal");
echo str_repeat("-", 55) . "\n";

foreach ($results as $r) {
    $emoji = match ($r["signal"]) {
        "BUY"  => "📈",
        "SELL" => "📉",
        default => "➡️",
    };

    printf("%-12s %8d %6d %6d %+7.3f %s %s\n",
        $r["stock"],
        $r["total"],
        $r["positive"],
        $r["negative"],
        $r["net_sentiment"],
        $emoji,
        $r["signal"]
    );
}

// Summary
$buys  = count(array_filter($results, fn($r) => $r["signal"] === "BUY"));
$sells = count(array_filter($results, fn($r) => $r["signal"] === "SELL"));
$holds = count(array_filter($results, fn($r) => $r["signal"] === "HOLD"));

echo "\nSummary:\n";
echo "  BUY signals:  {$buys}\n";
echo "  SELL signals: {$sells}\n";
echo "  HOLD signals: {$holds}\n";
```

### Weekly Investment Digest Generator

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$watchlist = ["Apple", "Microsoft", "NVIDIA", "Tesla", "Amazon"];

function generateWeeklyDigest(array $stocks): string
{
    global $apiKey, $baseUrl;

    $start = (new DateTime("-7 days"))->format("Y-m-d");
    $digest = [];

    foreach ($stocks as $stock) {
        // Get sentiment
        $sentiments = [];
        foreach (["positive", "negative"] as $polarity) {
            $query = http_build_query([
                "api_key"                    => $apiKey,
                "organization.name"          => $stock,
                "sentiment.overall.polarity" => $polarity,
                "published_at.start"         => $start,
                "language.code"              => "en",
                "per_page"                   => 1,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            $sentiments[$polarity] = count($data["results"] ?? []);
        }

        // Get top headlines
        $query = http_build_query([
            "api_key"            => $apiKey,
            "organization.name"  => $stock,
            "source.rank.opr.min"=> 5,
            "published_at.start" => $start,
            "language.code"      => "en",
            "sort.by"            => "source.rank.opr",
            "sort.order"         => "desc",
            "per_page"           => 5,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $headlines = array_map(
            fn($a) => ["title" => $a["title"], "source" => $a["source"]["domain"]],
            $data["results"] ?? []
        );

        $total = $sentiments["positive"] + $sentiments["negative"] ?: 1;
        $netSentiment = ($sentiments["positive"] - $sentiments["negative"]) / $total;

        $digest[$stock] = [
            "positive"      => $sentiments["positive"],
            "negative"      => $sentiments["negative"],
            "net_sentiment" => $netSentiment,
            "trend"         => $netSentiment > 0.1 ? "📈 Bullish" :
                              ($netSentiment < -0.1 ? "📉 Bearish" : "➡️ Neutral"),
            "headlines"     => $headlines,
        ];
    }

    return formatDigest($digest);
}

function formatDigest(array $digest): string
{
    $output = "WEEKLY INVESTMENT DIGEST\n";
    $output .= str_repeat("=", 60) . "\n";
    $output .= "Period: " . (new DateTime("-7 days"))->format("Y-m-d") . " to " . date("Y-m-d") . "\n\n";

    foreach ($digest as $stock => $data) {
        $output .= "## {$stock} {$data['trend']}\n";
        $output .= sprintf("Sentiment: +%d / -%d (Net: %+.3f)\n",
            $data["positive"], $data["negative"], $data["net_sentiment"]);

        if (!empty($data["headlines"])) {
            $output .= "Top Headlines:\n";
            foreach (array_slice($data["headlines"], 0, 3) as $h) {
                $output .= "  • [{$h['source']}] " . substr($h["title"], 0, 50) . "...\n";
            }
        }

        $output .= "\n";
    }

    return $output;
}

echo generateWeeklyDigest($watchlist);
```
