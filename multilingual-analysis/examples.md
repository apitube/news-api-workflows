# Multilingual Analysis — Code Examples

Detailed examples for cross-language news monitoring, comparison, and analytics using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Cross-Language Volume Comparison

```python
import requests
import matplotlib.pyplot as plt

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko", "hi"]

def get_language_volume(topic, languages, days=7):
    from datetime import datetime, timedelta

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    volumes = {}
    for lang in languages:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "topic.id": topic,
            "language.code": lang,
            "published_at.start": start_date.strftime("%Y-%m-%d"),
            "published_at.end": end_date.strftime("%Y-%m-%d"),
            "per_page": 1,
        })
        response.raise_for_status()
        data = response.json()
        volumes[lang] = len(data.get("results", []))

    return volumes

volumes = get_language_volume("cryptocurrency", LANGUAGES)

print(f"Article volume by language for 'cryptocurrency' (last 7 days):\n")
sorted_volumes = sorted(volumes.items(), key=lambda x: x[1], reverse=True)

for lang, count in sorted_volumes:
    bar = "█" * (count // 100)
    print(f"  {lang:>3}: {count:>6} {bar}")

plt.figure(figsize=(12, 6))
plt.bar([lang for lang, _ in sorted_volumes], [count for _, count in sorted_volumes])
plt.xlabel("Language")
plt.ylabel("Article Count")
plt.title("Cross-Language Article Volume: Cryptocurrency (Last 7 Days)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("language_volume.png")
print("\nChart saved as 'language_volume.png'")
```

### Multilingual Sentiment Matrix

```python
import requests
import pandas as pd

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def build_sentiment_matrix(topic, languages, days=7):
    from datetime import datetime, timedelta

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    matrix = []

    for lang in languages:
        row = {"language": lang}

        for polarity in ["positive", "negative", "neutral"]:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "topic.id": topic,
                "language.code": lang,
                "sentiment.overall.polarity": polarity,
                "published_at.start": start_date.strftime("%Y-%m-%d"),
                "published_at.end": end_date.strftime("%Y-%m-%d"),
                "per_page": 1,
            })
            response.raise_for_status()
            data = response.json()
            row[polarity] = len(data.get("results", []))

        total = row["positive"] + row["negative"] + row["neutral"]
        row["total"] = total
        row["pos_pct"] = (row["positive"] / total * 100) if total > 0 else 0
        row["neg_pct"] = (row["negative"] / total * 100) if total > 0 else 0

        matrix.append(row)

    return pd.DataFrame(matrix)

languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja"]
df = build_sentiment_matrix("technology", languages)

print("Multilingual Sentiment Matrix for 'technology':\n")
print(df.to_string(index=False))

print("\n\nSentiment Breakdown by Language:")
for _, row in df.iterrows():
    lang = row["language"]
    pos = row["pos_pct"]
    neg = row["neg_pct"]
    total = row["total"]

    if total > 0:
        pos_bar = "+" * int(pos / 5)
        neg_bar = "-" * int(neg / 5)
        print(f"\n  {lang}: {total} articles")
        print(f"    Positive: {pos:5.1f}% {pos_bar}")
        print(f"    Negative: {neg:5.1f}% {neg_bar}")
```

### Global Story Coverage Tracker

```python
import requests
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def track_daily_coverage(keyword, languages, days=7):
    end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    coverage = defaultdict(lambda: defaultdict(int))

    for i in range(days):
        day_end = end_date - timedelta(days=i)
        day_start = day_end - timedelta(days=1)
        date_str = day_start.strftime("%Y-%m-%d")

        for lang in languages:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "title": keyword,
                "language.code": lang,
                "published_at.start": day_start.strftime("%Y-%m-%d"),
                "published_at.end": day_end.strftime("%Y-%m-%d"),
                "per_page": 1,
            })
            response.raise_for_status()
            data = response.json()
            coverage[date_str][lang] = len(data.get("results", []))

    return coverage

languages = ["en", "es", "fr", "de", "zh", "ja", "ru"]
coverage = track_daily_coverage("OpenAI", languages, days=7)

print(f"Global coverage tracker for 'OpenAI' (last 7 days):\n")
print(f"{'Date':<12} " + " ".join(f"{lang:>5}" for lang in languages) + "  Total")
print("-" * (12 + 7 * len(languages) + 8))

for date in sorted(coverage.keys()):
    counts = [coverage[date][lang] for lang in languages]
    total = sum(counts)
    count_str = " ".join(f"{c:>5}" for c in counts)
    print(f"{date:<12} {count_str}  {total:>5}")

print("\n\nDaily trend by language:")
for lang in languages:
    daily_counts = [coverage[date][lang] for date in sorted(coverage.keys())]
    trend_line = "".join(["█" if c > 10 else "▓" if c > 5 else "░" if c > 0 else " " for c in daily_counts])
    total = sum(daily_counts)
    print(f"  {lang}: {trend_line}  (total: {total})")
```

### Regional Source Diversity Analyzer

```python
import requests
from collections import defaultdict

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def analyze_source_diversity(topic, country_language_pairs, per_page=100):
    diversity = {}

    for country, language in country_language_pairs:
        sources = set()
        page = 1

        while len(sources) < per_page:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "topic.id": topic,
                "language": language,
                "source.country.code": country,
                "per_page": 100,
                "page": page,
            })
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                break

            for article in data["results"]:
                sources.add(article["source"]["domain"])

            if len(data["results"]) < 100:
                break

            page += 1

        diversity[f"{country}-{language}"] = {
            "country": country,
            "language": language,
            "unique_sources": len(sources),
            "sources": sorted(sources)
        }

    return diversity

country_language_pairs = [
    ("US", "en"),
    ("GB", "en"),
    ("ES", "es"),
    ("MX", "es"),
    ("FR", "fr"),
    ("DE", "de"),
    ("IT", "it"),
    ("BR", "pt"),
    ("RU", "ru"),
    ("CN", "zh"),
    ("JP", "ja"),
]

diversity = analyze_source_diversity("climate_change", country_language_pairs)

print("Regional Source Diversity for 'climate_change':\n")
print(f"{'Region':<10} {'Language':<8} {'Unique Sources':>15}")
print("-" * 40)

sorted_diversity = sorted(diversity.items(), key=lambda x: x[1]["unique_sources"], reverse=True)

for key, info in sorted_diversity:
    print(f"{info['country']:<10} {info['language']:<8} {info['unique_sources']:>15}")

print("\n\nTop 5 sources per region:")
for key, info in sorted_diversity[:5]:
    print(f"\n  {info['country']}-{info['language']}:")
    for source in info["sources"][:5]:
        print(f"    - {source}")
```

### Cross-Language Entity Perception

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def analyze_entity_perception(entity_name, languages, days=30):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    perceptions = {}

    for lang in languages:
        sentiment_data = {}

        for polarity in ["positive", "negative", "neutral"]:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity_name,
                "language.code": lang,
                "sentiment.overall.polarity": polarity,
                "published_at.start": start_date.strftime("%Y-%m-%d"),
                "published_at.end": end_date.strftime("%Y-%m-%d"),
                "per_page": 1,
            })
            response.raise_for_status()
            data = response.json()
            sentiment_data[polarity] = len(data.get("results", []))

        total = sum(sentiment_data.values())

        if total > 0:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": entity_name,
                "language.code": lang,
                "sort.by": "sentiment.overall.score",
                "sort.order": "desc",
                "published_at.start": start_date.strftime("%Y-%m-%d"),
                "published_at.end": end_date.strftime("%Y-%m-%d"),
                "per_page": 50,
            })
            response.raise_for_status()
            articles = response.json().get("results", [])

            avg_score = sum(a["sentiment"]["overall"]["score"] for a in articles) / len(articles) if articles else 0

            perceptions[lang] = {
                "total": total,
                "positive": sentiment_data["positive"],
                "negative": sentiment_data["negative"],
                "neutral": sentiment_data["neutral"],
                "pos_pct": sentiment_data["positive"] / total * 100,
                "neg_pct": sentiment_data["negative"] / total * 100,
                "avg_score": avg_score,
            }

    return perceptions

languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko"]
perceptions = analyze_entity_perception("Tesla", languages)

print(f"Cross-Language Entity Perception: 'Tesla' (last 30 days)\n")
print(f"{'Lang':<6} {'Total':>6} {'Pos%':>6} {'Neg%':>6} {'AvgScore':>9} {'Sentiment Visualization'}")
print("-" * 80)

sorted_perceptions = sorted(perceptions.items(), key=lambda x: x[1]["avg_score"], reverse=True)

for lang, data in sorted_perceptions:
    pos_bar = "+" * int(data["pos_pct"] / 5)
    neg_bar = "-" * int(data["neg_pct"] / 5)

    print(f"{lang:<6} {data['total']:>6} {data['pos_pct']:>5.1f}% {data['neg_pct']:>5.1f}% {data['avg_score']:>+8.3f}  {pos_bar}{neg_bar}")

print("\n\nPerception Summary:")
most_positive = max(sorted_perceptions, key=lambda x: x[1]["avg_score"])
most_negative = min(sorted_perceptions, key=lambda x: x[1]["avg_score"])
most_coverage = max(sorted_perceptions, key=lambda x: x[1]["total"])

print(f"  Most positive: {most_positive[0]} (avg score: {most_positive[1]['avg_score']:+.3f})")
print(f"  Most negative: {most_negative[0]} (avg score: {most_negative[1]['avg_score']:+.3f})")
print(f"  Most coverage: {most_coverage[0]} ({most_coverage[1]['total']} articles)")
```

---

## JavaScript

### Cross-Language Volume Comparison

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko", "hi"];

async function getLanguageVolume(topic, languages, days = 7) {
  const endDate = new Date();
  const startDate = new Date(endDate.getTime() - days * 86400000);

  const requests = languages.map(async (lang) => {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "topic.id": topic,
      "language.code": lang,
      "published_at.start": startDate.toISOString().split("T")[0],
      "published_at.end": endDate.toISOString().split("T")[0],
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    return { lang, count: data.results?.length || 0 };
  });

  const results = await Promise.all(requests);
  return Object.fromEntries(results.map((r) => [r.lang, r.count]));
}

const volumes = await getLanguageVolume("cryptocurrency", LANGUAGES);

console.log("Article volume by language for 'cryptocurrency' (last 7 days):\n");

const sorted = Object.entries(volumes).sort((a, b) => b[1] - a[1]);

for (const [lang, count] of sorted) {
  const bar = "█".repeat(Math.floor(count / 100));
  console.log(`  ${lang.padStart(3)}: ${String(count).padStart(6)} ${bar}`);
}
```

### Multilingual Sentiment Matrix

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function buildSentimentMatrix(topic, languages, days = 7) {
  const endDate = new Date();
  const startDate = new Date(endDate.getTime() - days * 86400000);

  const matrix = [];

  for (const lang of languages) {
    const row = { language: lang };

    for (const polarity of ["positive", "negative", "neutral"]) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topic,
        "language.code": lang,
        "sentiment.overall.polarity": polarity,
        "published_at.start": startDate.toISOString().split("T")[0],
        "published_at.end": endDate.toISOString().split("T")[0],
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      row[polarity] = data.results?.length || 0;
    }

    const total = row.positive + row.negative + row.neutral;
    row.total = total;
    row.pos_pct = total > 0 ? (row.positive / total) * 100 : 0;
    row.neg_pct = total > 0 ? (row.negative / total) * 100 : 0;

    matrix.push(row);
  }

  return matrix;
}

const languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja"];
const matrix = await buildSentimentMatrix("technology", languages);

console.log("Multilingual Sentiment Matrix for 'technology':\n");
console.log(`${"Lang".padEnd(6)} ${"Pos".padStart(6)} ${"Neg".padStart(6)} ${"Neu".padStart(6)} ${"Total".padStart(6)} ${"Pos%".padStart(6)} ${"Neg%".padStart(6)}`);
console.log("-".repeat(50));

for (const row of matrix) {
  console.log(
    `${row.language.padEnd(6)} ${String(row.positive).padStart(6)} ` +
    `${String(row.negative).padStart(6)} ${String(row.neutral).padStart(6)} ` +
    `${String(row.total).padStart(6)} ${row.pos_pct.toFixed(1).padStart(6)} ${row.neg_pct.toFixed(1).padStart(6)}`
  );
}

console.log("\n\nSentiment Breakdown by Language:");
for (const row of matrix) {
  if (row.total > 0) {
    const posBar = "+".repeat(Math.floor(row.pos_pct / 5));
    const negBar = "-".repeat(Math.floor(row.neg_pct / 5));
    console.log(`\n  ${row.language}: ${row.total} articles`);
    console.log(`    Positive: ${row.pos_pct.toFixed(1).padStart(5)}% ${posBar}`);
    console.log(`    Negative: ${row.neg_pct.toFixed(1).padStart(5)}% ${negBar}`);
  }
}
```

### Global Story Coverage Tracker

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function trackDailyCoverage(keyword, languages, days = 7) {
  const endDate = new Date();
  endDate.setUTCHours(0, 0, 0, 0);

  const coverage = {};

  for (let i = 0; i < days; i++) {
    const dayEnd = new Date(endDate.getTime() - i * 86400000);
    const dayStart = new Date(dayEnd.getTime() - 86400000);
    const dateStr = dayStart.toISOString().split("T")[0];

    coverage[dateStr] = {};

    const requests = languages.map(async (lang) => {
      const params = new URLSearchParams({
        api_key: API_KEY,
        title: keyword,
        "language.code": lang,
        "published_at.start": dayStart.toISOString().split("T")[0],
        "published_at.end": dayEnd.toISOString().split("T")[0],
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      return { lang, count: data.results?.length || 0 };
    });

    const results = await Promise.all(requests);
    for (const { lang, count } of results) {
      coverage[dateStr][lang] = count;
    }
  }

  return coverage;
}

const languages = ["en", "es", "fr", "de", "zh", "ja", "ru"];
const coverage = await trackDailyCoverage("OpenAI", languages, 7);

console.log("Global coverage tracker for 'OpenAI' (last 7 days):\n");
console.log(
  `${"Date".padEnd(12)} ${languages.map((l) => l.padStart(5)).join(" ")}  Total`
);
console.log("-".repeat(12 + 7 * languages.length + 8));

for (const date of Object.keys(coverage).sort()) {
  const counts = languages.map((lang) => coverage[date][lang]);
  const total = counts.reduce((a, b) => a + b, 0);
  const countStr = counts.map((c) => String(c).padStart(5)).join(" ");
  console.log(`${date.padEnd(12)} ${countStr}  ${String(total).padStart(5)}`);
}

console.log("\n\nDaily trend by language:");
for (const lang of languages) {
  const dailyCounts = Object.keys(coverage)
    .sort()
    .map((date) => coverage[date][lang]);
  const trendLine = dailyCounts
    .map((c) => (c > 10 ? "█" : c > 5 ? "▓" : c > 0 ? "░" : " "))
    .join("");
  const total = dailyCounts.reduce((a, b) => a + b, 0);
  console.log(`  ${lang}: ${trendLine}  (total: ${total})`);
}
```

---

## PHP

### Cross-Language Volume Comparison

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko", "hi"];

function getLanguageVolume(string $topic, array $languages, int $days = 7): array
{
    global $apiKey, $baseUrl;

    $endDate   = new DateTimeImmutable("now", new DateTimeZone("UTC"));
    $startDate = $endDate->modify("-{$days} days");

    $volumes = [];

    foreach ($languages as $lang) {
        $query = http_build_query([
            "api_key"            => $apiKey,
            "topic.id"           => $topic,
            "language.code"      => $lang,
            "published_at.start" => $startDate->format("Y-m-d"),
            "published_at.end"   => $endDate->format("Y-m-d"),
            "per_page"           => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $volumes[$lang] = count($data["results"] ?? []);
    }

    return $volumes;
}

$volumes = getLanguageVolume("cryptocurrency", $languages);

echo "Article volume by language for 'cryptocurrency' (last 7 days):\n\n";

arsort($volumes);

foreach ($volumes as $lang => $count) {
    $bar = str_repeat("█", (int) ($count / 100));
    printf("  %3s: %6d %s\n", $lang, $count, $bar);
}
```

### Cross-Language Entity Perception

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function analyzeEntityPerception(string $entityName, array $languages, int $days = 30): array
{
    global $apiKey, $baseUrl;

    $endDate   = new DateTimeImmutable("now", new DateTimeZone("UTC"));
    $startDate = $endDate->modify("-{$days} days");

    $perceptions = [];

    foreach ($languages as $lang) {
        $sentimentData = [];

        foreach (["positive", "negative", "neutral"] as $polarity) {
            $query = http_build_query([
                "api_key"                    => $apiKey,
                "organization.name"          => $entityName,
                "language.code"              => $lang,
                "sentiment.overall.polarity" => $polarity,
                "published_at.start"         => $startDate->format("Y-m-d"),
                "published_at.end"           => $endDate->format("Y-m-d"),
                "per_page"                   => 1,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            $sentimentData[$polarity] = count($data["results"] ?? []);
        }

        $total = array_sum($sentimentData);

        if ($total > 0) {
            $query = http_build_query([
                "api_key"            => $apiKey,
                "organization.name"  => $entityName,
                "language.code"      => $lang,
                "sort.by"            => "sentiment.overall.score",
                "sort.order"         => "desc",
                "published_at.start" => $startDate->format("Y-m-d"),
                "published_at.end"   => $endDate->format("Y-m-d"),
                "per_page"           => 50,
            ]);

            $data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            $articles = $data["results"] ?? [];

            $avgScore = 0;
            if (!empty($articles)) {
                $scores   = array_column(array_column(array_column($articles, "sentiment"), "overall"), "score");
                $avgScore = array_sum($scores) / count($scores);
            }

            $perceptions[$lang] = [
                "total"     => $total,
                "positive"  => $sentimentData["positive"],
                "negative"  => $sentimentData["negative"],
                "neutral"   => $sentimentData["neutral"],
                "pos_pct"   => $sentimentData["positive"] / $total * 100,
                "neg_pct"   => $sentimentData["negative"] / $total * 100,
                "avg_score" => $avgScore,
            ];
        }
    }

    return $perceptions;
}

$languages   = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko"];
$perceptions = analyzeEntityPerception("Tesla", $languages);

echo "Cross-Language Entity Perception: 'Tesla' (last 30 days)\n\n";
printf("%-6s %6s %6s %6s %9s  %s\n", "Lang", "Total", "Pos%", "Neg%", "AvgScore", "Sentiment Visualization");
echo str_repeat("-", 80) . "\n";

uasort($perceptions, fn($a, $b) => $b["avg_score"] <=> $a["avg_score"]);

foreach ($perceptions as $lang => $data) {
    $posBar = str_repeat("+", (int) ($data["pos_pct"] / 5));
    $negBar = str_repeat("-", (int) ($data["neg_pct"] / 5));

    printf(
        "%-6s %6d %5.1f%% %5.1f%% %+8.3f  %s%s\n",
        $lang,
        $data["total"],
        $data["pos_pct"],
        $data["neg_pct"],
        $data["avg_score"],
        $posBar,
        $negBar
    );
}

echo "\n\nPerception Summary:\n";

$mostPositive = array_reduce(
    array_keys($perceptions),
    fn($carry, $lang) => !$carry || $perceptions[$lang]["avg_score"] > $perceptions[$carry]["avg_score"] ? $lang : $carry
);

$mostNegative = array_reduce(
    array_keys($perceptions),
    fn($carry, $lang) => !$carry || $perceptions[$lang]["avg_score"] < $perceptions[$carry]["avg_score"] ? $lang : $carry
);

$mostCoverage = array_reduce(
    array_keys($perceptions),
    fn($carry, $lang) => !$carry || $perceptions[$lang]["total"] > $perceptions[$carry]["total"] ? $lang : $carry
);

printf("  Most positive: %s (avg score: %+.3f)\n", $mostPositive, $perceptions[$mostPositive]["avg_score"]);
printf("  Most negative: %s (avg score: %+.3f)\n", $mostNegative, $perceptions[$mostNegative]["avg_score"]);
printf("  Most coverage: %s (%d articles)\n", $mostCoverage, $perceptions[$mostCoverage]["total"]);
```

### Regional Source Diversity Analyzer

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function analyzeSourceDiversity(string $topic, array $countryLanguagePairs, int $perPage = 100): array
{
    global $apiKey, $baseUrl;

    $diversity = [];

    foreach ($countryLanguagePairs as [$country, $language]) {
        $sources = [];
        $page    = 1;

        while (count($sources) < $perPage) {
            $query = http_build_query([
                "api_key"             => $apiKey,
                "topic.id"            => $topic,
                "language"            => $language,
                "source.country.code" => $country,
                "per_page"            => 100,
                "page"                => $page,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

            if (empty($data["results"])) {
                break;
            }

            foreach ($data["results"] as $article) {
                $sources[$article["source"]["domain"]] = true;
            }

            if (count($data["results"]) < 100) {
                break;
            }

            $page++;
        }

        $diversity["{$country}-{$language}"] = [
            "country"        => $country,
            "language"       => $language,
            "unique_sources" => count($sources),
            "sources"        => array_keys($sources),
        ];
    }

    return $diversity;
}

$countryLanguagePairs = [
    ["US", "en"],
    ["GB", "en"],
    ["ES", "es"],
    ["MX", "es"],
    ["FR", "fr"],
    ["DE", "de"],
    ["IT", "it"],
    ["BR", "pt"],
    ["RU", "ru"],
    ["CN", "zh"],
    ["JP", "ja"],
];

$diversity = analyzeSourceDiversity("climate_change", $countryLanguagePairs);

echo "Regional Source Diversity for 'climate_change':\n\n";
printf("%-10s %-8s %15s\n", "Region", "Language", "Unique Sources");
echo str_repeat("-", 40) . "\n";

uasort($diversity, fn($a, $b) => $b["unique_sources"] <=> $a["unique_sources"]);

foreach ($diversity as $info) {
    printf("%-10s %-8s %15d\n", $info["country"], $info["language"], $info["unique_sources"]);
}

echo "\n\nTop 5 sources per region:\n";

$i = 0;
foreach ($diversity as $key => $info) {
    if ($i++ >= 5) break;

    echo "\n  {$info['country']}-{$info['language']}:\n";
    foreach (array_slice($info["sources"], 0, 5) as $source) {
        echo "    - {$source}\n";
    }
}
```
