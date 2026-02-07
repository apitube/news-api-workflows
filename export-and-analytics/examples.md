# Export and Analytics — Code Examples

Detailed examples for exporting news data and building analytics pipelines using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Paginated CSV Export

```python
import requests
import csv

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def export_to_csv(filename, params, max_articles=500):
    all_articles = []
    page = 1

    while len(all_articles) < max_articles:
        response = requests.get(BASE_URL, params={
            **params,
            "api_key": API_KEY,
            "per_page": 50,
            "page": page,
        })
        response.raise_for_status()
        articles = response.json().get("results", [])

        if not articles:
            break

        all_articles.extend(articles)
        print(f"  Page {page}: {len(articles)} articles")
        page += 1

    all_articles = all_articles[:max_articles]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "title", "description", "url", "source_name", "source_domain",
            "author", "language", "published_at",
            "sentiment_polarity", "sentiment_score",
        ])
        for a in all_articles:
            writer.writerow([
                a.get("title", ""),
                a.get("description", ""),
                a.get("href", ""),
                a.get("source", {}).get("domain", ""),
                a.get("source", {}).get("home_page_url", ""),
                a.get("author", ""),
                a.get("language", ""),
                a.get("published_at", ""),
                a.get("sentiment", {}).get("overall", {}).get("polarity", ""),
                a.get("sentiment", {}).get("overall", {}).get("score", ""),
            ])

    print(f"\nExported {len(all_articles)} articles to {filename}")
    return len(all_articles)

export_to_csv("tech_news.csv", {
    "topic.id": "technology",
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
})
```

### JSONL Streaming Export

```python
import requests
import json

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def export_to_jsonl(filename, params, max_articles=1000):
    page = 1
    count = 0

    with open(filename, "w", encoding="utf-8") as f:
        while count < max_articles:
            response = requests.get(BASE_URL, params={
                **params,
                "api_key": API_KEY,
                "per_page": 50,
                "page": page,
            })
            response.raise_for_status()
            articles = response.json().get("results", [])

            if not articles:
                break

            for article in articles:
                if count >= max_articles:
                    break
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                count += 1

            print(f"  Page {page}: wrote {len(articles)} articles (total: {count})")
            page += 1

    print(f"\nExported {count} articles to {filename}")
    return count

export_to_jsonl("ai_news.jsonl", {
    "topic.id": "artificial_intelligence",
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
})
```

### Multi-Topic Dataset Builder

```python
import requests
import csv
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topics = ["technology", "finance", "health", "politics", "sports"]
days = 7
today = datetime.utcnow()
start = today - timedelta(days=days)

filename = f"multi_topic_{today.strftime('%Y%m%d')}.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "topic", "title", "url", "source_name", "published_at",
        "sentiment_polarity", "sentiment_score", "language",
    ])

    total = 0
    for topic in topics:
        page = 1
        topic_count = 0

        while topic_count < 100:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "topic.id": topic,
                "published_at.start": start.strftime("%Y-%m-%d"),
                "published_at.end": today.strftime("%Y-%m-%d"),
                "language": "en",
                "sort.by": "published_at",
                "sort.order": "desc",
                "per_page": 50,
                "page": page,
            })
            response.raise_for_status()
            articles = response.json().get("results", [])

            if not articles:
                break

            for a in articles:
                if topic_count >= 100:
                    break
                writer.writerow([
                    topic,
                    a.get("title", ""),
                    a.get("href", ""),
                    a.get("source", {}).get("domain", ""),
                    a.get("published_at", ""),
                    a.get("sentiment", {}).get("overall", {}).get("polarity", ""),
                    a.get("sentiment", {}).get("overall", {}).get("score", ""),
                    a.get("language", ""),
                ])
                topic_count += 1

            page += 1

        print(f"  {topic}: {topic_count} articles")
        total += topic_count

print(f"\nExported {total} articles across {len(topics)} topics to {filename}")
```

### Analytics Summary Report

```python
import requests
from datetime import datetime, timedelta
from collections import Counter

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topic = "technology"
days = 7
today = datetime.utcnow()
start = today - timedelta(days=days)

# Collect articles
all_articles = []
page = 1

while len(all_articles) < 200:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "published_at.start": start.strftime("%Y-%m-%d"),
        "published_at.end": today.strftime("%Y-%m-%d"),
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 50,
        "page": page,
    })
    response.raise_for_status()
    articles = response.json().get("results", [])

    if not articles:
        break

    all_articles.extend(articles)
    page += 1

# Generate report
print(f"Analytics Report: {topic}")
print(f"Period: {start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
print(f"Total articles analyzed: {len(all_articles)}")
print("=" * 50)

# Sentiment distribution
sentiments = Counter(a["sentiment"]["overall"]["polarity"] for a in all_articles)
print("\nSentiment Distribution:")
for polarity, count in sentiments.most_common():
    pct = count / len(all_articles) * 100
    bar = "█" * int(pct / 2)
    print(f"  {polarity:<10} {count:>5} ({pct:5.1f}%) {bar}")

# Top sources
sources = Counter(a["source"]["domain"] for a in all_articles)
print("\nTop 10 Sources:")
for source, count in sources.most_common(10):
    print(f"  {source:<30} {count:>5}")

# Articles per day
daily = Counter(a["published_at"][:10] for a in all_articles)
print("\nArticles per Day:")
for date, count in sorted(daily.items()):
    bar = "█" * (count // 2)
    print(f"  {date} {count:>5} {bar}")

# Average sentiment score
scores = [a["sentiment"]["overall"]["score"] for a in all_articles
          if "sentiment" in a and "overall" in a["sentiment"]]
if scores:
    avg = sum(scores) / len(scores)
    print(f"\nAverage Sentiment Score: {avg:+.3f}")
```

### Scheduled Incremental Export

```python
import requests
import json
import os
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

STATE_FILE = "export_state.json"
OUTPUT_DIR = "exports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load last export timestamp
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
    last_export = state.get("last_export", "")
else:
    last_export = (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

now = datetime.utcnow()
now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

print(f"Incremental export: {last_export} -> {now_str}\n")

all_articles = []
page = 1

while True:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "published_at.start": last_export,
        "published_at.end": now_str,
        "language": "en",
        "sort.by": "published_at",
        "sort.order": "asc",
        "per_page": 50,
        "page": page,
    })
    response.raise_for_status()
    articles = response.json().get("results", [])

    if not articles:
        break

    all_articles.extend(articles)
    print(f"  Page {page}: {len(articles)} articles")
    page += 1

    if len(all_articles) >= 1000:
        break

# Save to file
output_file = os.path.join(OUTPUT_DIR, f"export_{now.strftime('%Y%m%d_%H%M%S')}.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for article in all_articles:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")

# Update state
with open(STATE_FILE, "w") as f:
    json.dump({"last_export": now_str}, f)

print(f"\nExported {len(all_articles)} new articles to {output_file}")
```

---

## JavaScript

### Paginated CSV Export

```javascript
import { writeFileSync } from "node:fs";

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function exportToCsv(filename, filterParams, maxArticles = 500) {
  const allArticles = [];
  let page = 1;

  while (allArticles.length < maxArticles) {
    const params = new URLSearchParams({
      ...filterParams,
      api_key: API_KEY,
      per_page: "50",
      page: String(page),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results || [];

    if (articles.length === 0) break;

    allArticles.push(...articles);
    console.log(`  Page ${page}: ${articles.length} articles`);
    page++;
  }

  const rows = allArticles.slice(0, maxArticles);

  const header = [
    "title", "description", "url", "source_name", "source_domain",
    "author", "language", "published_at",
    "sentiment_polarity", "sentiment_score",
  ].join(",");

  const csvRows = rows.map((a) => {
    const escape = (s) => `"${String(s || "").replace(/"/g, '""')}"`;
    return [
      escape(a.title),
      escape(a.description),
      escape(a.href),
      escape(a.source?.domain),
      escape(a.source?.home_page_url),
      escape(a.author),
      escape(a.language),
      escape(a.published_at),
      escape(a.sentiment?.overall?.polarity),
      a.sentiment?.overall?.score ?? "",
    ].join(",");
  });

  writeFileSync(filename, [header, ...csvRows].join("\n"));
  console.log(`\nExported ${rows.length} articles to ${filename}`);
}

await exportToCsv("tech_news.csv", {
  "topic.id": "technology",
  language: "en",
  "sort.by": "published_at",
  "sort.order": "desc",
});
```

### JSONL Streaming Export

```javascript
import { createWriteStream } from "node:fs";

const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function exportToJsonl(filename, filterParams, maxArticles = 1000) {
  const stream = createWriteStream(filename);
  let page = 1;
  let count = 0;

  while (count < maxArticles) {
    const params = new URLSearchParams({
      ...filterParams,
      api_key: API_KEY,
      per_page: "50",
      page: String(page),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results || [];

    if (articles.length === 0) break;

    for (const article of articles) {
      if (count >= maxArticles) break;
      stream.write(JSON.stringify(article) + "\n");
      count++;
    }

    console.log(`  Page ${page}: wrote ${articles.length} articles (total: ${count})`);
    page++;
  }

  stream.end();
  console.log(`\nExported ${count} articles to ${filename}`);
}

await exportToJsonl("ai_news.jsonl", {
  "topic.id": "artificial_intelligence",
  language: "en",
  "sort.by": "published_at",
  "sort.order": "desc",
});
```

### Analytics Summary Report

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function generateReport(topic, days = 7) {
  const now = new Date();
  const start = new Date(now.getTime() - days * 86400000);
  const fmt = (d) => d.toISOString().split("T")[0];

  // Collect articles
  const allArticles = [];
  let page = 1;

  while (allArticles.length < 200) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "topic.id": topic,
      "published_at.start": fmt(start),
      "published_at.end": fmt(now),
      language: "en",
      "sort.by": "published_at",
      "sort.order": "desc",
      per_page: "50",
      page: String(page),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const articles = data.results || [];

    if (articles.length === 0) break;
    allArticles.push(...articles);
    page++;
  }

  // Generate report
  console.log(`Analytics Report: ${topic}`);
  console.log(`Period: ${fmt(start)} to ${fmt(now)}`);
  console.log(`Total articles analyzed: ${allArticles.length}`);
  console.log("=".repeat(50));

  // Sentiment distribution
  const sentiments = {};
  allArticles.forEach((a) => {
    const p = a.sentiment?.overall?.polarity || "unknown";
    sentiments[p] = (sentiments[p] || 0) + 1;
  });

  console.log("\nSentiment Distribution:");
  Object.entries(sentiments)
    .sort((a, b) => b[1] - a[1])
    .forEach(([polarity, count]) => {
      const pct = ((count / allArticles.length) * 100).toFixed(1);
      const bar = "#".repeat(Math.round(pct / 2));
      console.log(`  ${polarity.padEnd(10)} ${String(count).padStart(5)} (${pct.padStart(5)}%) ${bar}`);
    });

  // Top sources
  const sources = {};
  allArticles.forEach((a) => {
    const name = a.source?.domain || "Unknown";
    sources[name] = (sources[name] || 0) + 1;
  });

  console.log("\nTop 10 Sources:");
  Object.entries(sources)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([source, count]) => {
      console.log(`  ${source.padEnd(30)} ${String(count).padStart(5)}`);
    });

  // Articles per day
  const daily = {};
  allArticles.forEach((a) => {
    const date = a.published_at?.slice(0, 10) || "unknown";
    daily[date] = (daily[date] || 0) + 1;
  });

  console.log("\nArticles per Day:");
  Object.entries(daily)
    .sort()
    .forEach(([date, count]) => {
      const bar = "#".repeat(Math.floor(count / 2));
      console.log(`  ${date} ${String(count).padStart(5)} ${bar}`);
    });

  // Average sentiment score
  const scores = allArticles
    .map((a) => a.sentiment?.overall?.score)
    .filter((s) => typeof s === "number");

  if (scores.length > 0) {
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const sign = avg >= 0 ? "+" : "";
    console.log(`\nAverage Sentiment Score: ${sign}${avg.toFixed(3)}`);
  }
}

await generateReport("technology");
```

---

## PHP

### Paginated CSV Export

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function exportToCsv(string $filename, array $filterParams, int $maxArticles = 500): int
{
    global $apiKey, $baseUrl;

    $allArticles = [];
    $page = 1;

    while (count($allArticles) < $maxArticles) {
        $query = http_build_query(array_merge($filterParams, [
            "api_key"  => $apiKey,
            "per_page" => 50,
            "page"     => $page,
        ]));

        $data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $articles = $data["results"] ?? [];

        if (empty($articles)) {
            break;
        }

        $allArticles = array_merge($allArticles, $articles);
        echo "  Page {$page}: " . count($articles) . " articles\n";
        $page++;
    }

    $allArticles = array_slice($allArticles, 0, $maxArticles);

    $fp = fopen($filename, "w");
    fputcsv($fp, [
        "title", "description", "url", "source_name", "source_domain",
        "author", "language", "published_at",
        "sentiment_polarity", "sentiment_score",
    ]);

    foreach ($allArticles as $a) {
        fputcsv($fp, [
            $a["title"] ?? "",
            $a["description"] ?? "",
            $a["href"] ?? "",
            $a["source"]["domain"] ?? "",
            $a["source"]["home_page_url"] ?? "",
            $a["author"] ?? "",
            $a["language"] ?? "",
            $a["published_at"] ?? "",
            $a["sentiment"]["overall"]["polarity"] ?? "",
            $a["sentiment"]["overall"]["score"] ?? "",
        ]);
    }

    fclose($fp);
    $count = count($allArticles);
    echo "\nExported {$count} articles to {$filename}\n";

    return $count;
}

exportToCsv("tech_news.csv", [
    "topic.id"   => "technology",
    "language"    => "en",
    "sort.by"     => "published_at",
    "sort.order"  => "desc",
]);
```

### JSONL Export

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function exportToJsonl(string $filename, array $filterParams, int $maxArticles = 1000): int
{
    global $apiKey, $baseUrl;

    $fp    = fopen($filename, "w");
    $page  = 1;
    $count = 0;

    while ($count < $maxArticles) {
        $query = http_build_query(array_merge($filterParams, [
            "api_key"  => $apiKey,
            "per_page" => 50,
            "page"     => $page,
        ]));

        $data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $articles = $data["results"] ?? [];

        if (empty($articles)) {
            break;
        }

        foreach ($articles as $article) {
            if ($count >= $maxArticles) {
                break;
            }
            fwrite($fp, json_encode($article, JSON_UNESCAPED_UNICODE) . "\n");
            $count++;
        }

        echo "  Page {$page}: wrote " . count($articles) . " articles (total: {$count})\n";
        $page++;
    }

    fclose($fp);
    echo "\nExported {$count} articles to {$filename}\n";

    return $count;
}

exportToJsonl("ai_news.jsonl", [
    "topic.id"   => "artificial_intelligence",
    "language"    => "en",
    "sort.by"     => "published_at",
    "sort.order"  => "desc",
]);
```

### Analytics Summary Report

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$topic = "technology";
$days  = 7;
$now   = new DateTimeImmutable("now", new DateTimeZone("UTC"));
$start = $now->modify("-{$days} days");

// Collect articles
$allArticles = [];
$page = 1;

while (count($allArticles) < 200) {
    $query = http_build_query([
        "api_key"            => $apiKey,
        "topic.id"           => $topic,
        "published_at.start" => $start->format("Y-m-d"),
        "published_at.end"   => $now->format("Y-m-d"),
        "language"           => "en",
        "sort.by"            => "published_at",
        "sort.order"         => "desc",
        "per_page"           => 50,
        "page"               => $page,
    ]);

    $data     = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $articles = $data["results"] ?? [];

    if (empty($articles)) {
        break;
    }

    $allArticles = array_merge($allArticles, $articles);
    $page++;
}

// Generate report
$count = count($allArticles);
echo "Analytics Report: {$topic}\n";
echo "Period: {$start->format('Y-m-d')} to {$now->format('Y-m-d')}\n";
echo "Total articles analyzed: {$count}\n";
echo str_repeat("=", 50) . "\n";

// Sentiment distribution
$sentiments = [];
foreach ($allArticles as $a) {
    $p = $a["sentiment"]["overall"]["polarity"] ?? "unknown";
    $sentiments[$p] = ($sentiments[$p] ?? 0) + 1;
}
arsort($sentiments);

echo "\nSentiment Distribution:\n";
foreach ($sentiments as $polarity => $c) {
    $pct = $c / $count * 100;
    $bar = str_repeat("#", (int) round($pct / 2));
    printf("  %-10s %5d (%5.1f%%) %s\n", $polarity, $c, $pct, $bar);
}

// Top sources
$sources = [];
foreach ($allArticles as $a) {
    $name = $a["source"]["domain"] ?? "Unknown";
    $sources[$name] = ($sources[$name] ?? 0) + 1;
}
arsort($sources);

echo "\nTop 10 Sources:\n";
$i = 0;
foreach ($sources as $source => $c) {
    if ($i++ >= 10) break;
    printf("  %-30s %5d\n", $source, $c);
}

// Articles per day
$daily = [];
foreach ($allArticles as $a) {
    $date = substr($a["published_at"] ?? "", 0, 10);
    $daily[$date] = ($daily[$date] ?? 0) + 1;
}
ksort($daily);

echo "\nArticles per Day:\n";
foreach ($daily as $date => $c) {
    $bar = str_repeat("#", intdiv($c, 2));
    printf("  %s %5d %s\n", $date, $c, $bar);
}

// Average sentiment score
$scores = array_filter(array_map(
    fn($a) => $a["sentiment"]["overall"]["score"] ?? null,
    $allArticles
), fn($s) => $s !== null);

if (!empty($scores)) {
    $avg = array_sum($scores) / count($scores);
    printf("\nAverage Sentiment Score: %+.3f\n", $avg);
}
```
