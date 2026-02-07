# Topic-Based Aggregation — Code Examples

Detailed examples for aggregating news by topic using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Topic Volume Comparison

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

TOPICS = [
    "artificial_intelligence",
    "cryptocurrency",
    "climate_change",
    "elections",
    "stock_market",
    "space",
    "cybersecurity",
    "electric_vehicles",
]

print("Topic volume comparison:\n")

volumes = {}
for topic in TOPICS:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "language": "en",
        "per_page": 1,
    })
    response.raise_for_status()
    volumes[topic] = response.json().get("total_results", 0)

max_vol = max(volumes.values()) or 1

for topic, count in sorted(volumes.items(), key=lambda x: -x[1]):
    bar = "█" * int(count / max_vol * 40)
    print(f"  {topic:<25} {count:>8} {bar}")
```

### Multi-Topic Digest Builder

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def build_digest(topics, articles_per_topic=5, days=1):
    today = datetime.utcnow()
    yesterday = today - timedelta(days=days)

    digest = {}
    for topic in topics:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "topic.id": topic,
            "published_at.start": yesterday.strftime("%Y-%m-%d"),
            "published_at.end": today.strftime("%Y-%m-%d"),
            "sort.by": "published_at",
            "sort.order": "desc",
            "language": "en",
            "per_page": articles_per_topic,
        })
        response.raise_for_status()
        digest[topic] = response.json().get("results", [])

    return digest

topics = ["technology", "finance", "health", "science", "sports"]
digest = build_digest(topics)

print(f"Daily News Digest — {datetime.utcnow().strftime('%Y-%m-%d')}\n")
print("=" * 60)

for topic, articles in digest.items():
    print(f"\n## {topic.replace('_', ' ').title()} ({len(articles)} articles)\n")
    for article in articles:
        print(f"  • {article['title']}")
        print(f"    {article['source']['domain']} — {article['href']}")
    print()
```

### Topic Sentiment Heatmap Data

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topics = ["technology", "finance", "politics", "health", "climate_change"]
days = 7
today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

print("Topic Sentiment Heatmap (positive ratio by day):\n")
header = f"{'Topic':<18}" + "".join(
    f" {(today - timedelta(days=d)).strftime('%m/%d'):>6}" for d in range(days - 1, -1, -1)
)
print(header)
print("-" * len(header))

for topic in topics:
    row = f"{topic:<18}"
    for d in range(days - 1, -1, -1):
        day_start = today - timedelta(days=d + 1)
        day_end = today - timedelta(days=d)

        pos_resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "topic.id": topic,
            "sentiment.overall.polarity": "positive",
            "published_at.start": day_start.strftime("%Y-%m-%d"),
            "published_at.end": day_end.strftime("%Y-%m-%d"),
            "language": "en",
            "per_page": 1,
        })
        pos_resp.raise_for_status()
        pos = pos_resp.json().get("total_results", 0)

        total_resp = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "topic.id": topic,
            "published_at.start": day_start.strftime("%Y-%m-%d"),
            "published_at.end": day_end.strftime("%Y-%m-%d"),
            "language": "en",
            "per_page": 1,
        })
        total_resp.raise_for_status()
        total = total_resp.json().get("total_results", 0) or 1

        ratio = pos / total
        row += f" {ratio:>5.0%} "

    print(row)
```

### Topic Trend Detection

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topics = [
    "artificial_intelligence", "cryptocurrency", "climate_change",
    "cybersecurity", "electric_vehicles", "space",
]

today = datetime.utcnow()
this_week_start = today - timedelta(days=7)
last_week_start = today - timedelta(days=14)

print("Topic trend detection (week-over-week change):\n")
print(f"{'Topic':<25} {'Last Week':>10} {'This Week':>10} {'Change':>10}")
print("-" * 58)

for topic in topics:
    # This week
    resp1 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "published_at.start": this_week_start.strftime("%Y-%m-%d"),
        "published_at.end": today.strftime("%Y-%m-%d"),
        "language": "en",
        "per_page": 1,
    })
    resp1.raise_for_status()
    this_week = resp1.json().get("total_results", 0)

    # Last week
    resp2 = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "published_at.start": last_week_start.strftime("%Y-%m-%d"),
        "published_at.end": this_week_start.strftime("%Y-%m-%d"),
        "language": "en",
        "per_page": 1,
    })
    resp2.raise_for_status()
    last_week = resp2.json().get("total_results", 0)

    if last_week > 0:
        change = (this_week - last_week) / last_week * 100
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {topic:<23} {last_week:>10} {this_week:>10} {arrow} {change:>+6.1f}%")
    else:
        print(f"  {topic:<23} {last_week:>10} {this_week:>10}       N/A")
```

### Cross-Topic Title Keyword Analysis

```python
import requests
from collections import Counter

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topic = "artificial_intelligence"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "topic.id": topic,
    "language": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    "per_page": 50,
})
response.raise_for_status()

articles = response.json().get("results", [])
stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
              "to", "for", "of", "and", "or", "with", "by", "from", "as", "it",
              "its", "that", "this", "has", "have", "how", "new", "will", "can"}

words = Counter()
for article in articles:
    title_words = article["title"].lower().split()
    words.update(w.strip(".,!?:;\"'()[]") for w in title_words
                 if len(w) > 2 and w.lower() not in stop_words)

print(f"Top keywords in '{topic}' titles ({len(articles)} articles):\n")
for word, count in words.most_common(20):
    bar = "█" * count
    print(f"  {word:<20} {count:>3} {bar}")
```

---

## JavaScript

### Topic Volume Comparison

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const TOPICS = [
  "artificial_intelligence",
  "cryptocurrency",
  "climate_change",
  "elections",
  "stock_market",
  "space",
  "cybersecurity",
  "electric_vehicles",
];

console.log("Topic volume comparison:\n");

const volumes = {};

for (const topic of TOPICS) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": topic,
    language: "en",
    per_page: "1",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  volumes[topic] = data.total_results || 0;
}

const maxVol = Math.max(...Object.values(volumes)) || 1;

Object.entries(volumes)
  .sort((a, b) => b[1] - a[1])
  .forEach(([topic, count]) => {
    const bar = "#".repeat(Math.round((count / maxVol) * 40));
    console.log(`  ${topic.padEnd(25)} ${String(count).padStart(8)} ${bar}`);
  });
```

### Multi-Topic Digest Builder

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function buildDigest(topics, articlesPerTopic = 5, days = 1) {
  const now = new Date();
  const start = new Date(now.getTime() - days * 86400000);

  const digest = {};

  for (const topic of topics) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "topic.id": topic,
      "published_at.start": start.toISOString().split("T")[0],
      "published_at.end": now.toISOString().split("T")[0],
      "sort.by": "published_at",
      "sort.order": "desc",
      language: "en",
      per_page: String(articlesPerTopic),
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    digest[topic] = data.results || [];
  }

  return digest;
}

const topics = ["technology", "finance", "health", "science", "sports"];
const digest = await buildDigest(topics);

console.log(`Daily News Digest — ${new Date().toISOString().split("T")[0]}\n`);
console.log("=".repeat(60));

for (const [topic, articles] of Object.entries(digest)) {
  const name = topic.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  console.log(`\n## ${name} (${articles.length} articles)\n`);
  articles.forEach((a) => {
    console.log(`  • ${a.title}`);
    console.log(`    ${a.source.domain} — ${a.href}`);
  });
  console.log();
}
```

### Topic Trend Detection

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const topics = [
  "artificial_intelligence", "cryptocurrency", "climate_change",
  "cybersecurity", "electric_vehicles", "space",
];

const now = new Date();
const thisWeekStart = new Date(now.getTime() - 7 * 86400000);
const lastWeekStart = new Date(now.getTime() - 14 * 86400000);

const fmt = (d) => d.toISOString().split("T")[0];

console.log("Topic trend detection (week-over-week change):\n");
console.log(
  `${"Topic".padEnd(25)} ${"Last Week".padStart(10)} ` +
  `${"This Week".padStart(10)} ${"Change".padStart(10)}`
);
console.log("-".repeat(58));

for (const topic of topics) {
  const thisParams = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": topic,
    "published_at.start": fmt(thisWeekStart),
    "published_at.end": fmt(now),
    language: "en",
    per_page: "1",
  });

  const lastParams = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": topic,
    "published_at.start": fmt(lastWeekStart),
    "published_at.end": fmt(thisWeekStart),
    language: "en",
    per_page: "1",
  });

  const [thisResp, lastResp] = await Promise.all([
    fetch(`${BASE_URL}?${thisParams}`).then((r) => r.json()),
    fetch(`${BASE_URL}?${lastParams}`).then((r) => r.json()),
  ]);

  const thisWeek = thisResp.total_results || 0;
  const lastWeek = lastResp.total_results || 0;

  if (lastWeek > 0) {
    const change = ((thisWeek - lastWeek) / lastWeek) * 100;
    const arrow = change > 0 ? "↑" : change < 0 ? "↓" : "→";
    const sign = change >= 0 ? "+" : "";
    console.log(
      `  ${topic.padEnd(23)} ${String(lastWeek).padStart(10)} ` +
      `${String(thisWeek).padStart(10)} ${arrow} ${(sign + change.toFixed(1) + "%").padStart(7)}`
    );
  } else {
    console.log(
      `  ${topic.padEnd(23)} ${String(lastWeek).padStart(10)} ` +
      `${String(thisWeek).padStart(10)}       N/A`
    );
  }
}
```

---

## PHP

### Topic Volume Comparison

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$topics = [
    "artificial_intelligence",
    "cryptocurrency",
    "climate_change",
    "elections",
    "stock_market",
    "space",
    "cybersecurity",
    "electric_vehicles",
];

echo "Topic volume comparison:\n\n";

$volumes = [];
foreach ($topics as $topic) {
    $query = http_build_query([
        "api_key"  => $apiKey,
        "topic.id" => $topic,
        "language" => "en",
        "per_page" => 1,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $volumes[$topic] = $data["total_results"] ?? 0;
}

arsort($volumes);
$maxVol = max($volumes) ?: 1;

foreach ($volumes as $topic => $count) {
    $bar = str_repeat("#", (int) round($count / $maxVol * 40));
    printf("  %-25s %8d %s\n", $topic, $count, $bar);
}
```

### Multi-Topic Digest Builder

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function buildDigest(array $topics, int $articlesPerTopic = 5, int $days = 1): array
{
    global $apiKey, $baseUrl;

    $now   = new DateTimeImmutable("now", new DateTimeZone("UTC"));
    $start = $now->modify("-{$days} days");

    $digest = [];
    foreach ($topics as $topic) {
        $query = http_build_query([
            "api_key"            => $apiKey,
            "topic.id"           => $topic,
            "published_at.start" => $start->format("Y-m-d"),
            "published_at.end"   => $now->format("Y-m-d"),
            "sort.by"            => "published_at",
            "sort.order"         => "desc",
            "language"           => "en",
            "per_page"           => $articlesPerTopic,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $digest[$topic] = $data["results"] ?? [];
    }

    return $digest;
}

$topics = ["technology", "finance", "health", "science", "sports"];
$digest = buildDigest($topics);

echo "Daily News Digest — " . date("Y-m-d") . "\n\n";
echo str_repeat("=", 60) . "\n";

foreach ($digest as $topic => $articles) {
    $name = ucwords(str_replace("_", " ", $topic));
    echo "\n## {$name} (" . count($articles) . " articles)\n\n";
    foreach ($articles as $article) {
        echo "  • {$article['title']}\n";
        echo "    {$article['source']['domain']} — {$article['href']}\n";
    }
    echo "\n";
}
```

### Topic Trend Detection

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$topics = [
    "artificial_intelligence", "cryptocurrency", "climate_change",
    "cybersecurity", "electric_vehicles", "space",
];

$now           = new DateTimeImmutable("now", new DateTimeZone("UTC"));
$thisWeekStart = $now->modify("-7 days");
$lastWeekStart = $now->modify("-14 days");

echo "Topic trend detection (week-over-week change):\n\n";
printf("  %-23s %10s %10s %10s\n", "Topic", "Last Week", "This Week", "Change");
echo str_repeat("-", 58) . "\n";

foreach ($topics as $topic) {
    // This week
    $q1 = http_build_query([
        "api_key"            => $apiKey,
        "topic.id"           => $topic,
        "published_at.start" => $thisWeekStart->format("Y-m-d"),
        "published_at.end"   => $now->format("Y-m-d"),
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $d1       = json_decode(file_get_contents("{$baseUrl}?{$q1}"), true);
    $thisWeek = $d1["total_results"] ?? 0;

    // Last week
    $q2 = http_build_query([
        "api_key"            => $apiKey,
        "topic.id"           => $topic,
        "published_at.start" => $lastWeekStart->format("Y-m-d"),
        "published_at.end"   => $thisWeekStart->format("Y-m-d"),
        "language"           => "en",
        "per_page"           => 1,
    ]);
    $d2       = json_decode(file_get_contents("{$baseUrl}?{$q2}"), true);
    $lastWeek = $d2["total_results"] ?? 0;

    if ($lastWeek > 0) {
        $change = ($thisWeek - $lastWeek) / $lastWeek * 100;
        $arrow  = $change > 0 ? "↑" : ($change < 0 ? "↓" : "→");
        printf("  %-23s %10d %10d %s %+6.1f%%\n",
            $topic, $lastWeek, $thisWeek, $arrow, $change);
    } else {
        printf("  %-23s %10d %10d       N/A\n",
            $topic, $lastWeek, $thisWeek);
    }
}
```
