# Sentiment Analysis — Code Examples

Detailed examples for filtering and analyzing news by sentiment using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Filter by Sentiment Polarity

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def get_news_by_sentiment(polarity, topic=None, language="en", per_page=20):
    params = {
        "api_key": API_KEY,
        "sentiment.overall.polarity": polarity,
        "language.code": language,
        "per_page": per_page,
    }
    if topic:
        params["topic.id"] = topic

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

# Fetch positive tech news
data = get_news_by_sentiment("positive", topic="technology")
print(f"Found {len(data['results'])} positive tech articles\n")

for article in data["results"]:
    score = article["sentiment"]["overall"]["score"]
    print(f"  [{score:+.2f}] {article['title']}")
```

### Sort by Sentiment Score

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

response = requests.get(BASE_URL, params={
    "api_key": API_KEY,
    "topic.id": "cryptocurrency",
    "sort.by": "sentiment.overall.score",
    "sort.order": "desc",
    "language.code": "en",
    "per_page": 20,
})
response.raise_for_status()

data = response.json()
print("Crypto news sorted by sentiment (most positive first):\n")

for article in data["results"]:
    score = article["sentiment"]["overall"]["score"]
    polarity = article["sentiment"]["overall"]["polarity"]
    print(f"  [{polarity:>8} {score:+.2f}] {article['title']}")
    print(f"    Source: {article['source']['domain']}")
    print()
```

### Sentiment Comparison Across Topics

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

topics = ["technology", "politics", "finance", "health", "sports"]

print("Sentiment distribution by topic:\n")

for topic in topics:
    results = {}
    for polarity in ["positive", "negative", "neutral"]:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "topic.id": topic,
            "sentiment.overall.polarity": polarity,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        results[polarity] = len(response.json().get("results", []))

    total = sum(results.values()) or 1
    print(f"  {topic}:")
    for polarity, count in results.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {polarity:>8}: {count:>6} ({pct:5.1f}%) {bar}")
    print()
```

### Sentiment Trend Over Time

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def get_daily_sentiment(topic, days=14, language="en"):
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    daily_data = []

    for i in range(days):
        day_start = today - timedelta(days=i + 1)
        day_end = today - timedelta(days=i)

        counts = {}
        for polarity in ["positive", "negative", "neutral"]:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "topic.id": topic,
                "sentiment.overall.polarity": polarity,
                "published_at.start": day_start.strftime("%Y-%m-%d"),
                "published_at.end": day_end.strftime("%Y-%m-%d"),
                "language.code": language,
                "per_page": 1,
            })
            response.raise_for_status()
            counts[polarity] = len(response.json().get("results", []))

        daily_data.append({
            "date": day_start.strftime("%Y-%m-%d"),
            **counts,
        })

    return daily_data

trend = get_daily_sentiment("technology")

print("Daily sentiment trend for 'technology':\n")
print(f"{'Date':<12} {'Pos':>6} {'Neg':>6} {'Neu':>6} {'Ratio':>8}")
print("-" * 42)

for day in reversed(trend):
    total = day["positive"] + day["negative"] + day["neutral"] or 1
    ratio = day["positive"] / total
    print(f"{day['date']:<12} {day['positive']:>6} {day['negative']:>6} "
          f"{day['neutral']:>6} {ratio:>7.1%}")
```

### Negative Sentiment Alert System

```python
import requests
import time

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

WATCH_TOPICS = ["finance", "technology"]
ALERT_THRESHOLD = 100
POLL_INTERVAL = 300  # seconds

def check_negative_spike(topic, language="en"):
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": topic,
        "sentiment.overall.polarity": "negative",
        "published_at.start": datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z"),
        "language.code": language,
        "per_page": 1,
    })
    response.raise_for_status()
    data = response.json()
    return len(data.get("results", []))

from datetime import datetime

print("Monitoring negative sentiment spikes...\n")

while True:
    for topic in WATCH_TOPICS:
        count = check_negative_spike(topic)
        status = "ALERT" if count >= ALERT_THRESHOLD else "OK"
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        print(f"  [{timestamp}] {topic}: {count} negative articles today [{status}]")

        if count >= ALERT_THRESHOLD:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "topic.id": topic,
                "sentiment.overall.polarity": "negative",
                "sort.by": "sentiment.overall.score",
                "sort.order": "asc",
                "language.code": "en",
                "per_page": 5,
            })
            response.raise_for_status()
            for article in response.json()["results"]:
                print(f"    -> {article['title']}")

    print()
    time.sleep(POLL_INTERVAL)
```

---

## JavaScript

### Filter by Sentiment Polarity

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getNewsBySentiment(polarity, options = {}) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "sentiment.overall.polarity": polarity,
    "language.code": options.language || "en",
    per_page: String(options.perPage || 20),
  });

  if (options.topic) params.set("topic.id", options.topic);

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await getNewsBySentiment("positive", { topic: "technology" });
console.log(`Found ${data.results.length} positive tech articles\n`);

data.results.forEach((article) => {
  const score = article.sentiment.overall.score;
  console.log(`  [${score > 0 ? "+" : ""}${score.toFixed(2)}] ${article.title}`);
});
```

### Sort by Sentiment Score

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getCryptoNewsBySentiment() {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": "cryptocurrency",
    "sort.by": "sentiment.overall.score",
    "sort.order": "desc",
    "language.code": "en",
    per_page: "20",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();

  console.log("Crypto news sorted by sentiment (most positive first):\n");

  data.results.forEach((article) => {
    const { polarity, score } = article.sentiment.overall;
    const sign = score > 0 ? "+" : "";
    console.log(`  [${polarity.padStart(8)} ${sign}${score.toFixed(2)}] ${article.title}`);
    console.log(`    Source: ${article.source.domain}\n`);
  });
}

getCryptoNewsBySentiment();
```

### Sentiment Comparison Across Topics

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function compareSentiment(topics) {
  console.log("Sentiment distribution by topic:\n");

  for (const topic of topics) {
    const results = {};

    for (const polarity of ["positive", "negative", "neutral"]) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topic,
        "sentiment.overall.polarity": polarity,
        "language.code": "en",
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      results[polarity] = data.results?.length || 0;
    }

    const total = Object.values(results).reduce((a, b) => a + b, 0) || 1;

    console.log(`  ${topic}:`);
    for (const [polarity, count] of Object.entries(results)) {
      const pct = ((count / total) * 100).toFixed(1);
      const bar = "#".repeat(Math.round(count / total * 50));
      console.log(`    ${polarity.padStart(8)}: ${String(count).padStart(6)} (${pct.padStart(5)}%) ${bar}`);
    }
    console.log();
  }
}

compareSentiment(["technology", "politics", "finance", "health", "sports"]);
```

### Sentiment Trend Over Time

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getDailySentiment(topic, days = 14) {
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);
  const dailyData = [];

  for (let i = 0; i < days; i++) {
    const dayEnd = new Date(today.getTime() - i * 86400000);
    const dayStart = new Date(dayEnd.getTime() - 86400000);

    const counts = {};
    for (const polarity of ["positive", "negative", "neutral"]) {
      const params = new URLSearchParams({
        api_key: API_KEY,
        "topic.id": topic,
        "sentiment.overall.polarity": polarity,
        "published_at.start": dayStart.toISOString().split("T")[0],
        "published_at.end": dayEnd.toISOString().split("T")[0],
        "language.code": "en",
        per_page: "1",
      });

      const response = await fetch(`${BASE_URL}?${params}`);
      const data = await response.json();
      counts[polarity] = data.results?.length || 0;
    }

    dailyData.push({
      date: dayStart.toISOString().split("T")[0],
      ...counts,
    });
  }

  return dailyData.reverse();
}

const trend = await getDailySentiment("technology");

console.log("Daily sentiment trend for 'technology':\n");
console.log(`${"Date".padEnd(12)} ${"Pos".padStart(6)} ${"Neg".padStart(6)} ${"Neu".padStart(6)} ${"Ratio".padStart(8)}`);
console.log("-".repeat(42));

trend.forEach((day) => {
  const total = day.positive + day.negative + day.neutral || 1;
  const ratio = ((day.positive / total) * 100).toFixed(1);
  console.log(
    `${day.date.padEnd(12)} ${String(day.positive).padStart(6)} ` +
    `${String(day.negative).padStart(6)} ${String(day.neutral).padStart(6)} ` +
    `${(ratio + "%").padStart(8)}`
  );
});
```

---

## PHP

### Filter by Sentiment Polarity

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function getNewsBySentiment(string $polarity, ?string $topic = null, string $language = "en", int $perPage = 20): array
{
    global $apiKey, $baseUrl;

    $params = [
        "api_key"                    => $apiKey,
        "sentiment.overall.polarity" => $polarity,
        "language.code"              => $language,
        "per_page"                   => $perPage,
    ];
    if ($topic !== null) {
        $params["topic.id"] = $topic;
    }

    $url      = $baseUrl . "?" . http_build_query($params);
    $response = file_get_contents($url);
    return json_decode($response, true);
}

$data = getNewsBySentiment("positive", "technology");
echo "Found " . count($data["results"]) . " positive tech articles\n\n";

foreach ($data["results"] as $article) {
    $score = $article["sentiment"]["overall"]["score"];
    printf("  [%+.2f] %s\n", $score, $article["title"]);
}
```

### Sort by Sentiment Score

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$query = http_build_query([
    "api_key"       => $apiKey,
    "topic.id"      => "cryptocurrency",
    "sort.by"       => "sentiment.overall.score",
    "sort.order"    => "desc",
    "language.code" => "en",
    "per_page"      => 20,
]);

$response = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);

echo "Crypto news sorted by sentiment (most positive first):\n\n";

foreach ($response["results"] as $article) {
    $score    = $article["sentiment"]["overall"]["score"];
    $polarity = $article["sentiment"]["overall"]["polarity"];
    printf("  [%8s %+.2f] %s\n", $polarity, $score, $article["title"]);
    echo "    Source: {$article['source']['domain']}\n\n";
}
```

### Sentiment Comparison Across Topics

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$topics = ["technology", "politics", "finance", "health", "sports"];

echo "Sentiment distribution by topic:\n\n";

foreach ($topics as $topic) {
    $results = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "topic.id"                   => $topic,
            "sentiment.overall.polarity" => $polarity,
            "language.code"              => "en",
            "per_page"                   => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $results[$polarity] = count($data["results"] ?? []);
    }

    $total = array_sum($results) ?: 1;

    echo "  {$topic}:\n";
    foreach ($results as $polarity => $count) {
        $pct = $count / $total * 100;
        $bar = str_repeat("#", (int) ($pct / 2));
        printf("    %8s: %6d (%5.1f%%) %s\n", $polarity, $count, $pct, $bar);
    }
    echo "\n";
}
```

### Sentiment Trend Over Time

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function getDailySentiment(string $topic, int $days = 14, string $language = "en"): array
{
    global $apiKey, $baseUrl;

    $today     = new DateTimeImmutable("today", new DateTimeZone("UTC"));
    $dailyData = [];

    for ($i = 0; $i < $days; $i++) {
        $dayStart = $today->modify("-" . ($i + 1) . " days");
        $dayEnd   = $today->modify("-{$i} days");

        $counts = [];
        foreach (["positive", "negative", "neutral"] as $polarity) {
            $query = http_build_query([
                "api_key"                    => $apiKey,
                "topic.id"                   => $topic,
                "sentiment.overall.polarity" => $polarity,
                "published_at.start"         => $dayStart->format("Y-m-d"),
                "published_at.end"           => $dayEnd->format("Y-m-d"),
                "language.code"              => $language,
                "per_page"                   => 1,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            $counts[$polarity] = count($data["results"] ?? []);
        }

        $dailyData[] = [
            "date"     => $dayStart->format("Y-m-d"),
            "positive" => $counts["positive"],
            "negative" => $counts["negative"],
            "neutral"  => $counts["neutral"],
        ];
    }

    return array_reverse($dailyData);
}

$trend = getDailySentiment("technology");

echo "Daily sentiment trend for 'technology':\n\n";
printf("%-12s %6s %6s %6s %8s\n", "Date", "Pos", "Neg", "Neu", "Ratio");
echo str_repeat("-", 42) . "\n";

foreach ($trend as $day) {
    $total = $day["positive"] + $day["negative"] + $day["neutral"] ?: 1;
    $ratio = $day["positive"] / $total * 100;
    printf("%-12s %6d %6d %6d %7.1f%%\n",
        $day["date"], $day["positive"], $day["negative"], $day["neutral"], $ratio);
}
```
