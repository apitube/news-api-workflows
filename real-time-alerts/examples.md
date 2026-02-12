# Real-Time Alerts — Code Examples

Detailed examples for building real-time alerting pipelines using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Breaking News Detector

```python
import requests
import time
from datetime import datetime

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"
POLL_INTERVAL = 60

seen_ids = set()

def poll_breaking_news():
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "is_breaking": "1",
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 20,
    })
    response.raise_for_status()
    return response.json()

print("Breaking news detector started. Polling every 60s...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    data = poll_breaking_news()

    new_articles = []
    for article in data["results"]:
        article_id = article["href"]
        if article_id not in seen_ids:
            seen_ids.add(article_id)
            new_articles.append(article)

    if new_articles:
        print(f"[{timestamp}] Found {len(new_articles)} NEW breaking articles:")
        for article in new_articles:
            print(f"  [ALERT] {article['title']}")
            print(f"    Source: {article['source']['domain']}")
            print(f"    URL: {article['href']}\n")
    else:
        print(f"[{timestamp}] No new breaking news.")

    time.sleep(POLL_INTERVAL)
```

### Multi-Signal Alert Engine

```python
import requests
import time
from datetime import datetime, timedelta
from collections import deque

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ENTITY_NAME = "Tesla"
POLL_INTERVAL = 300
WINDOW_SIZE = 12

volume_history = deque(maxlen=WINDOW_SIZE)
sentiment_history = deque(maxlen=WINDOW_SIZE)
breaking_history = deque(maxlen=WINDOW_SIZE)

def fetch_metrics():
    now = datetime.utcnow()
    start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    volume_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": ENTITY_NAME,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    volume = len(volume_response.json().get("results", []))

    neg_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": ENTITY_NAME,
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    negative_count = len(neg_response.json().get("results", []))

    breaking_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": ENTITY_NAME,
        "is_breaking": "1",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    breaking_count = len(breaking_response.json().get("results", []))

    sentiment_ratio = negative_count / volume if volume > 0 else 0

    return volume, sentiment_ratio, breaking_count

print(f"Multi-signal alert engine for {ENTITY_NAME} started...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    volume, sentiment_ratio, breaking_count = fetch_metrics()

    volume_history.append(volume)
    sentiment_history.append(sentiment_ratio)
    breaking_history.append(breaking_count)

    print(f"[{timestamp}]")
    print(f"  Volume (last 1h): {volume}")
    print(f"  Negative ratio: {sentiment_ratio:.2f}")
    print(f"  Breaking count: {breaking_count}")

    if len(volume_history) >= 6:
        avg_volume = sum(volume_history) / len(volume_history)
        avg_sentiment = sum(sentiment_history) / len(sentiment_history)
        avg_breaking = sum(breaking_history) / len(breaking_history)

        volume_spike = volume > avg_volume * 2.5
        sentiment_spike = sentiment_ratio > avg_sentiment * 2.0 and sentiment_ratio > 0.3
        breaking_spike = breaking_count > avg_breaking * 2.0

        alerts = []
        if volume_spike:
            alerts.append("VOLUME SPIKE")
        if sentiment_spike:
            alerts.append("SENTIMENT SPIKE")
        if breaking_spike:
            alerts.append("BREAKING SPIKE")

        if alerts:
            print(f"  >>> ALERT: {', '.join(alerts)}")
        else:
            print(f"  Status: OK")

    print()
    time.sleep(POLL_INTERVAL)
```

### Entity Mention Velocity Tracker

```python
import requests
import time
from datetime import datetime, timedelta
from collections import deque

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ENTITY_NAME = "Apple"
POLL_INTERVAL = 300
WINDOW_HOURS = 6

hourly_counts = deque(maxlen=WINDOW_HOURS)

def get_mentions_last_hour():
    now = datetime.utcnow()
    start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": ENTITY_NAME,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    response.raise_for_status()
    return len(response.json().get("results", []))

print(f"Entity mention velocity tracker for {ENTITY_NAME} started...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mentions = get_mentions_last_hour()
    hourly_counts.append(mentions)

    print(f"[{timestamp}] Mentions in last hour: {mentions}")

    if len(hourly_counts) >= 3:
        baseline = sum(list(hourly_counts)[:-1]) / (len(hourly_counts) - 1)
        velocity = mentions / baseline if baseline > 0 else 0

        print(f"  Baseline: {baseline:.1f} mentions/hour")
        print(f"  Velocity: {velocity:.2f}x")

        if velocity > 2.0:
            print(f"  >>> ALERT: Mention velocity exceeds 2x baseline!")

            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": ENTITY_NAME,
                "published_at.start": (datetime.utcnow() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sort.by": "published_at",
                "sort.order": "desc",
                "language.code": "en",
                "per_page": 5,
            })

            for article in response.json()["results"]:
                print(f"    - {article['title']}")
                print(f"      {article['source']['domain']}")

    print()
    time.sleep(POLL_INTERVAL)
```

### Sentiment Anomaly Detector

```python
import requests
import time
from datetime import datetime, timedelta
from collections import deque
import statistics

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

TOPIC_ID = "technology"
POLL_INTERVAL = 3600
WINDOW_SIZE = 168

hourly_sentiment_scores = deque(maxlen=WINDOW_SIZE)

def get_hourly_sentiment():
    now = datetime.utcnow()
    start = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "topic.id": TOPIC_ID,
        "published_at.start": start,
        "language.code": "en",
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": 100,
    })
    response.raise_for_status()

    articles = response.json()["results"]
    if not articles:
        return None

    scores = [a["sentiment"]["overall"]["score"] for a in articles]
    return sum(scores) / len(scores)

print(f"Sentiment anomaly detector for topic '{TOPIC_ID}' started...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_score = get_hourly_sentiment()

    if current_score is not None:
        hourly_sentiment_scores.append(current_score)

        print(f"[{timestamp}] Current hourly sentiment: {current_score:.3f}")

        if len(hourly_sentiment_scores) >= 24:
            mean = statistics.mean(hourly_sentiment_scores)
            stdev = statistics.stdev(hourly_sentiment_scores)
            z_score = (current_score - mean) / stdev if stdev > 0 else 0

            print(f"  7-day mean: {mean:.3f}")
            print(f"  Std dev: {stdev:.3f}")
            print(f"  Z-score: {z_score:.2f}")

            if abs(z_score) > 2.0:
                direction = "NEGATIVE" if z_score < 0 else "POSITIVE"
                print(f"  >>> ANOMALY DETECTED: {direction} sentiment spike!")
                print(f"      Current sentiment deviates {abs(z_score):.1f} standard deviations from baseline.")
        else:
            print(f"  Building baseline... ({len(hourly_sentiment_scores)}/24 hours)")
    else:
        print(f"[{timestamp}] No articles in last hour.")

    print()
    time.sleep(POLL_INTERVAL)
```

### Composite Crisis Score Dashboard

```python
import requests
import time
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ENTITIES = [
    {"name": "Tesla"},
    {"name": "Apple"},
    {"name": "Microsoft"},
]
POLL_INTERVAL = 600

def calculate_crisis_score(entity_name):
    now = datetime.utcnow()
    start = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    total_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity_name,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    total = len(total_response.json().get("results", []))

    if total == 0:
        return 0.0, {}

    neg_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity_name,
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    negative = len(neg_response.json().get("results", []))

    breaking_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity_name,
        "is_breaking": "1",
        "published_at.start": start,
        "language.code": "en",
        "per_page": 1,
    })
    breaking = len(breaking_response.json().get("results", []))

    articles_response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": entity_name,
        "published_at.start": start,
        "language.code": "en",
        "per_page": 100,
    })
    articles = articles_response.json()["results"]
    sources = len(set(a["source"]["domain"] for a in articles))

    negative_ratio = negative / total
    breaking_ratio = breaking / total
    mention_velocity = total / 6
    source_diversity = sources

    crisis_score = (
        negative_ratio * 40 +
        breaking_ratio * 30 +
        min(mention_velocity / 10, 1.0) * 20 +
        min(source_diversity / 20, 1.0) * 10
    )

    metrics = {
        "total": total,
        "negative": negative,
        "breaking": breaking,
        "sources": sources,
        "negative_ratio": negative_ratio,
        "mention_velocity": mention_velocity,
    }

    return crisis_score, metrics

print("Composite Crisis Score Dashboard\n")
print("=" * 80)

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}]\n")

    scores = []
    for entity in ENTITIES:
        score, metrics = calculate_crisis_score(entity["name"])
        scores.append((entity["name"], score, metrics))

    scores.sort(key=lambda x: -x[1])

    for name, score, m in scores:
        status = "CRITICAL" if score > 60 else "WARNING" if score > 40 else "OK"
        print(f"{name:<15} Crisis Score: {score:>5.1f} [{status}]")
        print(f"  Total mentions: {m['total']:>4} | Negative: {m['negative']:>3} ({m['negative_ratio']*100:.0f}%)")
        print(f"  Breaking: {m['breaking']:>4} | Sources: {m['sources']:>3} | Velocity: {m['mention_velocity']:.1f}/h")
        print()

    print("=" * 80)
    time.sleep(POLL_INTERVAL)
```

---

## JavaScript

### Breaking News Detector

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";
const POLL_INTERVAL = 60000;

const seenIds = new Set();

async function pollBreakingNews() {
  const params = new URLSearchParams({
    api_key: API_KEY,
    is_breaking: "1",
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "20",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

console.log("Breaking news detector started. Polling every 60s...\n");

setInterval(async () => {
  const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  const data = await pollBreakingNews();

  const newArticles = [];
  for (const article of data.results) {
    const articleId = article.href;
    if (!seenIds.has(articleId)) {
      seenIds.add(articleId);
      newArticles.push(article);
    }
  }

  if (newArticles.length > 0) {
    console.log(`[${timestamp}] Found ${newArticles.length} NEW breaking articles:`);
    for (const article of newArticles) {
      console.log(`  [ALERT] ${article.title}`);
      console.log(`    Source: ${article.source.domain}`);
      console.log(`    URL: ${article.href}\n`);
    }
  } else {
    console.log(`[${timestamp}] No new breaking news.`);
  }
}, POLL_INTERVAL);
```

### Multi-Signal Alert Engine

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const ENTITY_NAME = "Tesla";
const POLL_INTERVAL = 300000;
const WINDOW_SIZE = 12;

const volumeHistory = [];
const sentimentHistory = [];
const breakingHistory = [];

async function fetchMetrics() {
  const now = new Date();
  const start = new Date(now - 3600000).toISOString();

  const [volumeRes, negRes, breakingRes] = await Promise.all([
    fetch(
      `${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "organization.name": ENTITY_NAME,
        "published_at.start": start,
        "language.code": "en",
        per_page: "1",
      })}`
    ),
    fetch(
      `${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "organization.name": ENTITY_NAME,
        "sentiment.overall.polarity": "negative",
        "published_at.start": start,
        "language.code": "en",
        per_page: "1",
      })}`
    ),
    fetch(
      `${BASE_URL}?${new URLSearchParams({
        api_key: API_KEY,
        "organization.name": ENTITY_NAME,
        is_breaking: "1",
        "published_at.start": start,
        "language.code": "en",
        per_page: "1",
      })}`
    ),
  ]);

  const volume = (await volumeRes.json()).results?.length || 0;
  const negativeCount = (await negRes.json()).results?.length || 0;
  const breakingCount = (await breakingRes.json()).results?.length || 0;

  const sentimentRatio = volume > 0 ? negativeCount / volume : 0;

  return { volume, sentimentRatio, breakingCount };
}

console.log(`Multi-signal alert engine for ${ENTITY_NAME} started...\n`);

setInterval(async () => {
  const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  const { volume, sentimentRatio, breakingCount } = await fetchMetrics();

  volumeHistory.push(volume);
  sentimentHistory.push(sentimentRatio);
  breakingHistory.push(breakingCount);

  if (volumeHistory.length > WINDOW_SIZE) volumeHistory.shift();
  if (sentimentHistory.length > WINDOW_SIZE) sentimentHistory.shift();
  if (breakingHistory.length > WINDOW_SIZE) breakingHistory.shift();

  console.log(`[${timestamp}]`);
  console.log(`  Volume (last 1h): ${volume}`);
  console.log(`  Negative ratio: ${sentimentRatio.toFixed(2)}`);
  console.log(`  Breaking count: ${breakingCount}`);

  if (volumeHistory.length >= 6) {
    const avgVolume = volumeHistory.reduce((a, b) => a + b) / volumeHistory.length;
    const avgSentiment = sentimentHistory.reduce((a, b) => a + b) / sentimentHistory.length;
    const avgBreaking = breakingHistory.reduce((a, b) => a + b) / breakingHistory.length;

    const volumeSpike = volume > avgVolume * 2.5;
    const sentimentSpike = sentimentRatio > avgSentiment * 2.0 && sentimentRatio > 0.3;
    const breakingSpike = breakingCount > avgBreaking * 2.0;

    const alerts = [];
    if (volumeSpike) alerts.push("VOLUME SPIKE");
    if (sentimentSpike) alerts.push("SENTIMENT SPIKE");
    if (breakingSpike) alerts.push("BREAKING SPIKE");

    if (alerts.length > 0) {
      console.log(`  >>> ALERT: ${alerts.join(", ")}`);
    } else {
      console.log(`  Status: OK`);
    }
  }

  console.log();
}, POLL_INTERVAL);
```

### Sentiment Anomaly Detector

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const TOPIC_ID = "technology";
const POLL_INTERVAL = 3600000;
const WINDOW_SIZE = 168;

const hourlySentimentScores = [];

async function getHourlySentiment() {
  const now = new Date();
  const start = new Date(now - 3600000).toISOString();

  const params = new URLSearchParams({
    api_key: API_KEY,
    "topic.id": TOPIC_ID,
    "published_at.start": start,
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: "100",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();

  if (data.results.length === 0) return null;

  const scores = data.results.map((a) => a.sentiment.overall.score);
  return scores.reduce((a, b) => a + b) / scores.length;
}

function calculateStdDev(arr, mean) {
  const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
  return Math.sqrt(variance);
}

console.log(`Sentiment anomaly detector for topic '${TOPIC_ID}' started...\n`);

setInterval(async () => {
  const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  const currentScore = await getHourlySentiment();

  if (currentScore !== null) {
    hourlySentimentScores.push(currentScore);
    if (hourlySentimentScores.length > WINDOW_SIZE) hourlySentimentScores.shift();

    console.log(`[${timestamp}] Current hourly sentiment: ${currentScore.toFixed(3)}`);

    if (hourlySentimentScores.length >= 24) {
      const mean =
        hourlySentimentScores.reduce((a, b) => a + b) / hourlySentimentScores.length;
      const stdev = calculateStdDev(hourlySentimentScores, mean);
      const zScore = stdev > 0 ? (currentScore - mean) / stdev : 0;

      console.log(`  7-day mean: ${mean.toFixed(3)}`);
      console.log(`  Std dev: ${stdev.toFixed(3)}`);
      console.log(`  Z-score: ${zScore.toFixed(2)}`);

      if (Math.abs(zScore) > 2.0) {
        const direction = zScore < 0 ? "NEGATIVE" : "POSITIVE";
        console.log(`  >>> ANOMALY DETECTED: ${direction} sentiment spike!`);
        console.log(
          `      Current sentiment deviates ${Math.abs(zScore).toFixed(1)} standard deviations from baseline.`
        );
      }
    } else {
      console.log(`  Building baseline... (${hourlySentimentScores.length}/24 hours)`);
    }
  } else {
    console.log(`[${timestamp}] No articles in last hour.`);
  }

  console.log();
}, POLL_INTERVAL);
```

---

## PHP

### Breaking News Detector

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";
$pollInterval = 60;

$seenIds = [];

function pollBreakingNews(): array
{
    global $apiKey, $baseUrl;

    $query = http_build_query([
        "api_key"       => $apiKey,
        "is_breaking"   => "1",
        "language.code" => "en",
        "sort.by"       => "published_at",
        "sort.order"    => "desc",
        "per_page"      => 20,
    ]);

    return json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
}

echo "Breaking news detector started. Polling every 60s...\n\n";

while (true) {
    $timestamp = gmdate("Y-m-d H:i:s");
    $data = pollBreakingNews();

    $newArticles = [];
    foreach ($data["results"] as $article) {
        $articleId = $article["href"];
        if (!in_array($articleId, $seenIds, true)) {
            $seenIds[] = $articleId;
            $newArticles[] = $article;
        }
    }

    if (count($newArticles) > 0) {
        echo "[{$timestamp}] Found " . count($newArticles) . " NEW breaking articles:\n";
        foreach ($newArticles as $article) {
            echo "  [ALERT] {$article['title']}\n";
            echo "    Source: {$article['source']['domain']}\n";
            echo "    URL: {$article['href']}\n\n";
        }
    } else {
        echo "[{$timestamp}] No new breaking news.\n";
    }

    sleep($pollInterval);
}
```

### Entity Mention Velocity Tracker

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$entityName = "Apple";
$pollInterval = 300;
$windowHours = 6;

$hourlyCounts = [];

function getMentionsLastHour(): int
{
    global $apiKey, $baseUrl, $entityName;

    $start = gmdate("Y-m-d\TH:i:s\Z", time() - 3600);

    $query = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $entityName,
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    return count($data["results"] ?? []);
}

echo "Entity mention velocity tracker for {$entityName} started...\n\n";

while (true) {
    $timestamp = gmdate("Y-m-d H:i:s");
    $mentions = getMentionsLastHour();
    $hourlyCounts[] = $mentions;

    if (count($hourlyCounts) > $windowHours) {
        array_shift($hourlyCounts);
    }

    echo "[{$timestamp}] Mentions in last hour: {$mentions}\n";

    if (count($hourlyCounts) >= 3) {
        $baseline = array_sum(array_slice($hourlyCounts, 0, -1)) / (count($hourlyCounts) - 1);
        $velocity = $baseline > 0 ? $mentions / $baseline : 0;

        echo "  Baseline: " . number_format($baseline, 1) . " mentions/hour\n";
        echo "  Velocity: " . number_format($velocity, 2) . "x\n";

        if ($velocity > 2.0) {
            echo "  >>> ALERT: Mention velocity exceeds 2x baseline!\n";

            $start = gmdate("Y-m-d\TH:i:s\Z", time() - 3600);
            $query = http_build_query([
                "api_key"            => $apiKey,
                "organization.name"  => $entityName,
                "published_at.start" => $start,
                "sort.by"            => "published_at",
                "sort.order"         => "desc",
                "language.code"      => "en",
                "per_page"           => 5,
            ]);

            $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
            foreach ($data["results"] as $article) {
                echo "    - {$article['title']}\n";
                echo "      {$article['source']['domain']}\n";
            }
        }
    }

    echo "\n";
    sleep($pollInterval);
}
```

### Composite Crisis Score Dashboard

```php
<?php

$apiKey = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$entities = [
    ["name" => "Tesla"],
    ["name" => "Apple"],
    ["name" => "Microsoft"],
];
$pollInterval = 600;

function calculateCrisisScore(string $entityName): array
{
    global $apiKey, $baseUrl;

    $start = gmdate("Y-m-d\TH:i:s\Z", time() - 21600);

    $totalQuery = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $entityName,
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);
    $totalData = json_decode(file_get_contents("{$baseUrl}?{$totalQuery}"), true);
    $total = count($totalData["results"] ?? []);

    if ($total === 0) {
        return [0.0, []];
    }

    $negQuery = http_build_query([
        "api_key"                    => $apiKey,
        "organization.name"          => $entityName,
        "sentiment.overall.polarity" => "negative",
        "published_at.start"         => $start,
        "language.code"              => "en",
        "per_page"                   => 1,
    ]);
    $negData = json_decode(file_get_contents("{$baseUrl}?{$negQuery}"), true);
    $negative = count($negData["results"] ?? []);

    $breakingQuery = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $entityName,
        "is_breaking"        => "1",
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 1,
    ]);
    $breakingData = json_decode(file_get_contents("{$baseUrl}?{$breakingQuery}"), true);
    $breaking = count($breakingData["results"] ?? []);

    $articlesQuery = http_build_query([
        "api_key"            => $apiKey,
        "organization.name"  => $entityName,
        "published_at.start" => $start,
        "language.code"      => "en",
        "per_page"           => 100,
    ]);
    $articlesData = json_decode(file_get_contents("{$baseUrl}?{$articlesQuery}"), true);
    $sources = count(array_unique(array_column($articlesData["results"], "source")));

    $negativeRatio = $negative / $total;
    $breakingRatio = $breaking / $total;
    $mentionVelocity = $total / 6;
    $sourceDiversity = $sources;

    $crisisScore = (
        $negativeRatio * 40 +
        $breakingRatio * 30 +
        min($mentionVelocity / 10, 1.0) * 20 +
        min($sourceDiversity / 20, 1.0) * 10
    );

    $metrics = [
        "total"           => $total,
        "negative"        => $negative,
        "breaking"        => $breaking,
        "sources"         => $sources,
        "negative_ratio"  => $negativeRatio,
        "mention_velocity" => $mentionVelocity,
    ];

    return [$crisisScore, $metrics];
}

echo "Composite Crisis Score Dashboard\n\n";
echo str_repeat("=", 80) . "\n";

while (true) {
    $timestamp = gmdate("Y-m-d H:i:s");
    echo "\n[{$timestamp}]\n\n";

    $scores = [];
    foreach ($entities as $entity) {
        [$score, $metrics] = calculateCrisisScore($entity["name"]);
        $scores[] = [$entity["name"], $score, $metrics];
    }

    usort($scores, fn($a, $b) => $b[1] <=> $a[1]);

    foreach ($scores as [$name, $score, $m]) {
        $status = $score > 60 ? "CRITICAL" : ($score > 40 ? "WARNING" : "OK");
        printf("%-15s Crisis Score: %5.1f [%s]\n", $name, $score, $status);
        printf("  Total mentions: %4d | Negative: %3d (%.0f%%)\n",
            $m["total"], $m["negative"], $m["negative_ratio"] * 100);
        printf("  Breaking: %4d | Sources: %3d | Velocity: %.1f/h\n\n",
            $m["breaking"], $m["sources"], $m["mention_velocity"]);
    }

    echo str_repeat("=", 80) . "\n";
    sleep($pollInterval);
}
```
