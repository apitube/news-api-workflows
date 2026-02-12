# Entity Tracking — Code Examples

Detailed examples for tracking entities in news using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Track a Company

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def track_entity(name, language="en", per_page=20):
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": name,
        "language.code": language,
        "sort.by": "published_at",
        "sort.order": "desc",
        "per_page": per_page,
    })
    response.raise_for_status()
    return response.json()

data = track_entity("Google")
print(f"Found {len(data['results'])} articles mentioning Google\n")

for article in data["results"]:
    print(f"  [{article['published_at'][:10]}] {article['title']}")
    print(f"    {article['source']['domain']} — {article['href']}\n")
```

### Entity Sentiment Analysis

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def entity_sentiment_breakdown(entity_name):
    results = {}
    for polarity in ["positive", "negative", "neutral"]:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity_name,
            "sentiment.overall.polarity": polarity,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        results[polarity] = len(response.json().get("results", []))
    return results

companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"]

print("Entity sentiment breakdown:\n")
print(f"{'Company':<12} {'Positive':>10} {'Negative':>10} {'Neutral':>10} {'Score':>8}")
print("-" * 54)

for company in companies:
    s = entity_sentiment_breakdown(company)
    total = sum(s.values()) or 1
    score = (s["positive"] - s["negative"]) / total
    print(f"{company:<12} {s['positive']:>10} {s['negative']:>10} "
          f"{s['neutral']:>10} {score:>+7.2f}")
```

### Multi-Entity Monitoring with Alerts

```python
import requests
import time
from datetime import datetime

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

ENTITIES = [
    {"name": "Tesla", "param": "organization.name"},
    {"name": "Elon Musk", "param": "person.name"},
    {"name": "SpaceX", "param": "organization.name"},
]
POLL_INTERVAL = 300
NEGATIVE_THRESHOLD = 50

def check_entity(entity_name, entity_param):
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        entity_param: entity_name,
        "sentiment.overall.polarity": "negative",
        "published_at.start": datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z"),
        "language.code": "en",
        "per_page": 1,
    })
    response.raise_for_status()
    return len(response.json().get("results", []))

print("Entity monitoring started...\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]")

    for entity in ENTITIES:
        neg_count = check_entity(entity["name"], entity["param"])
        status = "ALERT" if neg_count >= NEGATIVE_THRESHOLD else "OK"
        print(f"  {entity['name']}: "
              f"{neg_count} negative articles [{status}]")

        if neg_count >= NEGATIVE_THRESHOLD:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                entity["param"]: entity["name"],
                "sentiment.overall.polarity": "negative",
                "sort.by": "published_at",
                "sort.order": "desc",
                "language.code": "en",
                "per_page": 3,
            })
            response.raise_for_status()
            for article in response.json()["results"]:
                print(f"    -> {article['title']}")
                print(f"       {article['source']['domain']}")

    print()
    time.sleep(POLL_INTERVAL)
```

### Entity Coverage by Source

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

def entity_coverage_by_source(entity_name, domains):
    coverage = {}
    for domain in domains:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity_name,
            "source.domain": domain,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        data = response.json()
        coverage[domain] = len(data.get("results", []))
    return coverage

domains = [
    "reuters.com", "bbc.com", "cnn.com",
    "nytimes.com", "theguardian.com", "bloomberg.com",
]

entity = "Apple"
coverage = entity_coverage_by_source(entity, domains)

print(f"Coverage of '{entity}' by source:\n")
max_count = max(coverage.values()) or 1

for domain, count in sorted(coverage.items(), key=lambda x: -x[1]):
    bar = "#" * int(count / max_count * 40)
    print(f"  {domain:<25} {count:>6} {bar}")
```

### Person Tracking with Date Range

```python
import requests
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

person = "Elon Musk"
days = 30
today = datetime.utcnow()
start = today - timedelta(days=days)

all_articles = []
page = 1

while True:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "person.name": person,
        "published_at.start": start.strftime("%Y-%m-%d"),
        "published_at.end": today.strftime("%Y-%m-%d"),
        "sort.by": "published_at",
        "sort.order": "desc",
        "language.code": "en",
        "per_page": 50,
        "page": page,
    })
    response.raise_for_status()
    data = response.json()
    articles = data.get("results", [])

    if not articles:
        break

    all_articles.extend(articles)
    print(f"Page {page}: {len(articles)} articles")

    if len(all_articles) >= 200:
        break
    page += 1

print(f"\nCollected {len(all_articles)} articles about {person} "
      f"in the last {days} days\n")

# Group by sentiment
from collections import Counter
sentiments = Counter(a["sentiment"]["overall"]["polarity"] for a in all_articles)
for polarity, count in sentiments.most_common():
    print(f"  {polarity}: {count}")
```

---

## JavaScript

### Track a Company

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function trackEntity(name, perPage = 20) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": name,
    "language.code": "en",
    "sort.by": "published_at",
    "sort.order": "desc",
    per_page: String(perPage),
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const data = await trackEntity("Google");
console.log(`Found ${data.results.length} articles mentioning Google\n`);

data.results.forEach((article) => {
  const date = article.published_at.slice(0, 10);
  console.log(`  [${date}] ${article.title}`);
  console.log(`    ${article.source.domain} — ${article.href}\n`);
});
```

### Entity Sentiment Analysis

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function entitySentimentBreakdown(entityName) {
  const results = {};

  for (const polarity of ["positive", "negative", "neutral"]) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": entityName,
      "sentiment.overall.polarity": polarity,
      "language.code": "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    results[polarity] = data.results?.length || 0;
  }

  return results;
}

const companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"];

console.log("Entity sentiment breakdown:\n");
console.log(
  `${"Company".padEnd(12)} ${"Positive".padStart(10)} ` +
  `${"Negative".padStart(10)} ${"Neutral".padStart(10)} ${"Score".padStart(8)}`
);
console.log("-".repeat(54));

for (const company of companies) {
  const s = await entitySentimentBreakdown(company);
  const total = s.positive + s.negative + s.neutral || 1;
  const score = (s.positive - s.negative) / total;
  const sign = score >= 0 ? "+" : "";
  console.log(
    `${company.padEnd(12)} ${String(s.positive).padStart(10)} ` +
    `${String(s.negative).padStart(10)} ${String(s.neutral).padStart(10)} ` +
    `${(sign + score.toFixed(2)).padStart(8)}`
  );
}
```

### Entity Coverage by Source

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function entityCoverageBySource(entityName, domains) {
  const coverage = {};

  for (const domain of domains) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": entityName,
      "source.domain": domain,
      "language.code": "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    coverage[domain] = data.results?.length || 0;
  }

  return coverage;
}

const domains = [
  "reuters.com", "bbc.com", "cnn.com",
  "nytimes.com", "theguardian.com", "bloomberg.com",
];

const coverage = await entityCoverageBySource("Apple", domains);
const maxCount = Math.max(...Object.values(coverage)) || 1;

console.log("Coverage of 'Apple' by source:\n");

Object.entries(coverage)
  .sort((a, b) => b[1] - a[1])
  .forEach(([domain, count]) => {
    const bar = "#".repeat(Math.round((count / maxCount) * 40));
    console.log(`  ${domain.padEnd(25)} ${String(count).padStart(6)} ${bar}`);
  });
```

---

## PHP

### Track a Company

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function trackEntity(string $name, int $perPage = 20): array
{
    global $apiKey, $baseUrl;

    $query = http_build_query([
        "api_key"           => $apiKey,
        "organization.name" => $name,
        "language.code"     => "en",
        "sort.by"           => "published_at",
        "sort.order"        => "desc",
        "per_page"          => $perPage,
    ]);

    return json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
}

$data = trackEntity("Google");
echo "Found " . count($data["results"]) . " articles mentioning Google\n\n";

foreach ($data["results"] as $article) {
    $date = substr($article["published_at"], 0, 10);
    echo "  [{$date}] {$article['title']}\n";
    echo "    {$article['source']['domain']} — {$article['href']}\n\n";
}
```

### Entity Sentiment Analysis

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function entitySentimentBreakdown(string $entityName): array
{
    global $apiKey, $baseUrl;

    $results = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "organization.name"          => $entityName,
            "sentiment.overall.polarity" => $polarity,
            "language.code"              => "en",
            "per_page"                   => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $results[$polarity] = count($data["results"] ?? []);
    }

    return $results;
}

$companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"];

echo "Entity sentiment breakdown:\n\n";
printf("%-12s %10s %10s %10s %8s\n", "Company", "Positive", "Negative", "Neutral", "Score");
echo str_repeat("-", 54) . "\n";

foreach ($companies as $company) {
    $s     = entitySentimentBreakdown($company);
    $total = array_sum($s) ?: 1;
    $score = ($s["positive"] - $s["negative"]) / $total;
    printf("%-12s %10d %10d %10d %+7.2f\n",
        $company, $s["positive"], $s["negative"], $s["neutral"], $score);
}
```

### Entity Coverage by Source

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function entityCoverageBySource(string $entityName, array $domains): array
{
    global $apiKey, $baseUrl;

    $coverage = [];
    foreach ($domains as $domain) {
        $query = http_build_query([
            "api_key"           => $apiKey,
            "organization.name" => $entityName,
            "source.domain"     => $domain,
            "language.code"     => "en",
            "per_page"          => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $coverage[$domain] = count($data["results"] ?? []);
    }

    return $coverage;
}

$domains = [
    "reuters.com", "bbc.com", "cnn.com",
    "nytimes.com", "theguardian.com", "bloomberg.com",
];

$coverage = entityCoverageBySource("Apple", $domains);
arsort($coverage);
$maxCount = max($coverage) ?: 1;

echo "Coverage of 'Apple' by source:\n\n";

foreach ($coverage as $domain => $count) {
    $bar = str_repeat("#", (int) round($count / $maxCount * 40));
    printf("  %-25s %6d %s\n", $domain, $count, $bar);
}
```
