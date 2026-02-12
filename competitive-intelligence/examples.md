# Competitive Intelligence — Code Examples

Detailed examples for building competitive intelligence dashboards using the APITube News API in **Python**, **JavaScript**, and **PHP**.

---

## Python

### Share of Voice Analysis

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

COMPETITORS = ["Tesla", "Rivian", "Lucid Motors", "BYD", "NIO"]

print("Share of Voice Analysis (Electric Vehicles)\n")

volumes = {}
for name in COMPETITORS:
    response = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": name,
        "language.code": "en",
        "per_page": 1,
    })
    response.raise_for_status()
    volumes[name] = len(response.json().get("results", []))

total = sum(volumes.values()) or 1

print(f"{'Company':<18} {'Articles':>10} {'Share':>8}")
print("-" * 38)

for name, count in sorted(volumes.items(), key=lambda x: -x[1]):
    share = count / total * 100
    bar = "█" * int(share / 2)
    print(f"  {name:<16} {count:>10} {share:>6.1f}% {bar}")
```

### Competitive Sentiment Scorecard

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

COMPETITORS = ["Apple", "Google", "Microsoft", "Amazon", "Meta"]

def get_sentiment_counts(entity_name):
    counts = {}
    for polarity in ["positive", "negative", "neutral"]:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity_name,
            "sentiment.overall.polarity": polarity,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        counts[polarity] = len(response.json().get("results", []))
    return counts

print("Competitive Sentiment Scorecard\n")
print(f"{'Company':<14} {'Positive':>10} {'Negative':>10} {'Neutral':>10} "
      f"{'Total':>10} {'Net Score':>10}")
print("-" * 68)

scorecard = []
for company in COMPETITORS:
    s = get_sentiment_counts(company)
    total = sum(s.values()) or 1
    net_score = (s["positive"] - s["negative"]) / total
    scorecard.append((company, s, total, net_score))

for company, s, total, net_score in sorted(scorecard, key=lambda x: -x[3]):
    print(f"  {company:<12} {s['positive']:>10} {s['negative']:>10} "
          f"{s['neutral']:>10} {total:>10} {net_score:>+9.3f}")
```

### Competitor Topic Coverage Matrix

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

companies = ["Apple", "Google", "Microsoft"]
topics = ["artificial_intelligence", "cybersecurity", "cloud_computing",
          "electric_vehicles", "space"]

print("Competitor × Topic Coverage Matrix\n")

# Header
header = f"{'Company':<14}" + "".join(f" {t[:12]:>12}" for t in topics)
print(header)
print("-" * len(header))

for company in companies:
    row = f"  {company:<12}"
    for topic in topics:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": company,
            "topic.id": topic,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        count = len(response.json().get("results", []))
        row += f" {count:>12}"
    print(row)
```

### PR Crisis Detection Dashboard

```python
import requests
import time
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

COMPANIES = ["Tesla", "Apple", "Google", "Microsoft", "Amazon"]
POLL_INTERVAL = 600
CRISIS_THRESHOLD = 2.0  # negative-to-positive ratio

def get_today_sentiment(entity_name):
    today = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    counts = {}
    for polarity in ["positive", "negative"]:
        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "organization.name": entity_name,
            "sentiment.overall.polarity": polarity,
            "published_at.start": today,
            "language.code": "en",
            "per_page": 1,
        })
        response.raise_for_status()
        counts[polarity] = len(response.json().get("results", []))
    return counts

print("PR Crisis Detection Dashboard\n")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}]\n")

    for company in COMPANIES:
        s = get_today_sentiment(company)
        ratio = s["negative"] / max(s["positive"], 1)

        if ratio >= CRISIS_THRESHOLD:
            status = "🔴 CRISIS"
        elif ratio >= 1.0:
            status = "🟡 WARNING"
        else:
            status = "🟢 OK"

        print(f"  {company:<14} pos={s['positive']:<5} neg={s['negative']:<5} "
              f"ratio={ratio:.2f}  {status}")

        if ratio >= CRISIS_THRESHOLD:
            response = requests.get(BASE_URL, params={
                "api_key": API_KEY,
                "organization.name": company,
                "sentiment.overall.polarity": "negative",
                "sort.by": "published_at",
                "sort.order": "desc",
                "language.code": "en",
                "per_page": 3,
            })
            response.raise_for_status()
            for article in response.json()["results"]:
                print(f"    -> [{article['source']['domain']}] {article['title']}")

    print()
    time.sleep(POLL_INTERVAL)
```

### Competitive Coverage from Top-Tier Sources

```python
import requests

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.apitube.io/v1/news/everything"

TOP_SOURCES = "reuters.com,bloomberg.com,ft.com,wsj.com,nytimes.com"
COMPETITORS = ["Apple", "Google", "Microsoft", "Amazon"]

print("Coverage from top-tier financial sources:\n")
print(f"{'Company':<14} {'Total':>8} {'Positive':>10} {'Negative':>10} {'Ratio':>8}")
print("-" * 54)

for company in COMPETITORS:
    total_resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "source.domain": TOP_SOURCES,
        "language.code": "en",
        "per_page": 1,
    })
    total_resp.raise_for_status()
    total = len(total_resp.json().get("results", []))

    pos_resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "source.domain": TOP_SOURCES,
        "sentiment.overall.polarity": "positive",
        "language.code": "en",
        "per_page": 1,
    })
    pos_resp.raise_for_status()
    positive = len(pos_resp.json().get("results", []))

    neg_resp = requests.get(BASE_URL, params={
        "api_key": API_KEY,
        "organization.name": company,
        "source.domain": TOP_SOURCES,
        "sentiment.overall.polarity": "negative",
        "language.code": "en",
        "per_page": 1,
    })
    neg_resp.raise_for_status()
    negative = len(neg_resp.json().get("results", []))

    ratio = positive / max(negative, 1)
    print(f"  {company:<12} {total:>8} {positive:>10} {negative:>10} {ratio:>7.2f}")
```

---

## JavaScript

### Share of Voice Analysis

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const COMPETITORS = ["Tesla", "Rivian", "Lucid Motors", "BYD", "NIO"];

console.log("Share of Voice Analysis (Electric Vehicles)\n");

const volumes = {};

for (const name of COMPETITORS) {
  const params = new URLSearchParams({
    api_key: API_KEY,
    "organization.name": name,
    "language.code": "en",
    per_page: "1",
  });

  const response = await fetch(`${BASE_URL}?${params}`);
  const data = await response.json();
  volumes[name] = data.results?.length || 0;
}

const total = Object.values(volumes).reduce((a, b) => a + b, 0) || 1;

console.log(`${"Company".padEnd(18)} ${"Articles".padStart(10)} ${"Share".padStart(8)}`);
console.log("-".repeat(38));

Object.entries(volumes)
  .sort((a, b) => b[1] - a[1])
  .forEach(([name, count]) => {
    const share = ((count / total) * 100).toFixed(1);
    const bar = "#".repeat(Math.round(share / 2));
    console.log(
      `  ${name.padEnd(16)} ${String(count).padStart(10)} ${(share + "%").padStart(7)} ${bar}`
    );
  });
```

### Competitive Sentiment Scorecard

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

async function getSentimentCounts(entityName) {
  const counts = {};

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
    counts[polarity] = data.results?.length || 0;
  }

  return counts;
}

const companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"];

console.log("Competitive Sentiment Scorecard\n");
console.log(
  `${"Company".padEnd(14)} ${"Positive".padStart(10)} ${"Negative".padStart(10)} ` +
  `${"Neutral".padStart(10)} ${"Total".padStart(10)} ${"Net Score".padStart(10)}`
);
console.log("-".repeat(68));

const scorecard = [];

for (const company of companies) {
  const s = await getSentimentCounts(company);
  const total = s.positive + s.negative + s.neutral || 1;
  const netScore = (s.positive - s.negative) / total;
  scorecard.push({ company, s, total, netScore });
}

scorecard
  .sort((a, b) => b.netScore - a.netScore)
  .forEach(({ company, s, total, netScore }) => {
    const sign = netScore >= 0 ? "+" : "";
    console.log(
      `  ${company.padEnd(12)} ${String(s.positive).padStart(10)} ` +
      `${String(s.negative).padStart(10)} ${String(s.neutral).padStart(10)} ` +
      `${String(total).padStart(10)} ${(sign + netScore.toFixed(3)).padStart(10)}`
    );
  });
```

### Competitor Topic Coverage Matrix

```javascript
const API_KEY = "YOUR_API_KEY";
const BASE_URL = "https://api.apitube.io/v1/news/everything";

const companies = ["Apple", "Google", "Microsoft"];
const topics = [
  "artificial_intelligence", "cybersecurity", "cloud_computing",
  "electric_vehicles", "space",
];

console.log("Competitor × Topic Coverage Matrix\n");

const header =
  "Company".padEnd(14) +
  topics.map((t) => t.slice(0, 12).padStart(12)).join(" ");
console.log(header);
console.log("-".repeat(header.length));

for (const company of companies) {
  let row = `  ${company.padEnd(12)}`;

  for (const topic of topics) {
    const params = new URLSearchParams({
      api_key: API_KEY,
      "organization.name": company,
      "topic.id": topic,
      "language.code": "en",
      per_page: "1",
    });

    const response = await fetch(`${BASE_URL}?${params}`);
    const data = await response.json();
    const count = data.results?.length || 0;
    row += ` ${String(count).padStart(12)}`;
  }

  console.log(row);
}
```

---

## PHP

### Share of Voice Analysis

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$competitors = ["Tesla", "Rivian", "Lucid Motors", "BYD", "NIO"];

echo "Share of Voice Analysis (Electric Vehicles)\n\n";

$volumes = [];
foreach ($competitors as $name) {
    $query = http_build_query([
        "api_key"           => $apiKey,
        "organization.name" => $name,
        "language.code"     => "en",
        "per_page"          => 1,
    ]);

    $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
    $volumes[$name] = count($data["results"] ?? []);
}

arsort($volumes);
$total = array_sum($volumes) ?: 1;

printf("%-18s %10s %8s\n", "Company", "Articles", "Share");
echo str_repeat("-", 38) . "\n";

foreach ($volumes as $name => $count) {
    $share = $count / $total * 100;
    $bar   = str_repeat("#", (int) round($share / 2));
    printf("  %-16s %10d %6.1f%% %s\n", $name, $count, $share, $bar);
}
```

### Competitive Sentiment Scorecard

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

function getSentimentCounts(string $entityName): array
{
    global $apiKey, $baseUrl;

    $counts = [];
    foreach (["positive", "negative", "neutral"] as $polarity) {
        $query = http_build_query([
            "api_key"                    => $apiKey,
            "organization.name"          => $entityName,
            "sentiment.overall.polarity" => $polarity,
            "language.code"              => "en",
            "per_page"                   => 1,
        ]);

        $data = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $counts[$polarity] = count($data["results"] ?? []);
    }

    return $counts;
}

$companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"];

echo "Competitive Sentiment Scorecard\n\n";
printf("%-14s %10s %10s %10s %10s %10s\n",
    "Company", "Positive", "Negative", "Neutral", "Total", "Net Score");
echo str_repeat("-", 68) . "\n";

$scorecard = [];
foreach ($companies as $company) {
    $s     = getSentimentCounts($company);
    $total = array_sum($s) ?: 1;
    $net   = ($s["positive"] - $s["negative"]) / $total;
    $scorecard[] = compact("company", "s", "total", "net");
}

usort($scorecard, fn($a, $b) => $b["net"] <=> $a["net"]);

foreach ($scorecard as $row) {
    printf("  %-12s %10d %10d %10d %10d %+9.3f\n",
        $row["company"],
        $row["s"]["positive"], $row["s"]["negative"], $row["s"]["neutral"],
        $row["total"], $row["net"]);
}
```

### Competitor Topic Coverage Matrix

```php
<?php

$apiKey  = "YOUR_API_KEY";
$baseUrl = "https://api.apitube.io/v1/news/everything";

$companies = ["Apple", "Google", "Microsoft"];
$topics    = ["artificial_intelligence", "cybersecurity", "cloud_computing",
              "electric_vehicles", "space"];

echo "Competitor × Topic Coverage Matrix\n\n";

$header = sprintf("%-14s", "Company");
foreach ($topics as $t) {
    $header .= sprintf(" %12s", substr($t, 0, 12));
}
echo $header . "\n";
echo str_repeat("-", strlen($header)) . "\n";

foreach ($companies as $company) {
    $row = sprintf("  %-12s", $company);

    foreach ($topics as $topic) {
        $query = http_build_query([
            "api_key"           => $apiKey,
            "organization.name" => $company,
            "topic.id"          => $topic,
            "language.code"     => "en",
            "per_page"          => 1,
        ]);

        $data  = json_decode(file_get_contents("{$baseUrl}?{$query}"), true);
        $count = count($data["results"] ?? []);
        $row  .= sprintf(" %12d", $count);
    }

    echo $row . "\n";
}
```
